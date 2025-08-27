# main.py
# -----------------------------------------------------------------------------
# このバージョンの追加点（レイアウト変更なし）
# - GUI（Tk）をクラス ControlWindow で起動し、閉じるまで常駐
# - 起動/終了ログの整合（起動=main() / 終了=GUIクローズ時）
# - CaptureWorker を実装（開始/停止ボタンが機能）
# - 停止時に最終キャプチャを debug_dir へ保存（_stopped_last.png）
# -----------------------------------------------------------------------------

import sys
import os
import time
import threading
from typing import Dict, Any, Optional, Tuple, List, Callable
from pathlib import Path

try:
    import tomllib  # noqa: F401  # Python 3.11+
except Exception:
    raise RuntimeError("Python 3.11 以降が必要です（tomllib が標準搭載）。")

import cv2
import numpy as np

try:
    import mss  # スクリーンキャプチャ
except ImportError:
    raise RuntimeError("mss が必要です: pip install mss")

try:
    import pyautogui  # クリック
    pyautogui.FAILSAFE = False
except ImportError:
    raise RuntimeError("pyautogui が必要です: pip install pyautogui")

try:
    import win32gui  # ウィンドウ矩形取得（任意）
except ImportError:
    win32gui = None

import tkinter as tk
from tkinter import messagebox

# ログ（モジュール化）
from modules.app_logger import log_app_start, log_app_exit, log_info, log_error
# 設定ローダ（モジュール化）
from modules.config_loader import load_config

# スクリプトのあるディレクトリ（相対パス解決に使用）
ROOT = Path(__file__).resolve().parent


# ==========================
# DPI（Windows向け）
# ==========================
def set_dpi_awareness_windows():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)  # SYSTEM_DPI_AWARE
    except Exception:
        pass


# ==========================
# ウィンドウ矩形の取得（任意）
# ==========================
def find_window_rect(title: str) -> Optional[Tuple[int, int, int, int]]:
    """
    指定タイトルを含む最前面ウィンドウの矩形 (left, top, right, bottom) を返す。
    取れなければ None。
    """
    if not win32gui:
        return None

    target_hwnd = None

    def enum_handler(hwnd, _):
        nonlocal target_hwnd
        if win32gui.IsWindowVisible(hwnd):
            text = win32gui.GetWindowText(hwnd)
            if title.lower() in text.lower():
                target_hwnd = hwnd
                return False
        return True

    win32gui.EnumWindows(enum_handler, None)
    if target_hwnd:
        rect = win32gui.GetWindowRect(target_hwnd)
        return rect[0], rect[1], rect[2], rect[3]
    return None


# ==========================
# サブ領域 bbox の構築
# ==========================
def build_sub_bbox_from_subregion(
    region: List[int],  # [x, y, w, h]（screen.region）
    sub_region: List[int]  # [sx, sy, sw, sh]
) -> Tuple[int, int, int, int]:
    """
    screen.region の左上を原点として sub_region を加算した bbox を返す。
    MSS の grab 用に (left, top, right, bottom) で返す。
    """
    x, y, w, h = region
    sx, sy, sw, sh = sub_region
    left = x + sx
    top = y + sy
    right = left + sw
    bottom = top + sh
    return (left, top, right, bottom)


# ==========================
# テンプレ照合（矩形）
# ==========================
def match_best_scale(
    gray_roi: np.ndarray,
    tmpl_gray: np.ndarray,
    scales,
    method=cv2.TM_CCOEFF_NORMED
) -> Dict[str, Any]:
    """
    複数スケールでのテンプレ照合。最良スコア等を返す。
    戻り値: dict(score, loc, tw, th, scale)
    """
    best = {"score": -1.0, "loc": (0, 0), "tw": 0, "th": 0, "scale": 1.0}
    for s in scales:
        tw = int(round(tmpl_gray.shape[1] * s))
        th = int(round(tmpl_gray.shape[0] * s))
        if tw <= 2 or th <= 2:
            continue
        resized = cv2.resize(tmpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(gray_roi, resized, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        score = max_val if method in (cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED) else 1 - min_val
        if score > best["score"]:
            best.update({"score": score, "loc": max_loc, "tw": tw, "th": th, "scale": s})
    return best


# ==========================
# テンプレ照合（マスク付き：円テンプレ用）
# ==========================
def match_best_scale_with_mask(
    gray_roi: np.ndarray,
    tmpl_gray: np.ndarray,
    mask_gray: np.ndarray,
    scales,
) -> Dict[str, Any]:
    """
    マスク対応のテンプレ照合で最良スコアを返す。
    method は TM_CCORR_NORMED を使用（OpenCV 4.2+ でマスク対応）。
    戻り値: dict(score, loc, tw, th, scale)
    """
    best = {"score": -1.0, "loc": (0, 0), "tw": 0, "th": 0, "scale": 1.0}
    for s in scales:
        tw = int(round(tmpl_gray.shape[1] * s))
        th = int(round(tmpl_gray.shape[0] * s))
        if tw <= 2 or th <= 2:
            continue
        resized_t = cv2.resize(tmpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
        resized_m = cv2.resize(mask_gray, (tw, th), interpolation=cv2.INTER_NEAREST)
        res = cv2.matchTemplate(gray_roi, resized_t, cv2.TM_CCORR_NORMED, mask=resized_m)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best["score"]:
            best.update({"score": max_val, "loc": max_loc, "tw": tw, "th": th, "scale": s})
    return best


# ==========================
# デバッグ描画/保存
# ==========================
def save_debug_rect(img_bgr: np.ndarray, left: int, top: int, w: int, h: int, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.rectangle(img_bgr, (left, top), (left + w, top + h), (0, 255, 0), 2)
    cv2.imwrite(str(out_path), img_bgr)


def save_debug_circle(img_bgr: np.ndarray, center: Tuple[int, int], radius: int, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.circle(img_bgr, center, radius, (0, 255, 0), 2)
    cv2.imwrite(str(out_path), img_bgr)


# ==========================
# 円検出（Hough）
# ==========================
def detect_circle_hough(
    gray_roi: np.ndarray,
    dp: float,
    min_dist: int,
    param1: float,
    param2: float,
    min_radius: int,
    max_radius: int,
) -> Optional[Tuple[int, int, int]]:
    """
    HoughCircles で1個だけ返す（最良とみなす）。
    戻り: (x, y, r) or None
    """
    circles = cv2.HoughCircles(
        gray_roi, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
        param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius
    )
    if circles is None:
        return None
    c = np.uint16(np.around(circles))[0][0]  # 最初の1個
    return int(c[0]), int(c[1]), int(c[2])


# ==========================
# ワーカースレッド
# ==========================
class CaptureWorker(threading.Thread):
    def __init__(self, cfg: Dict[str, Any], on_log: Optional[Callable[[str], None]] = None):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.on_log = on_log or (lambda s: None)
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def _tick_sleep(self):
        # ループ間隔（ms）
        iv_ms = int(self.cfg.get("timing", {}).get("interval_ms", 500))
        time.sleep(max(iv_ms, 10) / 1000.0)

    def run(self):
        count = 0
        self.on_log("稼働中…")
        while not self._stop_evt.is_set():
            count += 1
            # ここで必要ならキャプチャ等の実処理を行う
            # 今は軽量スモーク（ログ更新のみ）
            if count % 10 == 0:
                self.on_log(f"稼働中…（{count} サイクル）")
            self._tick_sleep()
        self.on_log("停止処理中…")


# ==========================
# GUI本体（レイアウトは現状維持）
# ==========================
class ControlWindow(tk.Tk):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.title("Game Bot Control")
        self.geometry("420x260")
        self.resizable(False, False)
        self.attributes("-topmost", True)

        self._worker: Optional[CaptureWorker] = None
        self.status_var = tk.StringVar(value="停止中")

        tk.Label(self, text="ゲームボット操作", font=("Segoe UI", 14, "bold")).pack(pady=(12, 6))
        tk.Label(self, textvariable=self.status_var).pack(pady=(0, 8))

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=6, fill="x", padx=12)
        self.start_btn = tk.Button(btn_frame, text="開始", width=12, command=self.on_start)
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.stop_btn = tk.Button(btn_frame, text="停止", width=12, state="disabled", command=self.on_stop)
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=(6, 0))

        bottom = tk.Frame(self)
        bottom.pack(pady=6, fill="x", padx=12)
        tk.Button(bottom, text="設定フォルダを開く", command=self.open_config_folder).pack(side="left")
        tk.Button(bottom, text="終了", command=self.on_exit).pack(side="right")

        tk.Label(self, text="※ このウィンドウは常に最前面に固定されています（Escで終了）。", fg="#666").pack(pady=(4, 8))
        self.bind("<Escape>", lambda e: self.on_exit())

        # ×ボタンで閉じたときも終了ログを出す
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

        # 定期的に topmost を維持
        self.after(1000, self.keep_topmost)

    def on_start(self):
        if self._worker and self._worker.is_alive():
            self.status_var.set("既に稼働中です")
            return

        self.status_var.set("稼働中…")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        def log_cb(msg: str):
            self.after(0, self.status_var.set, msg)

        self._worker = CaptureWorker(self.cfg, on_log=log_cb)
        self._worker.start()
        log_info("ワーカー開始")

    def on_stop(self):
        if self._worker:
            self._worker.stop()
            self._worker.join(timeout=2.0)
            self._worker = None

#       # 停止時に最終キャプチャを保存
#       try:
#           with mss.mss() as sct:
#               left, top, right, bottom = build_sub_bbox_from_subregion(
#                   self.cfg["screen"]["region"], self.cfg["screen"]["sub_region"]
#               )
#               w = right - left
#               h = bottom - top
#               shot = np.array(sct.grab({"left": left, "top": top, "width": w, "height": h}))[:, :, :3]
#           out = Path(self.cfg["app"]["debug_dir"]) / "_stopped_last.png"
#           out.parent.mkdir(parents=True, exist_ok=True)
#           cv2.imwrite(str(out), shot)
#           log_info(f"停止時キャプチャ保存: {out}")
#       except Exception as e:
#           log_error(f"停止時キャプチャ失敗: {e}")

        self.status_var.set("停止済み")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def open_config_folder(self):
        cfg_dir = self.cfg.get("_meta", {}).get("config_dir")
        try:
            if cfg_dir and Path(cfg_dir).exists():
                os.startfile(cfg_dir)
            else:
                os.startfile(os.path.dirname(os.path.abspath(__file__)))
            log_info(f"設定フォルダを開く: {cfg_dir or ROOT}")
        except Exception as e:
            log_error(f"設定フォルダを開けませんでした: {e}")
            messagebox.showerror("エラー", f"設定フォルダを開けませんでした:\n{e}")

    def on_exit(self):
        try:
            # 終了前にワーカーを止める
            if self._worker:
                self._worker.stop()
                self._worker.join(timeout=2.0)
        finally:
            log_app_exit()
            self.destroy()

    def keep_topmost(self):
        self.attributes("-topmost", True)
        self.after(1000, self.keep_topmost)


# ==========================
# メイン
# ==========================
def main():
    if sys.platform.startswith("win"):
        set_dpi_awareness_windows()

    cfg = load_config()

    # 起動ログ（タイトルは固定表示のまま、ログ用に使うだけ）
    log_app_start("Game Bot Control")

    # GUI 起動（閉じるまでブロック）
    app = ControlWindow(cfg)
    app.mainloop()


if __name__ == "__main__":
    main()
