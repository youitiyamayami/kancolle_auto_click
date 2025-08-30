# main.py
# -----------------------------------------------------------------------------
# レイアウト変更なし。イベント処理は modules/gui_actions へ委譲。
# -----------------------------------------------------------------------------

import sys
import os
import time
import threading
from typing import Dict, Any, Optional, Tuple, List, Callable
from pathlib import Path

try:
    import tomllib  # Python 3.11+
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

# 既存ロガー（ハンドラやフォーマットはこれが持つ）
from modules.app_logger import get_app_logger  # 既存の設定を流用

# 設定ローダ
from modules.config_loader import load_config

# メッセージカタログ式ロガー（外部化）
from modules.msglog import (
    configure_messages,
    log_app_start_event,
    log_app_exit_event,
)

# GUIアクションとアプリケーションコンテキスト
from modules.app_context import AppContext
import modules.gui_actions as gui_actions

# ルート
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

        # ---- AppContext を準備（UIはそのまま・処理だけ外出し） ----
        logger = get_app_logger()
        self.ctx = AppContext(
            logger=logger,
            config=self.cfg,
            project_root=ROOT,
            set_status=self.status_var.set,
            start_worker=self._start_worker,
            stop_worker=self._stop_worker,
            worker_is_alive=self._worker_is_alive,
            open_path=lambda p: os.startfile(p),
            get_config_dir=lambda: (self.cfg.get("_meta", {}) or {}).get("config_dir"),
        )

    # --- worker 管理（コンテキストへ提供するコールバック） ---
    def _start_worker(self, on_log: Callable[[str], None]) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = CaptureWorker(self.cfg, on_log=on_log)
        self._worker.start()

    def _stop_worker(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker.join(timeout=2.0)
            self._worker = None

    def _worker_is_alive(self) -> bool:
        return bool(self._worker and self._worker.is_alive())

    # --- GUIイベント（処理は gui_actions へ委譲） ---
    def on_start(self):
        started, msg = gui_actions.start(self.ctx)
        self.status_var.set(msg)
        if started:
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")

    def on_stop(self):
        stopped, msg = gui_actions.stop(self.ctx)
        self.status_var.set(msg)
        if stopped:
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")

    def open_config_folder(self):
        ok, msg = gui_actions.open_config_folder(self.ctx)
        self.status_var.set(msg)
        if not ok:
            messagebox.showwarning("Open Config", msg)

    def on_exit(self):
        try:
            # 稼働中であれば停止してから終了
            if self.ctx.worker_is_alive():
                gui_actions.stop(self.ctx)
        finally:
            log_app_exit_event()  # APP_EXIT
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

    # メッセージカタログの場所を設定（無ければデフォルト探索にフォールバック）
    msg_path = (cfg.get("logging", {}) or {}).get("messages", {}).get("path")
    configure_messages(msg_path, default_search_root=ROOT)

    # 起動ログ（APP_START）
    log_app_start_event("Game Bot Control")

    # GUI 起動（閉じるまでブロック）
    app = ControlWindow(cfg)
    app.mainloop()


if __name__ == "__main__":
    # 既存ロガーの初期化を強制したい場合はここで get_app_logger() を呼ぶ
    _ = get_app_logger()
    main()
