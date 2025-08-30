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

import tkinter as tk
from tkinter import ttk, messagebox

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
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PER_MONITOR_AWARE
    except Exception:
        pass


# ==========================
# 画像処理ユーティリティ（例）
# ==========================
import cv2
import numpy as np


def to_abs_rect(region: Tuple[int, int, int, int], sub_region: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    REGIONとSUB_REGIONから絶対座標の矩形(left, top, right, bottom)を返す
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
        score = max_val if method in (cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED) else 1.0 - min_val
        if score > best["score"]:
            best = {"score": score, "loc": max_loc, "tw": tw, "th": th, "scale": s}
    return best


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
        logger = get_app_logger()
        try:
            count = 0
            self.on_log("稼働中…")
            while not self._stop_evt.is_set():
                count += 1
                if count % 10 == 0:
                    self.on_log(f"稼働中…（{count} サイクル）")
                self._tick_sleep()
            self.on_log("停止処理中…")
        except Exception:
            logger.exception('CaptureWorker crashed')
            raise


# ==========================
# GUI本体（レイアウトは現状維持）
# ==========================
class ControlWindow(tk.Tk):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.title("Game Bot Control")
        self.geometry("420x300")
        self.resizable(False, False)

        # ---- ステート ----
        self.status_var = tk.StringVar(value="準備完了")
        self._worker: Optional[CaptureWorker] = None

        # ---- UI ----
        self.build_ui()

        # ---- AppContext を準備（UIはそのまま・処理だけ外出し） ----
        logger = get_app_logger()
        self.ctx = AppContext(
            logger=logger,
            config=self.cfg,
            project_root=ROOT,
            set_status=self.status_var.set,
            start_worker=self._start_worker,
            stop_worker=self._stop_worker,
            is_worker_alive=self._worker_is_alive,
        )

        # 最前面固定
        self.attributes("-topmost", True)
        self.after(1000, self.keep_topmost)

    def keep_topmost(self):
        try:
            self.attributes("-topmost", True)
        finally:
            self.after(1000, self.keep_topmost)

    def build_ui(self):
        frm = tk.Frame(self, padx=12, pady=10)
        frm.pack(fill="both", expand=True)

        title = tk.Label(frm, text="Game Bot Control", font=("Meiryo UI", 14, "bold"))
        title.pack(pady=(0, 8))

        # ステータス表示
        status = tk.Label(frm, textvariable=self.status_var, anchor="w", relief="sunken")
        status.pack(fill="x", pady=(0, 8))

        # ボタン群
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

    def on_exit(self):
        if self._worker and self._worker.is_alive():
            if not messagebox.askyesno("確認", "処理中です。終了してもよろしいですか？"):
                return
            self._stop_worker()
        log_app_exit_event()
        self.destroy()

    def open_config_folder(self):
        try:
            os.startfile(str(ROOT))
        except Exception:
            messagebox.showerror("エラー", "フォルダを開けませんでした。")


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

    # ロガーを設定ファイルで初期化（ここで初期化して以後はシングルトンを利用）
    _ = get_app_logger(cfg)

    # 起動ログ（APP_START）
    log_app_start_event("Game Bot Control")

    # GUI 起動（閉じるまでブロック）
    app = ControlWindow(cfg)
    app.mainloop()


if __name__ == "__main__":
    main()
