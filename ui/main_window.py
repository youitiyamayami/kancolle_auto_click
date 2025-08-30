# ./ui/main_window.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Tkinter ベースのメインウィンドウ実装（ログ非表示・約400x400・常に最前面）。

役割:
- GUI（ウィンドウ生成・レイアウト・イベントループ）を担う最上位レイヤ。
- ボタン操作から modules/gui_actions.py の start/stop/open_config_folder を呼び出す。
- AppContext の生成は可能な限り既存実装を使用し、不可なら安全なダミー実装へフォールバック。

本版の追加要件:
- GUIにはログを出力しない（ログビュー削除、メッセージはステータス表示のみ）。
- GUIサイズは約 400x400。
- GUIは常に最前面（-topmost=True）。
"""

import os
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# --- Tkinter 標準UI ---
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# --- プロジェクト依存: ボタン背後の処理（開始・停止・設定フォルダ） ---
from modules import gui_actions  # start/stop/open_config_folder を利用

# ------------------------------------------------------------
# ユーティリティ: 設定辞書から安全に値を取得する
# ------------------------------------------------------------
def _cfg_get(cfg: dict, dotted_key: str, default: Any) -> Any:
    """
    設定 dict から 'section.key' のようなドット表記で値を安全取得する。
    :param cfg: 設定辞書
    :param dotted_key: 'app.window_title' のようなドット区切りキー
    :param default: キーが無い/不正時の既定値
    :return: 取得値 or 既定値
    """
    cur: Any = cfg
    for part in dotted_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


# ------------------------------------------------------------
# AppContext を用意するファクトリ
# ------------------------------------------------------------
def _create_app_context(config: dict, log) -> Any:
    """
    既存の modules/app_context.py があればそれを利用し、なければダミーへフォールバック。
    - 戻り値: ctx オブジェクト（duck typing: gui_actions が要求するメソッドを備える）
    """
    root = Path(__file__).resolve().parents[1]  # プロジェクトルート
    try:
        # 既存の AppContext を最優先で利用する（存在すれば）
        from modules.app_context import AppContext  # type: ignore
        try:
            ctx = AppContext()  # シグネチャ不明のため最小呼び出し
        except TypeError:
            ctx = AppContext()
        _patch_context_if_needed(ctx, config, log, root)
        return ctx
    except Exception:
        # modules/app_context が使えない場合はダミーで代替
        return _DummyAppContext(config=config, log=log, project_root=root)


def _patch_context_if_needed(ctx: Any, config: dict, log, root: Path) -> None:
    """
    既存 AppContext が gui_actions の要求（worker_is_alive/start_worker/set_status/stop_worker/
    get_config_dir/open_path/project_root）を満たしていない場合に、最低限の実装を後付けする。
    既に備わっている場合は上書きしない。
    """
    # プロジェクトルートを持たせる
    if not hasattr(ctx, "project_root"):
        setattr(ctx, "project_root", root)

    # ステータス保持
    if not hasattr(ctx, "_status"):
        setattr(ctx, "_status", "待機中")

    # ステータス更新 API
    if not hasattr(ctx, "set_status"):
        def _set_status(msg: str) -> None:
            setattr(ctx, "_status", str(msg))
        setattr(ctx, "set_status", _set_status)

    # 設定フォルダパス取得
    if not hasattr(ctx, "get_config_dir"):
        def _get_config_dir() -> str:
            return str((root / "config").resolve())
        setattr(ctx, "get_config_dir", _get_config_dir)

    # OS既定アプリでパスを開く
    if not hasattr(ctx, "open_path"):
        def _open_path(p: str) -> None:
            if sys.platform.startswith("win"):
                os.startfile(p)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{p}"')
            else:
                os.system(f'xdg-open "{p}"')
        setattr(ctx, "open_path", _open_path)

    # ワーカー関連（最低限）
    if not hasattr(ctx, "_worker"):
        setattr(ctx, "_worker", None)

    if not hasattr(ctx, "worker_is_alive"):
        def _worker_is_alive() -> bool:
            w = getattr(ctx, "_worker", None)
            return (w is not None) and w.is_alive()
        setattr(ctx, "worker_is_alive", _worker_is_alive)

    if not hasattr(ctx, "start_worker"):
        def _start_worker(on_log_cb: Optional[Callable[[str], None]] = None) -> None:
            if ctx.worker_is_alive():
                return
            stop_flag = {"stop": False}
            def _run():
                # ダミーの作業ループ（既存実装がある場合はそちらが呼ばれる想定）
                while not stop_flag["stop"]:
                    # GUIにはログを出さない → on_log_cb は呼ばないか、必要最小限に抑える
                    ctx.set_status("稼働中…")
                    time.sleep(0.5)
                ctx.set_status("停止済み")
            t = threading.Thread(target=_run, name="WORKER", daemon=True)
            setattr(ctx, "_worker", t)
            setattr(ctx, "_stop_flag", stop_flag)
            t.start()
        setattr(ctx, "start_worker", _start_worker)

    if not hasattr(ctx, "stop_worker"):
        def _stop_worker() -> None:
            flag = getattr(ctx, "_stop_flag", None)
            if isinstance(flag, dict):
                flag["stop"] = True
            w = getattr(ctx, "_worker", None)
            if w is not None:
                w.join(timeout=1.0)
            setattr(ctx, "_worker", None)
        setattr(ctx, "stop_worker", _stop_worker)


class _DummyAppContext:
    """
    既存 AppContext が利用できない場合の安全なダミー実装。
    gui_actions が要求する最低限のAPIだけを提供する。
    """
    def __init__(self, config: dict, log, project_root: Path) -> None:
        self._config = config            # 設定辞書（UIタイトル取得などに使用）
        self._log = log                  # ロガーへの参照
        self.project_root = project_root # プロジェクトルート
        self._status = "待機中"           # ステータス表示用
        self._worker: Optional[threading.Thread] = None
        self._stop_flag = {"stop": False}

    def worker_is_alive(self) -> bool:
        """ワーカー（バックグラウンド処理）スレッドが動作中かを返す。"""
        return (self._worker is not None) and self._worker.is_alive()

    def set_status(self, msg: str) -> None:
        """UI表示用のステータス文字列を更新する。"""
        self._status = str(msg)

    def start_worker(self, on_log_cb: Optional[Callable[[str], None]] = None) -> None:
        """
        ワーカーを起動する（ダミー：0.5秒ごとにステータス更新）。
        GUIにはログを出さないため on_log_cb は使わない。
        """
        if self.worker_is_alive():
            return
        self._stop_flag = {"stop": False}
        def _run():
            while not self._stop_flag["stop"]:
                self.set_status("稼働中…(dummy)")
                time.sleep(0.5)
            self.set_status("停止済み")
        t = threading.Thread(target=_run, name="DUMMY-WORKER", daemon=True)
        self._worker = t
        t.start()

    def stop_worker(self) -> None:
        """ワーカーを停止する。"""
        self._stop_flag["stop"] = True
        if self._worker:
            self._worker.join(timeout=1.0)
        self._worker = None

    def get_config_dir(self) -> str:
        """設定フォルダのパスを返す。"""
        return str((self.project_root / "config").resolve())

    def open_path(self, p: str) -> None:
        """OS既定アプリでパスを開く。"""
        if sys.platform.startswith("win"):
            os.startfile(p)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{p}"')
        else:
            os.system(f'xdg-open "{p}"')


# ------------------------------------------------------------
# メインウィンドウ（ログ非表示・約400x400・常に最前面）
# ------------------------------------------------------------
class MainWindow(tk.Tk):
    """
    Tkinter のメインウィンドウ。
    - ボタン群（開始／停止／設定フォルダ／終了）
    - ステータスラベルのみ（GUIにはログを出力しない）
    - 500msごとのポーリングでワーカー状態を表示
    - 常に最前面（-topmost=True）
    """
    def __init__(self, ctx: Any, config: dict, log) -> None:
        super().__init__()
        self.ctx = ctx              # AppContext（既存 or ダミー）
        self.config_dict = config   # 設定辞書（タイトル等に使用）
        self.log = log              # ロガー（GUIには出さず、内部/コンソール用）

        # ---- ウィンドウ初期設定 ----
        self.title(_cfg_get(config, "app.window_title", "試製七四式電子観測儀"))  # タイトル
        self.geometry("400x400+100+100")   # 約400x400
        self.minsize(360, 320)             # 読める最小サイズ
        self.attributes("-topmost", True)  # 常に最前面
        self.lift()                        # 起動直後に前面化

        # ---- ウィジェット生成 ----
        self._create_widgets()

        # ---- 状態更新ポーリング開始 ----
        self._update_status_loop()

        # ---- 終了イベント ----
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_widgets(self) -> None:
        """ウィジェット（ボタン群／ステータス）を作成し、レイアウトする。"""
        # 操作ボタンフレーム
        btn_frame = ttk.Frame(self, padding=(10, 10))
        btn_frame.pack(side=tk.TOP, fill=tk.X)

        # 開始ボタン
        self.btn_start = ttk.Button(btn_frame, text="開始", command=self._on_start_click)
        self.btn_start.pack(side=tk.LEFT, padx=6, pady=4)

        # 停止ボタン
        self.btn_stop = ttk.Button(btn_frame, text="停止", command=self._on_stop_click)
        self.btn_stop.pack(side=tk.LEFT, padx=6, pady=4)

        # 設定フォルダボタン
        self.btn_open_cfg = ttk.Button(btn_frame, text="設定フォルダを開く", command=self._on_open_config_click)
        self.btn_open_cfg.pack(side=tk.LEFT, padx=6, pady=4)

        # 余白伸縮
        ttk.Label(btn_frame, text="").pack(side=tk.LEFT, expand=True, fill=tk.X)

        # 終了ボタン
        self.btn_exit = ttk.Button(btn_frame, text="終了", command=self._on_close)
        self.btn_exit.pack(side=tk.RIGHT, padx=6, pady=4)

        # ステータス表示（ログは出さないため、状態のみ表示）
        status_frame = ttk.LabelFrame(self, text="状態", padding=(10, 6))
        status_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(6, 10))
        self.var_status = tk.StringVar(value="待機中")  # ステータステキスト保持変数
        self.lbl_status = ttk.Label(status_frame, textvariable=self.var_status)
        self.lbl_status.pack(side=tk.LEFT)

    # ---------------- ボタンハンドラ ----------------
    def _on_start_click(self) -> None:
        """
        開始ボタン押下時の処理。
        - gui_actions.start(ctx) を呼び、戻りメッセージをステータスへ反映。
        - GUIへのログ出力は行わない。
        """
        ok, msg = gui_actions.start(self.ctx)
        # ステータスのみ更新（ログビューはなし）
        self.var_status.set(msg)
        self._refresh_buttons()

    def _on_stop_click(self) -> None:
        """
        停止ボタン押下時の処理。
        - gui_actions.stop(ctx) を呼び、戻りメッセージをステータスへ反映。
        """
        ok, msg = gui_actions.stop(self.ctx)
        self.var_status.set(msg)
        self._refresh_buttons()

    def _on_open_config_click(self) -> None:
        """
        設定フォルダを OS 既定アプリで開く。
        - エラー時のみメッセージボックスで通知（GUIログは出さない）。
        """
        ok, msg = gui_actions.open_config_folder(self.ctx)
        if not ok:
            messagebox.showerror("エラー", msg)
        else:
            # 成功時はステータスのみ更新
            self.var_status.set("設定フォルダを開きました")

    # ---------------- 内部ユーティリティ ----------------
    def _refresh_buttons(self) -> None:
        """ワーカー状態に合わせてボタン活性／非活性を切り替える。"""
        alive = False
        try:
            alive = bool(self.ctx.worker_is_alive())
        except Exception:
            pass
        self.btn_start.configure(state=("disabled" if alive else "normal"))
        self.btn_stop.configure(state=("normal" if alive else "disabled"))

    def _update_status_loop(self) -> None:
        """500msごとにワーカー生死とステータスを更新する。"""
        try:
            alive = bool(self.ctx.worker_is_alive())
        except Exception:
            alive = False
        current = getattr(self.ctx, "_status", "待機中" if not alive else "稼働中…")
        self.var_status.set(current)
        self._refresh_buttons()
        self.after(500, self._update_status_loop)

    def _on_close(self) -> None:
        """ウィンドウを閉じる際のクリーンアップ（稼働中なら停止）。"""
        try:
            if self.ctx and hasattr(self.ctx, "worker_is_alive") and self.ctx.worker_is_alive():
                try:
                    gui_actions.stop(self.ctx)
                except Exception:
                    pass
        finally:
            self.destroy()


# ------------------------------------------------------------
# UI 起動関数（main.py から呼ぶ）
# ------------------------------------------------------------
def launch_ui(config: dict, log) -> int:
    """
    UI を起動するメイン関数。AppContext を生成し、MainWindow を表示する。
    :param config: 設定辞書
    :param log:    ロガー（GUIには出さない。内部処理用）
    :return: 終了コード（正常終了=0）
    """
    ctx = _create_app_context(config, log)
    win = MainWindow(ctx=ctx, config=config, log=log)
    win.mainloop()
    return 0
