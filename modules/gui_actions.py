# modules/gui_actions.py
from __future__ import annotations
from typing import Tuple
from pathlib import Path
from modules.app_context import AppContext
from modules.msglog import log_event


def is_running(ctx: AppContext) -> bool:
    return ctx.worker_is_alive()


def start(ctx: AppContext) -> Tuple[bool, str]:
    """
    開始ボタン：既に稼働中なら何もしない。
    戻り値: (開始できたか, ステータスメッセージ)
    """
    if ctx.worker_is_alive():
        return (False, "既に稼働中です")

    # ワーカー開始（ログ更新コールバックはUI越しに注入）
    def _on_log(msg: str) -> None:
        ctx.set_status(msg)

    ctx.start_worker(_on_log)
    log_event("WORKER_START")
    return (True, "稼働中…")


def stop(ctx: AppContext) -> Tuple[bool, str]:
    """
    停止ボタン：稼働中なら停止。
    戻り値: (停止したか, ステータスメッセージ)
    """
    if not ctx.worker_is_alive():
        return (False, "すでに停止しています")

    ctx.stop_worker()
    log_event("WORKER_STOP")
    return (True, "停止済み")


def open_config_folder(ctx: AppContext) -> Tuple[bool, str]:
    """
    設定フォルダを開く。config側に指定がなければプロジェクトルートを開く。
    戻り値: (成功, メッセージ)
    """
    cfg_dir = ctx.get_config_dir()
    try:
        if cfg_dir and Path(cfg_dir).exists():
            ctx.open_path(cfg_dir)
            log_event("OPEN_CONFIG_OK", path=cfg_dir)
            return (True, f"設定フォルダを開きました：{cfg_dir}")
        else:
            fallback = str(ctx.project_root)
            ctx.open_path(fallback)
            log_event("OPEN_CONFIG_OK", path=fallback)
            return (True, f"プロジェクトフォルダを開きました：{fallback}")
    except Exception as e:
        log_event("OPEN_CONFIG_ERROR", error=str(e))
        return (False, f"設定フォルダを開けませんでした：{e}")
