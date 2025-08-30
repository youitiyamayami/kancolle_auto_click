# modules/msglog.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional
from modules.app_logger import get_app_logger

# ロード済みメッセージの保持
_MESSAGES: Dict[str, str] = {}


class MessagesLoadError(RuntimeError):
    """メッセージ定義ファイルの形式/内容が不正な場合に投げる例外。"""
    pass


def _load_messages_json(path: Path) -> Dict[str, str]:
    """JSONファイルを読み込み、{code: message} 形式の dict を返す。"""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise MessagesLoadError(f"Invalid messages file format (not a JSON object): {path}")

    # すべて文字列化しておく
    return {str(k): str(v) for k, v in data.items()}


def configure_messages(msg_path_str: Optional[str], default_search_root: Optional[Path] = None) -> None:
    """
    メッセージ定義ファイル（JSON）をロードしてグローバルに設定する。
    要件:
      - 明示パスが指定されていればそれを必須とする（存在しない/壊れていればエラー）
      - 明示パスがなければ <root>/config/messages.ja.json の存在を必須とする
      - いずれも満たせなければ FileNotFoundError を送出（フォールバック無し）
    """
    global _MESSAGES
    log = get_app_logger()

    # 1) 設定で明示されたパスを優先
    if msg_path_str:
        path = Path(msg_path_str)
        if not path.exists():
            raise FileNotFoundError(
                f"Messages file not found: {path}\n"
                "Please create it or fix [logging.messages].path in your config."
            )
        _MESSAGES = _load_messages_json(path)
        log.info(f"[MSG] messages loaded from: {path}")
        return

    # 2) 既定探索: <root>/config/messages.ja.json を必須化
    if default_search_root is None:
        raise FileNotFoundError(
            "default_search_root is not provided and no messages path is specified.\n"
            "Cannot locate required config/messages.ja.json."
        )

    ja_candidate = default_search_root / "config" / "messages.ja.json"
    if not ja_candidate.exists():
        raise FileNotFoundError(
            f"Required messages file not found: {ja_candidate}\n"
            "Create this file or set [logging.messages].path to a valid JSON."
        )

    _MESSAGES = _load_messages_json(ja_candidate)
    log.info(f"[MSG] messages loaded from: {ja_candidate}")


def log_event(code: str, **kwargs: Any) -> None:
    """
    イベントをメッセージカタログのコードで記録する。
    configure_messages() 実行前に呼ばれた場合は KeyError 回避のため code をそのまま出力。
    """
    log = get_app_logger()
    tmpl = _MESSAGES.get(code, code)
    try:
        msg = tmpl.format(**kwargs)
    except Exception:
        msg = tmpl
    log.info(msg)


def log_app_start_event(app_name: str) -> None:
    log_event("APP_START", app=app_name)


def log_app_exit_event() -> None:
    log_event("APP_EXIT")
