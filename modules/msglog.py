# modules/msglog.py
from __future__ import annotations
import json
import sys
import platform
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from modules.app_logger import get_app_logger

# 内部保持
_MESSAGES: Dict[str, Dict[str, Any]] = {}  # code -> {"template": str, "level": "INFO", "tags": [...]}


# 例外
class MessagesLoadError(RuntimeError):
    pass


# ---------- ロード & 正規化 ----------

def _normalize_entry(code: str, value: Any) -> Dict[str, Any]:
    """
    単純文字列 or 構造化(dict) の両方を統一フォーマットに正規化する。
    戻り値は {"template": str, "level": str, "tags": list}
    """
    if isinstance(value, str):
        return {"template": value, "level": "INFO", "tags": []}

    if isinstance(value, dict):
        if "template" not in value or not isinstance(value["template"], str):
            raise MessagesLoadError(f'messages[{code}] は "template"（文字列）が必須です')
        tmpl = value["template"]
        level = str(value.get("level", "INFO")).upper()
        tags = value.get("tags", [])
        if not isinstance(tags, list):
            raise MessagesLoadError(f'messages[{code}].tags は配列である必要があります')
        return {"template": tmpl, "level": level, "tags": list(tags)}

    raise MessagesLoadError(f"messages[{code}] は文字列またはオブジェクトである必要があります")


def _load_and_normalize(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise MessagesLoadError("messages ファイルは JSON オブジェクトである必要があります")

    normalized: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        code = str(k)
        normalized[code] = _normalize_entry(code, v)
    return normalized


def configure_messages(msg_path_str: Optional[str], default_search_root: Optional[Path] = None) -> None:
    """
    メッセージ定義ファイル（JSON）をロードし、グローバルへ設定。
    仕様:
      - 明示パスがあればそれを必須とする（無ければ FileNotFoundError）
      - 無ければ <root>/config/messages.ja.json を必須とする
      - フォールバックは行わない
    """
    global _MESSAGES
    log = get_app_logger()

    # 1) 明示パスが指定されていればそれを必須化
    if msg_path_str:
        p = Path(msg_path_str)
        if not p.exists():
            raise FileNotFoundError(
                f"Messages file not found: {p}\n"
                "Please create it or fix [logging.messages].path in your config."
            )
        _MESSAGES = _load_and_normalize(p)
        log.info(f"[MSG] messages loaded from: {p}")
        return

    # 2) <root>/config/messages.ja.json を必須化
    if default_search_root is None:
        raise FileNotFoundError(
            "default_search_root is not provided and no messages path is specified.\n"
            "Cannot locate required config/messages.ja.json."
        )
    ja = default_search_root / "config" / "messages.ja.json"
    if not ja.exists():
        raise FileNotFoundError(
            f"Required messages file not found: {ja}\n"
            "Create this file or set [logging.messages].path to a valid JSON."
        )
    _MESSAGES = _load_and_normalize(ja)
    log.info(f"[MSG] messages loaded from: {ja}")


# ---------- ログ出力 ----------

def _format_message(tmpl: str, kwargs: Dict[str, Any]) -> str:
    try:
        return tmpl.format(**kwargs)
    except Exception:
        # プレースホルダ不足のときもそのまま出す（安全側）
        return tmpl


def _route_level(level: str):
    level = (level or "INFO").upper()
    log = get_app_logger()
    return {
        "DEBUG": log.debug,
        "INFO": log.info,
        "WARNING": log.warning,
        "WARN": log.warning,
        "ERROR": log.error,
        "CRITICAL": log.critical,
    }.get(level, log.info)


def log_event(code: str, **kwargs: Any) -> None:
    """
    メッセージカタログのコードでログ出力。
    - 既定: INFO
    - 未定義コードは code 自体をメッセージとして INFO 出力（安全側）
    """
    entry = _MESSAGES.get(code)
    if entry is None:
        _route_level("INFO")(code)
        return

    msg = _format_message(entry["template"], kwargs)
    _route_level(entry.get("level", "INFO"))(msg)


# ---------- 便利関数（サンプルコード用） ----------

def log_app_start_event(title: str) -> None:
    """
    APP_START: 構造化でも文字列でも OK
    - template 側が {title} / {python} / {platform} のどれを使っても埋まるように全て渡す
    - 後方互換で {app} も渡す
    """
    py = platform.python_version()
    plat = platform.platform()
    log_event("APP_START", title=title, app=title, python=py, platform=plat)


def log_app_exit_event() -> None:
    log_event("APP_EXIT")
