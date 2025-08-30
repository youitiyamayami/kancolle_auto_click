# modules/app_logger.py
"""
アプリ全体で共通利用するロガーの初期化モジュール。

■ 目的
- どこでも `from modules.app_logger import get_logger` で同一設定のロガーを取得
- ログは「log\\YYYYMMDD_app.log」に日付ごとに追記、コンソールにも出力可能
- フォーマット: [YYYY:MM:DD hh:mm:ss][LEVEL] メッセージ
- タイムゾーンは Asia/Tokyo（ローカル時刻）
- Shift_JIS(CP932) でファイル出力（ユーザー環境要件）
- ハンドラ重複防止、未捕捉例外もERRORで記録可能
- 実行中にログレベルの動的変更も可能

■ 主なAPI
- get_logger(name="app", cfg: dict | None = None, **overrides) -> logging.Logger
- set_level(level: str | int, logger: logging.Logger | None = None) -> None
- log_app_start(title="App") / log_app_exit() / log_info(msg) / log_error(msg)
- log_job(title, level=logging.INFO) -> contextmanager（開始/終了と経過時間を自動ログ）
- install_global_excepthook(logger: logging.Logger | None = None) -> None

■ 互換API
- get_app_logger(cfg: dict | None = None) -> logging.Logger
- log_app_start_event(title: str) / log_app_exit_event()
- log_info(msg) / log_error(msg)
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterator, Optional
import time
import os

try:
    # 3.9+ 標準
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


# =================================
# デフォルト値
# =================================
@dataclass
class LoggerDefaults:
    level: int = logging.INFO       # 見かけの既定レベル（console向け）
    level_file: int = logging.DEBUG # ファイルは詳細目
    format: str = "[%(asctime)s][%(levelname)s] %(message)s"
    datefmt: str = "%Y:%m:%d %H:%M:%S"
    console: bool = True         # コンソール出力の有無
    file: bool = True            # ファイル出力の有無
    dir: str = "log"             # ログディレクトリ
    encoding: str = "cp932"      # ファイル出力エンコーディング（Shift_JIS互換）
    timezone: str = "Asia/Tokyo" # 表示タイムゾーン
    logfile_prefix: str = "app"  # ログファイル名のプレフィックス
    logfile_ext: str = ".log"    # 拡張子（.log 推奨）
    backup_count: int = 14  # 何日分保持するか（0 なら無制限）


_DEFAULTS = LoggerDefaults()


# ==========================================================
# ユーティリティ
# ==========================================================
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _today_str(tz_name: str) -> str:
    tz = ZoneInfo(tz_name) if ZoneInfo and tz_name else None
    now = datetime.now(tz=tz)
    return now.strftime("%Y%m%d")


def _build_logfile_path(root: Path, prefix: str, ext: str, tz_name: str) -> Path:
    return root / f"{_today_str(tz_name)}_{prefix}{ext}"


def _to_level(level: Optional[str | int], fallback: int = logging.INFO) -> int:
    if level is None:
        return fallback
    if isinstance(level, int):
        return level
    name = str(level).upper()
    return getattr(logging, name, fallback)


class _TzFormatter(logging.Formatter):
    """
    タイムゾーン付きの asctime を出力するためのフォーマッタ。
    """
    def __init__(self, fmt: str, datefmt: Optional[str], tz_name: str) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._tz = ZoneInfo(tz_name) if ZoneInfo and tz_name else None

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:  # noqa: N802
        dt = datetime.fromtimestamp(record.created, tz=self._tz)
        if datefmt:
            return dt.strftime(datefmt)
        # デフォルト表記
        return dt.strftime("%Y:%m:%d %H:%M:%S")


# ====
# 当モジュールが追加したハンドラかどうかを識別するタグ
# ====
_HANDLER_TAG = "_app_logger_tag"


def _is_ours(handler: logging.Handler) -> bool:
    return bool(getattr(handler, _HANDLER_TAG, False))


def _tag(handler: logging.Handler) -> None:
    setattr(handler, _HANDLER_TAG, True)


# ----------------------------------------------------------
# 日付プレフィックス付きのローテーションハンドラ（00:00で切替）
# ----------------------------------------------------------
class DatePrefixTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    ログファイル名を「YYYYMMDD_{prefix}{ext}」の形で日次ローテーションするハンドラ。
    backupCount に基づいて古いファイルを自動削除する。
    """
    def __init__(self, *, root_dir: Path, prefix: str, ext: str, tz_name: str, backupCount: int = 14, encoding: str = "cp932") -> None:
        self.root_dir = Path(root_dir)
        self.prefix = prefix
        self.ext = ext
        self.tz_name = tz_name
        # 初回は当日ファイルを baseFilename に設定
        filename = str(_build_logfile_path(self.root_dir, self.prefix, self.ext, self.tz_name))
        super().__init__(filename, when="midnight", interval=1, backupCount=backupCount, encoding=encoding, delay=True, utc=False)

    def shouldRollover(self, record):  # type: ignore[override]
        # 期待される「今日の」ファイル名と異なればロールオーバー
        expected = str(_build_logfile_path(self.root_dir, self.prefix, self.ext, self.tz_name))
        return 1 if self.baseFilename != expected else 0

    def doRollover(self):  # type: ignore[override]
        # 現在のストリームを閉じる
        if self.stream:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        # baseFilename を「今日の」ファイル名へ更新
        self.baseFilename = str(_build_logfile_path(self.root_dir, self.prefix, self.ext, self.tz_name))
        # 古いファイルを削除（backupCount を超える分）
        try:
            files = sorted(self.root_dir.glob(f"[0-9]"*8 + f"_{self.prefix}{self.ext}"))
        except Exception:
            files = []
        if self.backupCount and self.backupCount > 0 and files:
            excess = len(files) - self.backupCount
            if excess > 0:
                for f in files[:excess]:
                    try:
                        f.unlink()
                    except Exception:
                        pass
        # 次回ロールオーバー時刻を再計算
        currentTime = int(time.time())
        self.rolloverAt = self.computeRollover(currentTime)


# ==========================================================
# 主関数: get_logger
# ==========================================================
def get_logger(
    name: str = "app",
    *,
    cfg: Optional[Dict[str, Any]] = None,
    level: Optional[str | int] = None,
    console: Optional[bool] = None,
    file: Optional[bool] = None,
    logfile_root: Optional[Path | str] = None,
    encoding: Optional[str] = None,
    timezone: Optional[str] = None,
    logfile_prefix: Optional[str] = None,
    logfile_ext: Optional[str] = None,
) -> logging.Logger:
    """
    共通ロガーを返す。複数回呼んでもハンドラは重複追加されない。
    """
    # 1) cfg と引数から最終値を決定
    d = (cfg or {}).get("logging", {}) if (cfg and "logging" in cfg) else (cfg or {})
    level_val = _to_level(level if level is not None else d.get("level", _DEFAULTS.level))
    use_console = bool(console if console is not None else d.get("console", _DEFAULTS.console))
    use_file = bool(file if file is not None else d.get("file", _DEFAULTS.file))
    backup_count = int(d.get("backup_count", _DEFAULTS.backup_count))

    root_dir = logfile_root or d.get("dir", _DEFAULTS.dir)
    enc = encoding or d.get("encoding", _DEFAULTS.encoding)
    tz_name = timezone or d.get("timezone", _DEFAULTS.timezone)
    prefix = logfile_prefix or d.get("logfile_prefix", _DEFAULTS.logfile_prefix)
    ext = logfile_ext or d.get("logfile_ext", _DEFAULTS.logfile_ext)

    logger = logging.getLogger(name)

    # 2) フォーマッタ
    formatter = _TzFormatter(fmt=d.get("format", _DEFAULTS.format), datefmt=d.get("datefmt", _DEFAULTS.datefmt), tz_name=tz_name)

    # 3) ロガー本体のレベルは DEBUG にして、ハンドラ側で絞るのが柔軟
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 祖先に伝播させない（重複防止）

    # 4) 既存ハンドラの再利用判定
    has_console = False
    has_file = False
    for h in logger.handlers:
        if _is_ours(h):
            if isinstance(h, logging.StreamHandler):
                has_console = True
            if isinstance(h, logging.FileHandler):
                has_file = True

    # 5) コンソールハンドラ
    if use_console and not has_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level_val)
        ch.setFormatter(formatter)
        _tag(ch)
        logger.addHandler(ch)

    # 6) ファイルハンドラ（TimedRotating: 日付切替＋保持数）
    if use_file and not has_file:
        logfile_dir = _ensure_dir(Path(root_dir))
        fh = DatePrefixTimedRotatingFileHandler(
            root_dir=logfile_dir,
            prefix=prefix,
            ext=ext,
            tz_name=tz_name,
            backupCount=backup_count,
            encoding=enc,
        )
        fh.setLevel(_to_level(d.get("level_file"), logging.DEBUG))
        fh.setFormatter(formatter)
        _tag(fh)
        logger.addHandler(fh)

    # 7) 明示レベル→主にコンソール閾値に反映
    set_level(level_val, logger=logger)

    return logger


# ========================================
# レベル操作・スコープログ
# ========================================
def set_level(level: str | int, logger: Optional[logging.Logger] = None) -> None:
    """
    実行中にコンソール（必要ならファイル）を含むハンドラの閾値を変更する。
    """
    lg = logger or logging.getLogger("app")
    val = _to_level(level)
    for h in lg.handlers:
        if _is_ours(h):
            h.setLevel(val)


@contextmanager
def log_job(title: str, level: int = logging.INFO) -> Iterator[None]:
    """
    with log_job("〜〜"): の形でスコープ計測ログを出す。
    """
    lg = logging.getLogger("app")
    start = perf_counter()
    lg.log(level, f"{title} - 開始")
    try:
        yield
        elapsed = perf_counter() - start
        lg.log(level, f"{title} - 終了 (elapsed={elapsed:.3f}s)")
    except Exception:
        elapsed = perf_counter() - start
        lg.error(f"{title} - 失敗 (elapsed={elapsed:.3f}s)", exc_info=True)
        raise


def install_global_excepthook(logger: Optional[logging.Logger] = None) -> None:
    """未捕捉例外を ERROR でロギングする excepthook をインストールする（メイン/スレッド）。"""
    lg = logger or logging.getLogger("app")

    def _hook(exc_type, exc, tb):
        try:
            lg.exception("Uncaught exception", exc_info=(exc_type, exc, tb))
        finally:
            sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook

    # スレッド例外（Python 3.8+）
    def _thread_hook(args):
        try:
            lg.exception("Uncaught exception in thread %s", args.thread.name,
                         exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        finally:
            if hasattr(threading, "__excepthook__"):
                threading.__excepthook__(args)
    try:
        threading.excepthook = _thread_hook  # type: ignore[attr-defined]
    except Exception:
        pass


# ==========================================================
# 互換API（main.py が使っている関数名を提供）
# ==========================================================
_APP_LOGGER: Optional[logging.Logger] = None


def get_app_logger(cfg: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """アプリ共通ロガー（シングルトン）を取得。"""
    global _APP_LOGGER
    if _APP_LOGGER is None:
        _APP_LOGGER = get_logger("app", cfg=cfg)
        # 未捕捉例外も拾う
        install_global_excepthook(_APP_LOGGER)
    return _APP_LOGGER


def log_app_start(title: str = "App") -> None:
    get_app_logger().info(f"アプリ起動: {title}")


def log_app_exit() -> None:
    get_app_logger().info("アプリ終了")


def log_info(msg: str) -> None:
    get_app_logger().info(msg)


def log_error(msg: str) -> None:
    get_app_logger().error(msg)


# ==========================================================
# サンプル（手動テスト用）
# ==========================================================
if __name__ == "__main__":
    log_app_start("Demo")
    log_info("INFO ログのテスト")
    try:
        with log_job("デモ処理"):
            raise ValueError("テスト例外")
    except Exception:
        log_error("例外を捕捉しました")
    log_app_exit()
