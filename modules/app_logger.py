# modules/app_logger.py
"""
アプリ全体で共通利用する**超軽量なロガー**。

- 目的
    - 人間が読みやすい1行ログを、日付ごとに `./log/YYYYMMDD_app.log` へ追記する
    - ロギング失敗時でも**アプリ動作を絶対に止めない**（例外握りつぶし）
    - 起動〜終了までの**実行時間**を簡易計測（`perf_counter()`）して記録する

- 出力形式
    - `[YYYY/MM/DD hh:mm:ss][LEVEL]実行された動作`
    - 例: `[2025/08/26 21:13:45][INFO]アプリ起動: Game Bot Control`

- ローテーション/ファイル構成
    - 日付（ローカルタイム）でファイル名を切り替え
    - 同一日のログは**追記**、日付が変われば**新規ファイル**が自動生成

- 想定ユースケース
    - `log_info("テンプレ判定開始")`
    - `log_warn("設定ファイルが見つからないためデフォルト値で継続")`
    - `log_error("画像の読み込みに失敗: path=...")`
    - `log_app_start("Game Bot Control")  # 起動時に呼ぶ`
    - `log_app_exit("終了ボタン押下")      # 終了時に呼ぶ`

- 設計上の注意
    - 依存を最小化（標準ライブラリのみ）
    - 例外は**握りつぶし**（ログが原因で本処理を止めない）
    - マルチプロセス/マルチスレッド環境での**同時追記**はOSに任せる（行単位の崩れが起きる可能性は理論上あるが、
      本用途（人間の目視デバッグ）では実害が小さいため割り切る）
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

# プロジェクトルート（このファイル: modules/app_logger.py の1つ上の階層）
# 例: /path/to/project/modules/app_logger.py -> /path/to/project がROOT
ROOT = Path(__file__).resolve().parent.parent

# アプリケーションの**起動時刻（単調増加カウンタ）**を保持
# - time.perf_counter() はスリープやタイムゾーン変更の影響を受けない高精度タイマー
# - None の間は「未初期化」を表す
_START_MONO: Optional[float] = None


def _current_log_path() -> Path:
    """
    現在日付（ローカルタイム）に対応するログファイルの Path を返す。

    振る舞い:
        - ログディレクトリ `./log` が無ければ作成（parents=True, exist_ok=True）
        - ファイル名: `YYYYMMDD_app.log`（例: 20250826_app.log）

    Returns:
        Path: 追記対象のログファイルパス
    """
    ymd = time.strftime("%Y%m%d", time.localtime())  # ローカルタイムで日付を生成（例: "20250826"）
    log_dir = ROOT / "log"
    log_dir.mkdir(parents=True, exist_ok=True)       # ディレクトリが無ければ作成（中間ディレクトリも）
    return log_dir / f"{ymd}_app.log"


def _write(level: str, action: str) -> None:
    """
    ログ1行を書き出す**内部ユーティリティ**。

    出力形式:
        [YYYY/MM/DD hh:mm:ss][LEVEL]action

    仕様:
        - 例外は**すべて無視**（ログ失敗でアプリが止まらないようにする）
        - 文字コードは UTF-8。Windows メモ帳でも開けるようにBOMは付けない
        - LEVEL は大文字に正規化
        - 改行は Unix LF（\\n）。Windows でも特に問題なし（多くのエディタが自動で認識）

    Args:
        level (str): "INFO" / "WARN" / "ERROR" などの任意のレベル文字列（大文字化して出力）
        action (str): 人間が読んで意味が通る「実行された動作」メッセージ
    """
    try:
        # 例: "2025/08/26 21:13:45"
        ts = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
        line = f"[{ts}][{level.upper()}]{action}"

        # append モードで開く：ファイルが無ければ**自動生成**される
        with _current_log_path().open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # ログが原因でアプリを止めないため、握りつぶし
        # （ディスクフル/パーミッション等の異常でも本処理は継続する）
        pass


def log_info(action: str) -> None:
    """
    情報レベルのログを出力する。

    Args:
        action (str): 実行内容（例: "テンプレ判定開始"）
    """
    _write("INFO", action)


def log_warn(action: str) -> None:
    """
    警告レベルのログを出力する。

    Args:
        action (str): 実行内容（例: "設定ファイルが見つからないためデフォルト値で継続"）
    """
    _write("WARN", action)


def log_error(action: str) -> None:
    """
    エラーレベルのログを出力する。

    Args:
        action (str): 実行内容（例: "画像の読み込みに失敗: path=..."}）
    """
    _write("ERROR", action)


def log_app_start(app_name: str = "Game Bot Control") -> None:
    """
    アプリケーションの**起動**を記録し、以降の実行時間計測を開始する。

    - 最初の1回だけ `time.perf_counter()` の値を `_START_MONO` に保存（**冪等**）
    - 計測開始後、`log_info("アプリ起動: ...")` を出力

    Args:
        app_name (str): アプリ名や実行コンテキスト名。ログの可読性向上のためのメタ情報
    """
    global _START_MONO
    if _START_MONO is None:                 # 二重初期化を防ぐ（複数回呼んでも最初の状態を維持）
        _START_MONO = time.perf_counter()
    log_info(f"アプリ起動: {app_name}")


def log_app_exit(label: str = "終了ボタン押下") -> None:
    """
    アプリケーションの**終了**を記録する（起動以降の**累積実行時間**を含めて出力）。

    - 起動時に `log_app_start()` が呼ばれていない場合は、実行時間を `00:00:00.000` として記録
    - 実行時間の整形には `_format_duration()` を使用（`HH:MM:SS.sss`）

    Args:
        label (str): 終了トリガーの説明（例: "正常終了", "終了ボタン押下", "Ctrl+C" など）
    """
    dur = _format_duration(time.perf_counter() - _START_MONO) if _START_MONO is not None else "00:00:00.000"
    log_info(f"{label}（実行時間={dur}）")


def _format_duration(sec: float) -> str:
    """
    秒（float）を `HH:MM:SS.sss` の可読形式へ整形するヘルパー。

    例:
        3661.234 -> "01:01:01.234"

    Args:
        sec (float): 秒（小数を含む）

    Returns:
        str: "HH:MM:SS.sss"（ゼロ詰め）
    """
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - (h * 3600 + m * 60)
    return f"{h:02}:{m:02}:{s:06.3f}"
