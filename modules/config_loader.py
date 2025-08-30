# -*- coding: utf-8 -*-
"""
modules/config_loader.py

main.py から独立させた「設定読込＆検証」モジュール。

【主な責務】
- TOML 読込（既定 + ユーザー）とディープマージ
- 既定値の補完
- スキーマ検証（型・必須・値域）
- 依存ファイルの存在チェック（messages.ja.json、テンプレ画像など）
- パス正規化（プロジェクトルート基準の絶対パスへ）

【使用方法（例）】
from modules.config_loader import load_config, ConfigError
try:
    loaded = load_config()  # LoadedConfig を返す
    cfg = loaded.data       # 正規化された dict
    print(loaded.messages_path)  # 代表パス
except ConfigError as e:
    print("設定に問題があります:", e)
    for detail in getattr(e, "errors", []):
        print(" -", detail)

【CLI 自己診断】
> python -m modules.config_loader
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import sys
import tomllib
import logging

# --------------------------------------------------------------------------------------
# ロガー初期化
# --------------------------------------------------------------------------------------

def _get_logger() -> logging.Logger:
    """
    グローバルロガーを取得。
    - 既存の共通ロガー（modules.app_logger.get_app_logger）があればそれを使用。
    - 無ければ最低限のロガーを構成。
    """
    try:
        # 既存プロジェクトの共通ロガーに委譲（あれば優先）
        from modules.app_logger import get_app_logger  # type: ignore
        return get_app_logger()
    except Exception:
        # フォールバック：簡易ロガー
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s][%(levelname)s] %(message)s",
            datefmt="%Y:%m:%d %H:%M:%S",
        )
        return logging.getLogger("config_loader")

# モジュール内で使い回すロガー
lg: logging.Logger = _get_logger()


# --------------------------------------------------------------------------------------
# 例外定義
# --------------------------------------------------------------------------------------

class ConfigError(Exception):
    """
    設定の検証で発生する例外。
    - errors: 具体的なエラー明細（複数件）を保持できる。
    """
    def __init__(self, msg: str, errors: Optional[List[str]] = None) -> None:
        super().__init__(msg)
        self.errors: List[str] = errors or []


# --------------------------------------------------------------------------------------
# ルート探索・パス解決
# --------------------------------------------------------------------------------------

def find_project_root(start: Optional[Path] = None) -> Path:
    """
    プロジェクトルート（main.py もしくは config/ が存在する階層）を上位に辿って探索。
    - start: 探索開始ディレクトリ（省略時は本ファイルの親）
    - 戻り値: 推定されたプロジェクトルート Path
    """
    # 探索の基点（既定はこのファイルのディレクトリ）
    start = start or Path(__file__).resolve().parent
    cur = start

    # 10 階層まで親へ辿って探索（暴走防止）
    for _ in range(10):
        if (cur / "main.py").exists() or (cur / "config").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent

    # 最後の砦：このファイルの2つ上をルートとみなす（modules/ 配下想定）
    return Path(__file__).resolve().parents[1]


def to_abs(path_like: str | Path, base: Path) -> Path:
    """
    相対パスを base 起点で絶対パスへ変換。
    - path_like が既に絶対ならそのまま返す。
    """
    p = Path(path_like)
    return p if p.is_absolute() else (base / p).resolve()


# --------------------------------------------------------------------------------------
# TOML 読込・ディープマージ
# --------------------------------------------------------------------------------------

def load_toml(path: Path) -> Dict[str, Any]:
    """
    TOML ファイルを読み込んで dict を返す。
    - path: 読み込み対象 TOML の絶対パス
    """
    with path.open("rb") as f:
        return tomllib.load(f)


def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """
    dict 同士を再帰的にマージ（src が優先）。
    - ネストした dict は再帰マージ
    - リストやプリミティブは上書き
    """
    out = dict(dst)
    for k, v in src.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------------------
# スキーマ（想定構造）と簡易データクラス
# --------------------------------------------------------------------------------------

@dataclass
class CircleConfig:
    """
    円検出（正円テンプレ/ハフ/ハイブリッド）関連の設定。
    """
    enabled: bool                    # 円検出を行うか
    method: str                      # "hough" | "template" | "hybrid"
    sub_region: List[int]            # 検出対象のサブ領域 [x, y, w, h]
    template_path: Optional[str] = None  # テンプレ一致で用いるテンプレ画像
    mask_path: Optional[str] = None      # テンプレ一致で用いるマスク画像（任意）
    threshold: Optional[float] = None    # テンプレ一致のしきい値（0.0〜1.0 推奨）
    min_radius: Optional[int] = None     # ハフ円検出：最小半径
    max_radius: Optional[int] = None     # ハフ円検出：最大半径
    dp: Optional[float] = None           # ハフ円検出：精度パラメータ
    param1: Optional[int] = None         # ハフ円検出：Canny 上位閾値など
    param2: Optional[int] = None         # ハフ円検出：投票閾値
    min_dist: Optional[int] = None       # ハフ円検出：最小中心間距離


@dataclass
class RegionConfig:
    """
    キャプチャ/探索領域の設定。
    """
    origin: List[int]                 # 大域領域（スクリーン基準）[x, y, w, h]
    search: Optional[List[int]] = None  # 追加サブ領域（origin と合成する相対座標）


@dataclass
class AppConfig:
    """
    アプリ全体の設定の一部を型で表現（任意使用）。
    """
    circle: Optional[CircleConfig] = None
    region: Optional[RegionConfig] = None
    messages_path: str = "config/messages.ja.json"  # メッセージ辞書
    templates_dir: str = "templates"                # テンプレ格納ディレクトリ
    debug_save: bool = False                        # デバッグ画像保存フラグ


# --------------------------------------------------------------------------------------
# 既定値の適用
# --------------------------------------------------------------------------------------

def apply_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    欠落キーに既定値を補充（破壊的変更を避けるためコピーして返す）。
    - messages_path / templates_dir / debug_save などの補完
    - circle / region セクションの最低限の既定
    """
    # 全体既定値
    defaults = {
        "messages_path": "config/messages.ja.json",
        "templates_dir": "templates",
        "debug_save": False,
    }

    out = dict(cfg)
    for k, v in defaults.items():
        out.setdefault(k, v)

    # circle セクションの既定
    if isinstance(out.get("circle"), dict):
        out["circle"].setdefault("method", "hough")
        out["circle"].setdefault("enabled", True)
        out["circle"].setdefault("sub_region", [0, 0, 0, 0])

    # region セクションの既定（プロジェクトの既知デフォルトがあるなら調整）
    if isinstance(out.get("region"), dict):
        out["region"].setdefault("origin", [13, 125, 1460, 874])

    return out


# --------------------------------------------------------------------------------------
# 型/必須/値域チェック
# --------------------------------------------------------------------------------------

def _ensure_list_of_ints(name: str, value: Any, length: Optional[int] = None, min_val: Optional[int] = None) -> Optional[str]:
    """
    リストが int のみで構成されているか（長さや最小値も）を検証し、問題があればメッセージを返す。
    """
    if not isinstance(value, list):
        return f"{name}: list[int] を要求します"
    if length is not None and len(value) != length:
        return f"{name}: 長さ {length} のリストを要求します（実際は {len(value)}）"
    for i, x in enumerate(value):
        if not isinstance(x, int):
            return f"{name}[{i}]: int を要求します（実際は {type(x).__name__}）"
        if min_val is not None and x < min_val:
            return f"{name}[{i}]: {min_val} 以上を要求します（実際は {x}）"
    return None


def validate_schema(cfg: Dict[str, Any]) -> List[str]:
    """
    構造・型・値域の検証を行う。
    - 問題があればエラーメッセージ（文字列）のリストを返す。
    """
    errors: List[str] = []

    # messages_path
    if "messages_path" in cfg and not isinstance(cfg["messages_path"], str):
        errors.append("messages_path: str を要求します")

    # templates_dir
    if "templates_dir" in cfg and not isinstance(cfg["templates_dir"], str):
        errors.append("templates_dir: str を要求します")

    # debug_save
    if "debug_save" in cfg and not isinstance(cfg["debug_save"], bool):
        errors.append("debug_save: bool を要求します")

    # region
    region = cfg.get("region")
    if region is not None:
        if not isinstance(region, dict):
            errors.append("region: table を要求します")
        else:
            if "origin" in region:
                e = _ensure_list_of_ints("region.origin", region["origin"], length=4, min_val=0)
                if e: errors.append(e)
            if "search" in region and region["search"] is not None:
                e = _ensure_list_of_ints("region.search", region["search"], length=4, min_val=0)
                if e: errors.append(e)

    # circle
    circle = cfg.get("circle")
    if circle is not None:
        if not isinstance(circle, dict):
            errors.append("circle: table を要求します")
        else:
            # enabled
            if "enabled" in circle and not isinstance(circle["enabled"], bool):
                errors.append("circle.enabled: bool を要求します")

            # method
            method = circle.get("method", "hough")
            if not isinstance(method, str):
                errors.append("circle.method: str を要求します")
            elif method not in ("hough", "template", "hybrid"):
                errors.append("circle.method: 'hough' | 'template' | 'hybrid' のいずれかを要求します")

            # sub_region
            if "sub_region" in circle:
                e = _ensure_list_of_ints("circle.sub_region", circle["sub_region"], length=4, min_val=0)
                if e: errors.append(e)

            # method 別の必須/範囲
            if method in ("template", "hybrid"):
                # template_path（必須）
                if not circle.get("template_path"):
                    errors.append("circle.template_path: method=template|hybrid のとき必須です")
                # threshold（任意だが範囲推奨）
                if "threshold" in circle and circle["threshold"] is not None:
                    if not isinstance(circle["threshold"], (int, float)):
                        errors.append("circle.threshold: float を要求します（0.0〜1.0推奨）")
                    elif not (0.0 <= float(circle["threshold"]) <= 1.0):
                        errors.append("circle.threshold: 0.0〜1.0 の範囲を推奨します")

            if method in ("hough", "hybrid"):
                # 主要パラメータ型チェック
                for k in ("min_radius", "max_radius", "param1", "param2", "min_dist"):
                    if k in circle and circle[k] is not None and not isinstance(circle[k], int):
                        errors.append(f"circle.{k}: int を要求します")
                if "dp" in circle and circle["dp"] is not None and not isinstance(circle["dp"], (int, float)):
                    errors.append("circle.dp: float を要求します")

    return errors


# --------------------------------------------------------------------------------------
# 依存ファイル/ディレクトリの存在検証
# --------------------------------------------------------------------------------------

def validate_dependencies(cfg: Dict[str, Any], project_root: Path) -> List[str]:
    """
    外部依存ファイルの存在検証を行う。
    - messages.ja.json（必須）
    - circle.method=template|hybrid のテンプレ/マスク
    - templates ディレクトリ（無ければ警告）
    """
    errs: List[str] = []

    # messages.ja.json は必須（ユーザー要望）
    messages_path = cfg.get("messages_path", "config/messages.ja.json")
    messages_abs = to_abs(messages_path, project_root)
    if not messages_abs.exists():
        errs.append(f"messages 辞書がありません: {messages_abs}")

    # templates ディレクトリ（無くても即エラーにはしない：警告のみ）
    templates_dir = cfg.get("templates_dir", "templates")
    templates_abs = to_abs(templates_dir, project_root)
    if not templates_abs.exists():
        lg.warning("テンプレートディレクトリが見つかりません: %s", templates_abs)

    # circle のテンプレ依存チェック
    circle = cfg.get("circle")
    if isinstance(circle, dict):
        method = circle.get("method", "hough")
        if method in ("template", "hybrid"):
            # template_path（必須）
            template_path = circle.get("template_path")
            if template_path:
                template_abs = to_abs(template_path, project_root)
                if not template_abs.exists():
                    errs.append(f"circle.template_path が見つかりません: {template_abs}")
            else:
                errs.append("circle.template_path が未設定です（method=template|hybrid）")
            # mask_path（任意）
            mask_path = circle.get("mask_path")
            if mask_path:
                mask_abs = to_abs(mask_path, project_root)
                if not mask_abs.exists():
                    errs.append(f"circle.mask_path が見つかりません: {mask_abs}")

    return errs


# --------------------------------------------------------------------------------------
# 公開 API （ロード + 正規化）
# --------------------------------------------------------------------------------------

@dataclass
class LoadedConfig:
    """
    正規化済みの設定情報をまとめたコンテナ。
    - data: 正規化された dict（パスは絶対化済み）
    - root: プロジェクトルート
    - messages_path / templates_dir: よく使う代表パス
    """
    data: Dict[str, Any]       # 正規化された設定 dict
    root: Path                 # プロジェクトルート
    messages_path: Path        # メッセージ辞書の絶対パス
    templates_dir: Path        # テンプレディレクトリの絶対パス


def load_config(
    config_dir: Optional[Path] = None,
    default_name: str = "config.default.toml",
    user_name: str = "config.toml",
    allow_env_override: bool = True,
    raise_on_error: bool = True,
) -> LoadedConfig:
    """
    設定ファイルを読み込み、ディープマージし、既定値補完・検証・依存チェックを行う。
    - config_dir: 設定ディレクトリ（省略時は <project_root>/config）
    - default_name: 既定 TOML のファイル名
    - user_name: ユーザー TOML のファイル名
    - allow_env_override: 環境変数 KANCOLLE_CONFIG によるユーザー TOML の差し替えを許可
    - raise_on_error: True ならエラー時に ConfigError を送出、False ならログ出しのみ
    - 戻り値: LoadedConfig（正規化済み設定 + 代表パス）
    """
    # プロジェクトルートの推定
    project_root = find_project_root()

    # 設定ディレクトリを決定
    cfg_dir = config_dir or (project_root / "config")

    # 読み込む TOML のパス
    default_toml = cfg_dir / default_name
    user_toml = cfg_dir / user_name

    # 環境変数によるユーザー TOML の差し替え
    if allow_env_override:
        env_path = os.environ.get("KANCOLLE_CONFIG")
        if env_path:
            user_toml = Path(env_path)

    # 既定 TOML は必須
    if not default_toml.exists():
        raise ConfigError(f"既定設定が見つかりません: {default_toml}")

    # 既定 TOML の読込
    try:
        base = load_toml(default_toml)
    except Exception as e:
        raise ConfigError(f"既定設定の読込に失敗: {default_toml} ({e})")

    # ユーザー TOML の読込（存在すればマージ）
    merged = dict(base)
    if user_toml.exists():
        try:
            user_cfg = load_toml(user_toml)
            merged = deep_merge(merged, user_cfg)
            lg.info("ユーザー設定を適用しました: %s", user_toml)
        except Exception as e:
            raise ConfigError(f"ユーザー設定の読込に失敗: {user_toml} ({e})")
    else:
        lg.info("ユーザー設定が見つかりません（既定のみ適用）: %s", user_toml)

    # 既定値補完
    merged = apply_defaults(merged)

    # スキーマ検証
    schema_errors = validate_schema(merged)
    if schema_errors:
        msg = "設定スキーマ検証に失敗しました。"
        if raise_on_error:
            # まとめて例外に格納
            raise ConfigError(msg, schema_errors)
        else:
            for e in schema_errors:
                lg.error("CONFIG ERROR: %s", e)

    # 依存ファイル検証
    dep_errors = validate_dependencies(merged, project_root)
    if dep_errors:
        msg = "依存ファイルの検証に失敗しました。"
        if raise_on_error:
            raise ConfigError(msg, dep_errors)
        else:
            for e in dep_errors:
                lg.error("CONFIG ERROR: %s", e)

    # パスの絶対化
    normalized = _normalize_paths(merged, project_root)

    # 代表パスの算出
    messages_abs = to_abs(normalized.get("messages_path", "config/messages.ja.json"), project_root)
    templates_abs = to_abs(normalized.get("templates_dir", "templates"), project_root)

    # 呼び出し側で扱いやすい形にまとめて返す
    return LoadedConfig(
        data=normalized,
        root=project_root,
        messages_path=messages_abs,
        templates_dir=templates_abs,
    )


def _normalize_paths(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """
    設定内のパス系項目を絶対パス化して返す（元 dict を破壊しない）。
    - messages_path / templates_dir / circle のテンプレ関連を正規化
    """
    out = dict(cfg)

    # メインのパス
    if "messages_path" in out and isinstance(out["messages_path"], str):
        out["messages_path"] = str(to_abs(out["messages_path"], root))
    if "templates_dir" in out and isinstance(out["templates_dir"], str):
        out["templates_dir"] = str(to_abs(out["templates_dir"], root))

    # circle セクションのテンプレ系パス
    circle = out.get("circle")
    if isinstance(circle, dict):
        if "template_path" in circle and isinstance(circle["template_path"], str):
            circle["template_path"] = str(to_abs(circle["template_path"], root))
        if "mask_path" in circle and isinstance(circle["mask_path"], str):
            circle["mask_path"] = str(to_abs(circle["mask_path"], root))
        out["circle"] = circle

    # region は配列中心のためそのまま
    return out


# --------------------------------------------------------------------------------------
# CLI / 自己診断
# --------------------------------------------------------------------------------------

def _format_errors(errors: List[str]) -> str:
    """
    エラー配列を人間向けの箇条書き文字列に整形。
    """
    return "\n".join(f"  - {e}" for e in errors)


def _selfcheck() -> int:
    """
    手動実行時の簡易自己診断（python -m modules.config_loader）。
    戻り値（終了コード）:
      0 = OK
      1 = スキーマ/依存ファイルエラー
      2 = その他の実行時エラー
    """
    try:
        # エラーはログ出力のみにして処理継続（raise_on_error=False）
        cfg = load_config(raise_on_error=False)
        lg.info("設定ロード成功: root=%s", cfg.root)
        lg.info("messages: %s", cfg.messages_path)
        lg.info("templates_dir: %s", cfg.templates_dir)
        return 0
    except ConfigError as ce:
        # 設定エラー（詳細を列挙）
        lg.error("設定エラー: %s", ce)
        if getattr(ce, "errors", None):
            for e in ce.errors:
                lg.error("  - %s", e)
        return 1
    except Exception as e:
        # 予期せぬ例外はトレースを含めて出力
        lg.exception("未知のエラー: %s", e)
        return 2


if __name__ == "__main__":
    # CLI エントリポイント
    sys.exit(_selfcheck())
