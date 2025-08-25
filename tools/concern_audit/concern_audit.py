#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
concern_audit.py (v0.3)

使い方（引数なし実行をサポート）:
  - tools/concern_audit ディレクトリで:
      PS ...\tools\concern_audit> python concern_audit.py
    → 解析ルート = このファイルの 2 つ上のディレクトリ（通常はプロジェクトルート）
    → concerns.toml があればそれを使用
    → concern_report.md / concern_report.json をルート直下に出力

  - 明示的に指定したい場合:
      python concern_audit.py --root <path> --config <file> --out <md_path> --json <json_path>

  - 分類の当たり/外れを確認（デバッグ）:
      python concern_audit.py --debug-match
"""

from __future__ import annotations
import argparse
import ast
import io
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path, PurePosixPath

VERSION = "0.3"

# --- tomllib: Python 3.11+ 標準（3.10 以下なら tomli を使うか、設定なしで実行可） ---
try:
    import tomllib  # type: ignore
except Exception:  # pragma: no cover
    tomllib = None

# ---------- 小ユーティリティ ----------
def read_text(p: str) -> str:
    with io.open(p, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def list_py_files(root: str) -> List[str]:
    ret: List[str] = []
    for d, _, files in os.walk(root):
        d_norm = d.replace("\\", "/")
        # 隠し/venv/ビルド系は除外
        if any(seg in d_norm for seg in ["/.git", "/.venv", "/venv", "/.mypy_cache", "/__pycache__", "/build", "/dist"]):
            continue
        for name in files:
            if name.endswith(".py"):
                ret.append(os.path.join(d, name))
    return ret

def relpath(p: str, root: str) -> str:
    return os.path.relpath(p, root).replace("\\", "/")

def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x

# ---------- LOC（コメント/空行を除く概算） ----------
def count_loc(src: str) -> int:
    loc = 0
    in_triple = False
    triple_q = None
    for line in src.splitlines():
        s = line.strip()
        if not s:
            continue
        # 簡易: トリプルクォート文字列をコメント扱いでスキップ
        if not in_triple and (s.startswith('"""') or s.startswith("'''")):
            in_triple = True
            triple_q = s[:3]
            continue
        if in_triple:
            if triple_q and triple_q in s[-3:]:
                in_triple = False
            continue
        if s.startswith("#"):
            continue
        loc += 1
    return loc

# ---------- Cyclomatic Complexity（AST 簡易版） ----------
class CCVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.score = 1  # 基本複雑度
    def generic_add(self, n: int = 1): self.score += n
    def visit_If(self, node): self.generic_add(1); self.generic_visit(node)
    def visit_For(self, node): self.generic_add(1); self.generic_visit(node)
    def visit_AsyncFor(self, node): self.generic_add(1); self.generic_visit(node)
    def visit_While(self, node): self.generic_add(1); self.generic_visit(node)
    def visit_With(self, node): self.generic_visit(node)
    def visit_AsyncWith(self, node): self.generic_visit(node)
    def visit_Try(self, node): self.generic_add(len(node.handlers)); self.generic_visit(node)
    def visit_BoolOp(self, node): self.generic_add(max(0, len(getattr(node, "values", [])) - 1)); self.generic_visit(node)
    def visit_IfExp(self, node): self.generic_add(1); self.generic_visit(node)
    def visit_comprehension(self, node): self.generic_add(1); self.generic_visit(node)  # list/set/dict comp
    def visit_Match(self, node): self.generic_add(len(getattr(node, "cases", []))); self.generic_visit(node)

def cc_of_node(node: ast.AST) -> int:
    v = CCVisitor()
    v.visit(node)
    return v.score

def function_cc_list(tree: ast.AST) -> List[int]:
    vals: List[int] = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            vals.append(cc_of_node(n))
    return vals or [cc_of_node(tree)]  # 関数が無い場合はモジュール全体

# ---------- Imports / 外向き依存（Ce 概算） ----------
def imported_toplevels(tree: ast.AST) -> List[str]:
    mods: List[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for alias in n.names:
                mods.append((alias.name.split(".")[0] if alias.name else "").strip())
        elif isinstance(n, ast.ImportFrom):
            if n.level and n.level > 0:
                continue  # 相対 import は内部扱い
            mod = (n.module.split(".")[0] if n.module else "").strip()
            if mod:
                mods.append(mod)
    return sorted({m for m in mods if m})

# ---------- IO 割合（副作用の素朴な近似） ----------
IO_PATTERNS = [
    r"\bopen\(", r"cv2\.im(write|encode)\(", r"\bpyautogui\.", r"\btkinter\.", r"\bos\.startfile\(",
    r"\bsubprocess\.", r"\btime\.sleep\(", r"\bmss\.", r"\blogging\.", r"\bwin32", r"\brequests\.", r"\bsocket\."
]
IO_RE = re.compile("|".join(IO_PATTERNS))
def io_ratio(src: str, loc: int) -> float:
    if loc <= 0: return 0.0
    cnt = 0
    for line in src.splitlines():
        if IO_RE.search(line):
            cnt += 1
    return cnt / max(1, loc)

# ---------- 関心定義 ----------
@dataclass
class ConcernDef:
    name: str
    include: List[str] = field(default_factory=list)
    forbid_imports: List[str] = field(default_factory=list)

@dataclass
class Config:
    internal_modules: List[str]
    concerns: List[ConcernDef]

def load_config(path: Optional[str]) -> Config:
    # 設定が無くても実行できるフォールバック
    if path is None:
        return Config(
            internal_modules=["modules", "detectors", "ui", "core", "tools"],
            concerns=[ConcernDef("runner", ["**/main.py"], [])]
        )
    if tomllib is None:
        raise RuntimeError("Python 3.11+（tomllib）または tomli が必要です。設定ファイルを使わない場合は concerns.toml を省略してください。")

    with open(path, "rb") as f:
        data = tomllib.load(f)

    internal = data.get("general", {}).get("internal_modules", []) or []

    concerns: List[ConcernDef] = []

    # 1) 推奨: [concern.<name>] というネスト構造（TOML標準）
    if isinstance(data.get("concern"), dict):
        for name, v in data["concern"].items():
            concerns.append(
                ConcernDef(
                    name=name,
                    include=list(v.get("include", [])),
                    forbid_imports=list(v.get("forbid_imports", [])),
                )
            )
    # 2) 後方互換: "concern.<name>" というフラットキーを許容
    for k, v in data.items():
        if isinstance(k, str) and k.startswith("concern.") and isinstance(v, dict):
            name = k.split(".", 1)[1]
            # すでに1)で入っていれば重複回避
            if not any(c.name == name for c in concerns):
                concerns.append(
                    ConcernDef(
                        name=name,
                        include=list(v.get("include", [])),
                        forbid_imports=list(v.get("forbid_imports", [])),
                    )
                )

    return Config(internal_modules=internal, concerns=concerns)

# ---------- パターン照合（** を正しく解釈） ----------
def path_matches(path_rel: str, pattern: str) -> bool:
    """
    PurePosixPath.match を使い、glob の ** を正しく解釈。
    さらに root 直下ファイルも拾えるよう、必要に応じて '**/' を前置して再試行。
    """
    patt = pattern.replace("\\", "/")
    p = PurePosixPath(path_rel)
    if p.match(patt):
        return True
    if not patt.startswith("**/"):
        if p.match(f"**/{patt}"):
            return True
    return False

# ---------- 分類（プラグマ優先 → パターン） ----------
def classify_concern_and_reason(path_rel: str, src_head: str, cfg: Config) -> Tuple[str, str]:
    # 先頭40行でプラグマを探索
    for line in src_head.splitlines()[:40]:
        m = re.search(r"#\s*concern\s*:\s*([a-zA-Z0-9_\-]+)", line)
        if m:
            name = m.group(1).strip().lower()
            return name, f"pragma: {name}"
    # パターン分類（** 対応）
    for c in cfg.concerns:
        for pat in c.include:
            if path_matches(path_rel, pat):
                return c.name, f"pattern: {c.name} <= {pat}"
    return "unclassified", "pattern: (no match)"

def classify_concern(path_rel: str, src_head: str, cfg: Config) -> str:
    return classify_concern_and_reason(path_rel, src_head, cfg)[0]

# ---------- 計測 ----------
@dataclass
class FileMetrics:
    path_rel: str
    concern: str
    loc: int
    f_cc: List[int]
    ce_external: int
    io_ratio: float
    imports: List[str]

def measure_file(path_abs: str, path_rel: str, cfg: Config) -> FileMetrics:
    src = read_text(path_abs)
    head = "\n".join(src.splitlines()[:40])
    concern = classify_concern(path_rel, head, cfg)
    loc = count_loc(src)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # 解析不能でも最低限は埋める
        return FileMetrics(path_rel, concern, loc, [1], 0, io_ratio(src, loc), [])
    f_cc = function_cc_list(tree)
    imports = imported_toplevels(tree)
    ce_ext = sum(1 for m in imports if m not in cfg.internal_modules)
    return FileMetrics(path_rel, concern, loc, f_cc, ce_ext, io_ratio(src, loc), imports)

# ---------- 集計と指数 ----------
@dataclass
class ConcernSummary:
    files: List[FileMetrics] = field(default_factory=list)
    loc: int = 0
    cc95: float = 0.0
    ce_avg: float = 0.0
    io_avg: float = 0.0

def p95(vals: List[float]) -> float:
    if not vals: return 0.0
    vals_sorted = sorted(vals)
    k = 0.95 * (len(vals_sorted) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(vals_sorted[int(k)])
    return float(vals_sorted[f] + (vals_sorted[c] - vals_sorted[f]) * (k - f))

def entropy_H(loc_by_concern: Dict[str, int]) -> float:
    total = sum(loc_by_concern.values()) or 1
    H = 0.0
    for v in loc_by_concern.values():
        p = v / total
        if p > 0:
            H -= p * math.log2(p)
    return H

def compute_skdi(total_loc: int, cc95_overall: float, fanout_avg: float, H: float, io_avg: float) -> float:
    nLOC = clamp01((total_loc - 300) / 700)          # 300〜1000で0→1
    nCC  = clamp01((cc95_overall - 10) / 10)         # 10〜20で0→1
    nFan = clamp01((fanout_avg - 10) / 20)           # 10〜30で0→1
    nEnt = clamp01((H - 1.0) / 1.0)                  # 1.0〜2.0で0→1
    nIO  = clamp01((io_avg - 0.2) / 0.5)             # 0.2〜0.7で0→1
    # 変更相関/状態は簡易に定数（必要なら拡張）
    nChurnSep, nState = 0.5, 0.3
    w = dict(LOC=0.15, CC=0.15, Fan=0.10, Ent=0.15, IO=0.10, Churn=0.15, State=0.10, Misc=0.10)
    return (
        w["LOC"]*nLOC + w["CC"]*nCC + w["Fan"]*nFan + w["Ent"]*nEnt + w["IO"]*nIO +
        w["Churn"]*nChurnSep + w["State"]*nState + w["Misc"]*0.4
    )

# ---------- レポート生成 ----------
def generate_reports(root: str, cfg: Config, out_md: str, out_json: str, debug_match: bool=False) -> None:
    files = list_py_files(root)

    # ---- デバッグ: まず分類の当たり/外れを出力（希望時）
    if debug_match:
        print(f"[debug] matcher = PurePosixPath.match, VERSION={VERSION}")
        for p in files:
            rel = relpath(p, root)
            head = "\n".join(read_text(p).splitlines()[:40])
            concern, reason = classify_concern_and_reason(rel, head, cfg)
            print(f"[debug] {rel}  ->  {concern}  ({reason})")

    metrics: List[FileMetrics] = []
    for p in files:
        m = measure_file(p, relpath(p, root), cfg)
        metrics.append(m)

    # 集計
    by: Dict[str, ConcernSummary] = {}
    for m in metrics:
        s = by.setdefault(m.concern, ConcernSummary())
        s.files.append(m)
    for k, s in by.items():
        s.loc = sum(f.loc for f in s.files)
        all_cc = [cc for f in s.files for cc in f.f_cc]
        s.cc95 = p95(all_cc)
        s.ce_avg = statistics.mean([f.ce_external for f in s.files]) if s.files else 0.0
        s.io_avg = statistics.mean([f.io_ratio for f in s.files]) if s.files else 0.0

    total_loc = sum(s.loc for s in by.values())
    H = entropy_H({k: s.loc for k, s in by.items() if k != "unclassified"})
    cc95_overall = p95([cc for s in by.values() for f in s.files for cc in f.f_cc])
    fanout_avg = statistics.mean([f.ce_external for f in metrics]) if metrics else 0.0
    io_avg = statistics.mean([f.io_ratio for f in metrics]) if metrics else 0.0
    skdi = compute_skdi(total_loc, cc95_overall, fanout_avg, H, io_avg)

    # 禁止依存の違反抽出
    violations: List[Tuple[str, str]] = []
    forbid_map = {c.name: set(c.forbid_imports) for c in cfg.concerns}
    for f in metrics:
        forbids = forbid_map.get(f.concern, set())
        viol = sorted(set(f.imports) & forbids)
        for v in viol:
            violations.append((f.path_rel, v))

    # JSON
    data = dict(
        total_loc=total_loc, concerns=list(by.keys()), entropy_H=H,
        cc95_overall=cc95_overall, fanout_avg=fanout_avg, io_avg=io_avg, skdi=skdi,
        per_concern={k: dict(
            loc=v.loc, cc95=v.cc95, ce_avg=v.ce_avg, io_avg=v.io_avg,
            files=[dict(path=f.path_rel, loc=f.loc, cc95=max(f.f_cc),
                        ce=f.ce_external, io=f.io_ratio, concern=f.concern)
                   for f in v.files]
        ) for k, v in by.items()},
        violations=[{"file": a, "import": b} for a, b in violations]
    )
    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out_json_path.as_posix(), "w", encoding="utf-8") as fj:
        json.dump(data, fj, ensure_ascii=False, indent=2)

    # Markdown
    lines: List[str] = []
    lines.append(f"# Concern Audit Report\n")
    lines.append(f"- Root: `{root}`")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"## Summary")
    lines.append(f"- Total LOC: **{total_loc}**")
    lines.append(f"- Concerns: **{len(by)}** (including `unclassified`)")
    lines.append(f"- Entropy H: **{H:.2f}**  |  CC p95: **{cc95_overall:.1f}**  |  Fan-out avg: **{fanout_avg:.1f}**  |  IO avg: **{io_avg:.2f}**")
    lines.append(f"- SKDI: **{skdi:.2f}** → {'Split (≥0.60)' if skdi>=0.60 else ('Keep (≤0.40)' if skdi<=0.40 else 'Gray zone (0.40–0.60)')}")
    lines.append("")
    lines.append("## Per Concern")
    lines.append("| Concern | Files | LOC | CC p95 | Fan-out avg | IO avg |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k, s in sorted(by.items(), key=lambda kv: (-kv[1].loc, kv[0])):
        lines.append(f"| `{k}` | {len(s.files)} | {s.loc} | {s.cc95:.1f} | {s.ce_avg:.1f} | {s.io_avg:.2f} |")
    lines.append("")
    if violations:
        lines.append("## Forbidden Import Violations")
        for a, b in violations:
            lines.append(f"- `{a}` imports **{b}**")
        lines.append("")
    lines.append("## Notes")
    lines.append("- CC/IO/Fan-out は簡易推定です。厳密な解析には radon / import-linter 等の導入を検討してください。")
    lines.append("- ファイル先頭に `# concern: <name>` と書くと手動分類できます。")
    out_md_path = Path(out_md)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out_md_path.as_posix(), "w", encoding="utf-8") as fm:
        fm.write("\n".join(lines))

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="解析ルート（既定: このファイルの2つ上のディレクトリ）")
    ap.add_argument("--config", default=None, help="関心定義ファイル（既定: <root>/concerns.toml）")
    ap.add_argument("--out", default=None, help="出力Markdown（既定: <root>/concern_report.md）")
    ap.add_argument("--json", default=None, help="出力JSON（既定: <root>/concern_report.json）")
    ap.add_argument("--debug-match", action="store_true", help="分類の当たり/外れを出力する")
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    # parents[0] = concern_audit, parents[1] = tools, parents[2] = <project-root>
    default_root = script_path.parents[2] if len(script_path.parents) >= 3 else Path(os.getcwd())

    root = Path(args.root).resolve() if args.root else default_root
    config_path = Path(args.config).resolve() if args.config else (root / "concerns.toml")
    out_md_path = Path(args.out).resolve() if args.out else (root / "concern_report.md")
    out_json_path = Path(args.json).resolve() if args.json else (root / "concern_report.json")

    cfg = load_config(config_path.as_posix() if config_path.exists() else None)

    generate_reports(root.as_posix(), cfg, out_md_path.as_posix(), out_json_path.as_posix(), debug_match=args.debug_match)
    print("OK:")
    print(f"  Version: {VERSION}")
    print(f"  Root   : {root}")
    print(f"  Config : {config_path if config_path.exists() else '(none: built-in default)'}")
    print(f"  Out MD : {out_md_path}")
    print(f"  Out JSON: {out_json_path}")

if __name__ == "__main__":
    sys.exit(main())
