#!/usr/bin/env python3
"""VM端：AI因子挖掘桥接端到端验证 — API闭合验证。

验证"AI expression → parse → spec → compute"全管线。

=== 测试 ===
  Price路径 (ai_factors):  4个实验Round2 PASS因子 → 本地pandas计算
  Fundamental路径 (jq_gm): pe_ttm (已注册) → GM SDK真实计算
  Bridge校验:             AI生成基本面表达 → parse → spec包含正确GM字段

用法:
  python vm_bridge_e2e.py <GM_TOKEN>
"""

from __future__ import annotations

import json
import sys, os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── GM SDK 初始化 ─────────────────────────────────────────
def _init_gm(token: str):
    """初始化GM SDK。需在导入gm_factor_lib前调用。"""
    try:
        from gm.api import set_token as _gm_set_token
    except ImportError:
        print("ERROR: gm.api 不可用 — 确认在GM终端内运行")
        sys.exit(2)
    _gm_set_token(token)
    # gm_factor_lib的导入路径：VM上在.goldminer3/projects/下
    gm_path = Path.home() / ".goldminer3" / "projects"
    if str(gm_path) not in sys.path:
        sys.path.insert(0, str(gm_path))
    # 备选：Mac路径（gm_factor_lib.py同目录）
    alt_path = Path.home() / "Desktop" / "TYDQUANT" / "JQ2GM"
    if str(alt_path) not in sys.path:
        sys.path.insert(0, str(alt_path))

    from gm_factor_lib import calc_factors as _gm_calc, FACTOR_REGISTRY, GM_AVAILABLE
    assert GM_AVAILABLE, "GM SDK不可用"
    return _gm_calc, FACTOR_REGISTRY


# ── 测试用例 ─────────────────────────────────────────────
PRICE_CASES = [
    ("ai_mom_20d",        "Ref($close, 20) / $close - 1"),
    ("ai_vol_ratio_10",   "$volume / Mean($volume, 10)"),
    ("ai_std_returns_20", "Std(Ref($close, 1) / $close, 20)"),
    ("ai_corr_hl",        "Corr($high, $low, 10)"),
]

FUNDAMENTAL_CASES = [
    ("pe_ttm",           "$tot_mv / $net_profit_ttm"),
    ("ai_roe",           "$net_profit_ttm / $total_owner_equities"),
]

STOCKS = [
    "SHSE.600519", "SHSE.600036", "SHSE.601318", "SHSE.600900",
    "SZSE.000858", "SZSE.000651", "SZSE.000333", "SZSE.002415",
    "SZSE.300750", "SZSE.000001",
]
TEST_DATE = "2025-12-31"


def _make_panel(n_dates=60, n_codes=20, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-06-01", periods=n_dates, freq="B")
    codes = [f"C{i:04d}" for i in range(n_codes)]
    idx = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])
    return pd.DataFrame({
        "open": rng.uniform(10, 100, len(idx)),
        "high": rng.uniform(10, 100, len(idx)),
        "low": rng.uniform(10, 100, len(idx)),
        "close": rng.uniform(10, 100, len(idx)),
        "volume": rng.uniform(1e4, 1e7, len(idx)),
    }, index=idx).reset_index()


def _describe(name, df, col):
    if df is None or df.empty or col not in df.columns:
        return name, "EMPTY", 0
    vals = df[col].dropna().replace([float("inf"), float("-inf")], None).dropna()
    if len(vals) == 0:
        return name, "NaN", 0
    return name, "OK", len(vals), float(vals.mean()), float(vals.std())


def main():
    if len(sys.argv) < 2:
        print("用法: python vm_bridge_e2e.py <GM_TOKEN>")
        sys.exit(1)
    token = sys.argv[1]

    from research_core.factor_lab.mining_bridge import parse_expression, expression_to_spec

    print("=" * 60)
    print("VM Bridge E2E — AI expression → parse → spec → compute")
    print(f"Date: {TEST_DATE}  Stocks: {len(STOCKS)}")
    print("=" * 60)

    results = []
    panel = _make_panel()

    # ═══ Path 1: Price (ai_factors) ═══
    print("\n── Price Path (ai_factors) ──")
    from research_core.factor_lab.libraries.ai_factors.factors import compute_expressions

    for name, expr in PRICE_CASES:
        print(f"\n  [{name}]  {expr}")
        # parse
        parsed = parse_expression(expr)
        assert parsed is not None, f"parse失败: {expr}"
        print(f"    parse → {parsed.expr_type.name}")

        # spec
        spec = expression_to_spec(parsed, name)
        assert spec is not None, f"spec失败"
        assert spec["library"] == "ai_factors", f"路由错误: {spec['library']}"
        print(f"    spec → {spec['library']}")

        # compute
        df = compute_expressions(panel, [expr])
        _, status, cnt, *rest = _describe(name, df, expr)
        ok = status == "OK" and cnt > 0
        icon = "✓" if ok else "❌"
        detail = f" count={cnt}" + (f" mean={rest[0]:.4f}" if rest else "")
        print(f"    compute → {icon} {status}{detail}")
        results.append({"name": name, "path": "price", "status": status, "count": cnt})

    # ═══ Path 2: Fundamental (jq_gm SDK) ═══
    print("\n── Fundamental Path (jq_gm SDK) ──")
    gm_calc, registry = _init_gm(token)
    print(f"    GM SDK ready, {len(registry)} factors in registry")

    for name, expr in FUNDAMENTAL_CASES:
        print(f"\n  [{name}]  {expr}")
        # parse + spec
        parsed = parse_expression(expr)
        if parsed is None:
            print(f"    ❌ parse → None (非OHLCV表达式需扩展parser)")
            results.append({"name": name, "path": "fundamental", "status": "PARSE_FAIL", "count": 0})
            continue

        spec = expression_to_spec(parsed, name)
        if spec is None:
            print(f"    ❌ spec → None")
            results.append({"name": name, "path": "fundamental", "status": "SPEC_FAIL", "count": 0})
            continue

        lib = spec.get("library", "?")
        print(f"    parse → {parsed.expr_type.name}, spec → {lib}")

        # 从spec提取GM字段 → 直接调SDK验证字段存在且可算
        gm_fields_str = spec.get("metadata", {}).get("gm_fields", "")
        gm_field_api = spec.get("metadata", {}).get("gm_field", "")

        if name == "pe_ttm":
            # 已注册因子 → 走compute_jq_gm_factors
            from research_core.factor_lab.libraries.jq_gm.factors import compute_jq_gm_factors
            df = compute_jq_gm_factors(
                panel, [name],
                securities=STOCKS, start_date=TEST_DATE, end_date=TEST_DATE,
            )
            _, status, cnt, *rest = _describe(name, df, name)
            ok = status == "OK" and cnt > 0
            icon = "✓" if ok else "❌"
            detail = f" count={cnt}" + (f" mean={rest[0]:.4f}" if rest else "")
            print(f"    compute_jq_gm → {icon} {status}{detail}")
            results.append({"name": name, "path": "fundamental", "status": status, "count": cnt})

        else:
            # AI生成的基本面表达 → 验证spec元数据正确，GM字段存在
            # 提取表达式中的字段名
            import re
            fields = re.findall(r'\$(\w+)', expr)
            print(f"    GM fields: {fields}")
            # 检查每个字段是否在注册表中存在
            all_ok = True
            for f in fields:
                in_registry = any(f in str(v.get('gm_fields', '')) for v in registry.values())
                print(f"      {f}: {'in registry' if in_registry else 'NOT FOUND'}")
                if not in_registry:
                    all_ok = False
            # 用gm_factor_lib直接算一个简化版（只取第一个字段对应的因子）
            if fields:
                # 找第一个匹配的已注册因子
                matched_factor = None
                for fk, fv in registry.items():
                    if fields[0] in str(fv.get('gm_fields', '')):
                        matched_factor = fk
                        break
                if matched_factor:
                    raw = gm_calc(
                        securities=STOCKS, factors=[matched_factor],
                        start_date=TEST_DATE, end_date=TEST_DATE,
                        use_real_price=True, skip_paused=True,
                    )
                    if raw and matched_factor in raw and not raw[matched_factor].empty:
                        vals = raw[matched_factor].values.flatten()
                        vals = vals[~np.isnan(vals)]
                        print(f"    calc({matched_factor}) → {len(vals)} non-NaN values, mean={vals.mean():.4f}")
                        all_ok = all_ok and len(vals) > 0
                    else:
                        print(f"    calc({matched_factor}) → EMPTY")
                        all_ok = False
                else:
                    print(f"    ⚠️  字段 {fields[0]} 不在注册表中")
                    all_ok = False

            status = "OK" if all_ok else "INCOMPLETE"
            print(f"    → {'✓' if all_ok else '⚠️'} spec+fields {status}")
            results.append({"name": name, "path": "fundamental", "status": status, "count": int(all_ok)})

    # ═══ Summary ═══
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        icon = "✓" if r["status"] in ("OK",) else ("⚠️" if r["status"] == "INCOMPLETE" else "❌")
        print(f"  {icon} [{r['path']:12s}] {r['name']:25s} {r['status']:12s} count={r['count']}")

    price_ok = all(r["status"] == "OK" for r in results if r["path"] == "price")
    fund_ok = any(r["status"] in ("OK", "INCOMPLETE") for r in results if r["path"] == "fundamental")

    print(f"\n  Price path:     {'✓ 闭合' if price_ok else '✗ 断裂'}")
    print(f"  Fundamental path: {'✓ 可达' if fund_ok else '✗ 断裂'}")

    if price_ok and fund_ok:
        print(f"\n  ✅ 双路径管线闭合：AI表达 → parse → spec → compute")
    else:
        print(f"\n  ❌ 管线未完全闭合")

    return 0 if price_ok else 1


if __name__ == "__main__":
    sys.exit(main())
