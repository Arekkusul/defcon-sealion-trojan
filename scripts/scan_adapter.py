"""
scan_adapter.py — defensive triage scanner for LoRA adapters
=============================================================
A small, CI-friendly wrapper around the Weight Space Detection / Luong & Chen
5-feature comparative method. Point it at any LoRA adapter and a trusted benign
reference adapter (trained on the same base) and it prints a verdict and exits
with a non-zero code if the adapter looks backdoored — so it can gate a model
supply-chain pipeline.

This is a DEFENSIVE tool: it inspects weights that already exist and never
trains, poisons, or triggers anything.

Usage:
    python scripts/scan_adapter.py --adapter ./trojan-lora \\
        --benign ./adapters/benign-lora-matched

    python scripts/scan_adapter.py --adapter ./some-lora \\
        --benign ./reference-lora --json

Exit codes:
    0  clean / inconclusive
    2  likely backdoored
    3  usage or I/O error
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sovereign.adapter import (
    extract_features, feature_means, module_suspicion_scores,
)
from sovereign.spectral import FEATURE_NAMES, verdict_from_means


def scan(adapter_path, benign_path, backdoor_min=4, inconclusive_min=2, top=0):
    """Return a result dict comparing an adapter to a benign reference."""
    cand_feats = extract_features(adapter_path)
    ref_feats = extract_features(benign_path)
    if not cand_feats:
        raise ValueError(f"No LoRA attention modules found in {adapter_path}")
    if not ref_feats:
        raise ValueError(f"No LoRA attention modules found in {benign_path}")

    cand_means = feature_means(cand_feats)
    ref_means = feature_means(ref_feats)
    label, flags = verdict_from_means(
        cand_means, ref_means,
        backdoor_min=backdoor_min, inconclusive_min=inconclusive_min,
    )
    result = {
        "adapter": adapter_path,
        "reference": benign_path,
        "modules_scanned": len(cand_feats),
        "verdict": label,
        "suspicious_features": flags,
        "feature_means": dict(zip(FEATURE_NAMES, [round(x, 6) for x in cand_means])),
        "reference_means": dict(zip(FEATURE_NAMES, [round(x, 6) for x in ref_means])),
    }
    if top:
        ranked = module_suspicion_scores(cand_feats, ref_feats)[:top]
        result["hotspots"] = [
            {"module": key, "score": round(score, 4)} for key, score in ranked
        ]
    return result


def _print_human(result):
    print("=" * 60)
    print("  LoRA ADAPTER SCAN")
    print("=" * 60)
    print(f"  adapter        : {result['adapter']}")
    print(f"  reference      : {result['reference']}")
    print(f"  modules        : {result['modules_scanned']}")
    print(f"  suspicious     : {result['suspicious_features']}/5 features")
    print(f"  VERDICT        : {result['verdict'].upper()}")
    if result.get("hotspots"):
        print("  hotspots (most anomalous modules vs benign):")
        for hs in result["hotspots"]:
            print(f"    {hs['module']:<16} z-sum {hs['score']:+.3f}")
    print("=" * 60)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Scan a LoRA adapter for a backdoor.")
    parser.add_argument("--adapter", required=True, help="Adapter directory to scan.")
    parser.add_argument("--benign", required=True, help="Trusted benign reference adapter.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--backdoor-min", type=int, default=4,
                        help="Flags needed for a 'backdoored' verdict (default 4).")
    parser.add_argument("--inconclusive-min", type=int, default=2,
                        help="Flags needed for an 'inconclusive' verdict (default 2).")
    parser.add_argument("--top", type=int, default=0, metavar="N",
                        help="Also list the N most anomalous modules vs the reference.")
    args = parser.parse_args(argv)

    try:
        result = scan(args.adapter, args.benign,
                      backdoor_min=args.backdoor_min,
                      inconclusive_min=args.inconclusive_min,
                      top=args.top)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)

    return 2 if result["verdict"] == "backdoored" else 0


if __name__ == "__main__":
    sys.exit(main())
