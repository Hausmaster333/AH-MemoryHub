from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.evaluation import internal_m3


def main() -> None:
    results = ROOT / "docs" / "testing" / "results"
    target = results / "m1-m2-m3-nitro-v3.json"
    report = json.loads(target.read_text(encoding="utf-8"))
    m1 = report["M1"]
    roles = [role for role in ("SUBJECT", "OBJECT", "LOCATION", "TIME", "TOOL", "STATE") if role in m1]
    literal_all = sum(m1[role]["f1"] * (2 if role in {"SUBJECT", "OBJECT"} else 1) for role in roles)
    literal_required = 2 * m1["SUBJECT"]["f1"] + 2 * m1["OBJECT"]["f1"] + m1["LOCATION"]["f1"]
    m1["weighted_f1"] = literal_all
    m1["normalized_weighted_f1"] = literal_all / sum(2 if role in {"SUBJECT", "OBJECT"} else 1 for role in roles)
    m1["required_roles_weighted_f1"] = literal_required
    m1["normalized_required_roles_weighted_f1"] = literal_required / 5
    report["M2_engine_smoke"] = report["M2"]
    report["M2"] = json.loads((results / "m2-e2e-v1.json").read_text(encoding="utf-8"))
    report["M3"] = internal_m3()
    report["metric_semantics"] = {
        "M1": "Primary values are literal unnormalized sums from the hackathon formula; normalized values are diagnostics.",
        "M2": "20 end-to-end causal questions through admission, AH hypernodes, query logic, ignition and evidence answer.",
        "M2_engine_smoke": "The former 100-chain result is retained only as an engine stress/smoke test.",
        "M3": "Includes the exact 200-orphan case and the zero-incident-weight orphan condition.",
    }
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"M1_required": literal_required, "M2": {key: report["M2"][key] for key in ("passed", "total", "trace_pass_rate", "explain_score", "max_depth")}, "M3": {key: report["M3"][key] for key in ("case_count", "orphan_nodes_before_gc", "live_nodes_before", "gc_efficiency", "false_deletions")}}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
