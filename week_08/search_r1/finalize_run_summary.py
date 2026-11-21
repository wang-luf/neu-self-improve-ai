"""Mark a run that was stopped on purpose (stop_at_step.sh) in its run_summary.json.

The trainer exits non-zero when it is terminated, so local_train_verl.py records
"failed". If `planned_stop.json` exists and every other evidence check holds for
the planned step, record status "stopped_at_planned_step" instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    run_dir = Path(sys.argv[1])
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    stop = json.loads((run_dir / "planned_stop.json").read_text(encoding="utf-8"))
    checks = {
        "checkpoint_at_planned_step": summary["last_checkpoint_step"] == stop["stopped_at_step"],
        "lora_metadata": bool(summary["lora_metadata_files"]),
        "adapter_exported": summary["adapter_export_returncode"] == 0 and len(summary["exported_adapter_files"]) >= 2,
        "tool_calls_logged": summary["tool_calls_logged"] > 0,
        "rewards_logged": summary["rewards_logged"] > 0,
    }
    summary["planned_stop"] = stop
    summary["evidence_checks"] = checks
    summary["status"] = "stopped_at_planned_step" if all(checks.values()) else "failed"
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "last_checkpoint_step", "evidence_checks", "train_seconds")}, indent=2))


if __name__ == "__main__":
    main()
