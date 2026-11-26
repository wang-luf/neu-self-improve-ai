"""Compare two AgentFlow Bamboogle runs question by question (judge verdicts and planner behaviour).

Usage: python compare_agentflow_runs.py BASE_DIR TRAINED_DIR
"""

import json
import sys
from pathlib import Path


def load(run_dir):
    run_dir = Path(run_dir)
    wrong = set(map(int, json.loads((run_dir / "final_scores_direct_output.json").read_text())["wrong_pids"]))
    stats = {}
    for i in range(125):
        r = json.loads((run_dir / f"output_{i}.json").read_text())
        memory = r.get("memory") or {}
        tools = [s.get("tool_name", "") for s in memory.values() if isinstance(s, dict)]
        stats[i] = {
            "correct": i not in wrong,
            "steps": len(tools),
            "searches": sum(t == "Local_Wikipedia_Search_Tool" for t in tools),
            "bad_tool": sum(t not in {"Local_Wikipedia_Search_Tool", "Generalist_Solution_Generator_Tool",
                                     "Python_Code_Generator_Tool"} for t in tools),
            "answer_tag": "<answer>" in str(r.get("direct_output", "")),
        }
    return stats


def summary(name, stats):
    n = len(stats)
    print(f"{name:10s} correct={sum(s['correct'] for s in stats.values()):3d}  steps/q={sum(s['steps'] for s in stats.values()) / n:.2f}"
          f"  searches/q={sum(s['searches'] for s in stats.values()) / n:.2f}"
          f"  unknown-tool steps={sum(s['bad_tool'] for s in stats.values())}"
          f"  answers with <answer> tag={sum(s['answer_tag'] for s in stats.values())}")


base, trained = load(sys.argv[1]), load(sys.argv[2])
summary("base", base)
summary("trained", trained)
gained = [i for i in range(125) if trained[i]["correct"] and not base[i]["correct"]]
lost = [i for i in range(125) if base[i]["correct"] and not trained[i]["correct"]]
print(f"both correct={sum(base[i]['correct'] and trained[i]['correct'] for i in range(125))}  gained={len(gained)} {gained}  lost={len(lost)} {lost}")
