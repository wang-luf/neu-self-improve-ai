"""Print one AgentFlow trajectory: planner sub-goals, tool calls, verifier decisions, final answer.

Usage: python show_trace.py OUTPUT_JSON
"""

import json
import sys

r = json.load(open(sys.argv[1]))
print("Question:", r["query"].split("When ready")[0].strip())
print("Gold answer:", r["answer"])
for name, step in (r.get("memory") or {}).items():
    print(f"\n{name}: tool = {step.get('tool_name')}")
    print(f"  planner sub-goal: {step.get('sub_goal')}")
    print(f"  tool command:     {step.get('command')}")
    print(f"  tool result:      {str(step.get('result'))[:300]}")
for key in sorted(k for k in r if k.startswith("verifier_") and k.endswith("_response")):
    print(f"\n{key}: {str(r[key])[-250:]}")
print("\nFinal output:", str(r.get("direct_output"))[-400:])
