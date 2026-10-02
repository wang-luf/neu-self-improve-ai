"""A small, runnable SPADE-inspired Grid2Op environment designer.

The script designs scenarios, validates them by executing Grid2Op episodes,
and writes the Designer's memory and results to JSON. It does not train a policy.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import grid2op


ENV_NAME = "l2rpn_case14_sandbox"
PROMPT_TEMPLATE = """You are the Environment Designer for power-grid operations.
Create a Grid2Op scenario specification that is valid, executable, and one
small step more challenging than the last accepted scenario. Keep the grid
physically connected; use only valid line IDs and scheduled line outages.
Use the previous evaluation feedback to avoid scenarios that immediately
collapse or are indistinguishable from the baseline. Do not write a policy.

Return JSON with: name, level, max_steps, outages. Each outage has step and
line_id. The allowed grid is {env_name}, with {n_line} lines. Current memory:
{memory}
"""


@dataclass(frozen=True)
class Outage:
    step: int
    line_id: int


@dataclass(frozen=True)
class Scenario:
    name: str
    level: int
    max_steps: int
    outages: tuple[Outage, ...]

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["outages"] = [asdict(event) for event in self.outages]
        return result


class EnvironmentDesigner:
    """Deterministic designer with a prompt-ready interface and feedback memory."""

    # These lines are valid, non-islanding contingencies in the selected case.
    candidate_lines = (0, 1, 2, 3)

    def __init__(self, memory_path: Path, use_llm: bool = False) -> None:
        self.memory_path = memory_path
        self.use_llm = use_llm
        self.memory: list[dict[str, Any]] = []
        if memory_path.exists():
            self.memory = json.loads(memory_path.read_text())

    def prompt(self, level: int, n_line: int) -> str:
        return PROMPT_TEMPLATE.format(
            env_name=ENV_NAME,
            n_line=n_line,
            memory=json.dumps(self.memory[-3:], indent=2),
        ) + f"\nRequested complexity level: {level}."

    def propose(self, level: int) -> Scenario:
        if self.use_llm:
            return self._propose_with_llm(level)
        # Complexity grows through more contingencies and a longer horizon.
        # If the previous probe ended in game over, give the next candidate
        # more recovery time between contingencies, using the recorded feedback.
        count = min(level, len(self.candidate_lines))
        previous_failed = bool(self.memory and self.memory[-1].get("game_over"))
        spacing = 4 if previous_failed else 3
        outages = tuple(
            Outage(step=spacing * (index + 1), line_id=self.candidate_lines[index])
            for index in range(count)
        )
        return Scenario(
            name=f"case14_contingencies_level_{level}",
            level=level,
            max_steps=8 + 2 * level,
            outages=outages,
        )

    def _propose_with_llm(self, level: int) -> Scenario:
        """Ask an OpenAI-compatible chat endpoint for a constrained scenario."""
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        env = grid2op.make(ENV_NAME)
        n_line = env.n_line
        env.close()
        request_body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return only a JSON object matching the requested schema."},
                {"role": "user", "content": self.prompt(level, n_line)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        request = Request(
            f"{base_url}/chat/completions",
            data=json.dumps(request_body).encode(),
            headers={
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
        content = json.loads(payload["choices"][0]["message"]["content"])
        candidate = Scenario(
            name=str(content["name"]),
            level=level,
            max_steps=int(content["max_steps"]),
            outages=tuple(
                Outage(step=int(item["step"]), line_id=int(item["line_id"]))
                for item in content["outages"]
            ),
        )
        if not 1 <= candidate.max_steps <= 100:
            raise ValueError("LLM proposal max_steps must be between 1 and 100")
        if len(candidate.outages) < min(level, len(self.candidate_lines)):
            raise ValueError("LLM proposal did not increase outage complexity for this level")
        if candidate.max_steps < 8 + 2 * level:
            raise ValueError("LLM proposal horizon is below the requested curriculum level")
        if any(
            event.step < 1 or event.step > candidate.max_steps
            or event.line_id < 0 or event.line_id >= n_line
            for event in candidate.outages
        ):
            raise ValueError("LLM proposal contains an invalid outage step or line_id")
        return candidate

    def remember(self, result: dict[str, Any]) -> None:
        # Prompts include recent memory; don't persist prompts into memory or
        # each round's record would recursively grow without bound.
        fields = (
            "scenario", "valid", "steps_survived", "survived_fraction",
            "game_over", "peak_line_loading_rho", "feedback",
        )
        self.memory.append({key: result[key] for key in fields if key in result})
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(json.dumps(self.memory, indent=2) + "\n")


def validate_scenario(scenario: Scenario, seed: int = 7) -> dict[str, Any]:
    """Execute a fixed no-op reference agent; this is a difficulty probe only."""
    env = grid2op.make(ENV_NAME)
    try:
        observation = env.reset(seed=seed)
        if any(event.line_id < 0 or event.line_id >= env.n_line for event in scenario.outages):
            raise ValueError("outage line_id is outside the Grid2Op environment")
        total_reward = 0.0
        peak_rho = float(observation.rho.max())
        done = False
        applied: set[int] = set()
        steps = 0
        while steps < scenario.max_steps and not done:
            step_number = steps + 1
            scheduled = [
                event.line_id
                for event in scenario.outages
                if event.step == step_number and event.line_id not in applied
            ]
            action = (
                env.action_space({"set_line_status": [(line_id, -1) for line_id in scheduled]})
                if scheduled
                else env.action_space()
            )
            observation, reward, done, info = env.step(action)
            total_reward += float(reward)
            peak_rho = max(peak_rho, float(observation.rho.max()))
            applied.update(scheduled)
            steps += 1

        survived_fraction = steps / scenario.max_steps
        return {
            "scenario": scenario.to_json(),
            "valid": True,
            "steps_survived": steps,
            "survived_fraction": round(survived_fraction, 4),
            "game_over": bool(done),
            "peak_line_loading_rho": round(peak_rho, 4),
            "reference_reward_sum": round(total_reward, 4),
            "difficulty_probe": "fixed DoNothing agent; no policy training",
            "feedback": (
                "reference agent survived the horizon"
                if not done
                else "reference agent triggered game over before the horizon"
            ),
        }
    except Exception as exc:
        return {
            "scenario": scenario.to_json(),
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
            "feedback": "scenario rejected; revise outage timing or line selection",
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=4, help="number of complexity levels")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--llm", action="store_true", help="use OPENAI_API_KEY with an OpenAI-compatible endpoint")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "runs_feedback")
    args = parser.parse_args()
    if args.levels < 1:
        parser.error("--levels must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.llm and not os.environ.get("OPENAI_API_KEY"):
        parser.error("--llm requires OPENAI_API_KEY")
    designer = EnvironmentDesigner(args.output_dir / "designer_memory.json", use_llm=args.llm)
    all_results: list[dict[str, Any]] = []
    probe_env = grid2op.make(ENV_NAME)
    n_line = probe_env.n_line
    probe_env.close()

    for level in range(args.levels):
        prompt = designer.prompt(level, n_line)
        scenario = designer.propose(level)
        result = validate_scenario(scenario, seed=args.seed)
        result["designer_prompt"] = prompt
        designer.remember(result)
        all_results.append(result)
        print(
            f"level={level} outages={len(scenario.outages)} horizon={scenario.max_steps} "
            f"valid={result['valid']} survived={result.get('steps_survived', 0)} "
            f"peak_rho={result.get('peak_line_loading_rho', 'n/a')} "
            f"feedback={result['feedback']}"
        )

    results_path = args.output_dir / "experiment_results.json"
    results_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
