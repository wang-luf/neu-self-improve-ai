"""Combine local evaluator summaries with Search-R1 paper reference scores.

Each `--run` is `LABEL=SUMMARY.json` or `LABEL=SUMMARY.json@PAPER_ROW`, where
PAPER_ROW is the 0-based row of `results/paper_reference.csv` that the run
should be compared against (omit when the paper has no matching number).
Writes `comparison.csv` and a grouped bar chart `comparison.png` in which each
method shows the measured score next to its paper reference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).parent
PAPER = HERE / "results" / "paper_reference.csv"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
MEASURED = "#2a78d6"
REFERENCE = "#b9b8b3"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare measured Bamboogle summaries against paper targets.")
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=SUMMARY.json[@PAPER_ROW]")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()

    with PAPER.open(encoding="utf-8", newline="") as handle:
        paper_rows = list(csv.DictReader(handle))

    rows: list[dict[str, object]] = []
    for item in args.run:
        if "=" not in item:
            parser.error(f"--run must be LABEL=SUMMARY.json[@PAPER_ROW], got: {item}")
        label, raw = item.split("=", 1)
        path, _, paper_index = raw.partition("@")
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        paper = paper_rows[int(paper_index)] if paper_index else None
        rows.append(
            {
                "label": label,
                "mode": summary.get("mode", "unknown"),
                "model": summary.get("model"),
                "adapter": summary.get("adapter"),
                "examples": int(summary["examples"]),
                "correct": int(summary["correct"]),
                "exact_match": round(float(summary["exact_match"]), 4),
                "mean_search_calls": round(float(summary.get("mean_search_calls", 0.0)), 3),
                "paper_reference_em": float(paper["accuracy"]) if paper else "",
                "paper_reference_correct": int(paper["correct_of_125"]) if paper else "",
                "paper_reference_source": f"{paper['method']} ({paper['algorithm']}), {paper['source']}" if paper else "",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 1.6), 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, row in enumerate(rows):
        ax.bar(i - width / 2 - 0.01, row["exact_match"], width, color=MEASURED,
                          label="Measured (this run)" if i == 0 else None)
        ax.text(i - width / 2 - 0.01, row["exact_match"] + 0.004, f"{row['exact_match']:.3f}\n{row['correct']}/125",
                ha="center", va="bottom", fontsize=8, color=TEXT_PRIMARY)
        if row["paper_reference_em"] != "":
            ax.bar(i + width / 2 + 0.01, row["paper_reference_em"], width, color=REFERENCE,
                   label="Paper reference" if "Paper reference" not in ax.get_legend_handles_labels()[1] else None)
            ax.text(i + width / 2 + 0.01, row["paper_reference_em"] + 0.004, f"{row['paper_reference_em']:.3f}",
                    ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    top = max([r["exact_match"] for r in rows] + [r["paper_reference_em"] or 0 for r in rows])
    ax.set_ylim(0, max(0.1, top * 1.25))
    ax.set_ylabel("Bamboogle exact match (125 questions)", color=TEXT_SECONDARY)
    ax.set_xticks(range(len(rows)), [str(r["label"]) for r in rows], rotation=20, ha="right", color=TEXT_PRIMARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Bamboogle EM: measured runs vs Search-R1 paper references", loc="left", color=TEXT_PRIMARY)
    fig.tight_layout()
    png_path = args.output_dir / "comparison.png"
    fig.savefig(png_path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
