"""Auto-generate competition-ready markdown tables and summaries.

Run after experiments to produce formatted output for docs/tech_innovation.md.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TableData:
    headers: List[str]
    rows: List[List[str]]
    title: str = ""


def make_markdown_table(data: TableData) -> str:
    """Render a TableData as a GitHub-flavored markdown table."""
    lines = []
    if data.title:
        lines.append(f"**{data.title}**\n")
    # Header
    lines.append("| " + " | ".join(data.headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(data.headers)) + " |")
    # Rows
    for row in data.rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def sample_efficiency_table(results: Dict[str, list], train_sizes: List[int]) -> str:
    """Generate sample efficiency comparison table."""
    headers = ["Model"] + [f"n={sz}" for sz in train_sizes] + ["Min to 70%"]

    # Find min_n for each model
    rows = []
    for name, accs in results.items():
        row = [name]
        for acc in accs:
            row.append(f"{acc:.1%}")
        min_n = None
        for sz, acc in zip(train_sizes, accs):
            if acc >= 0.70 and min_n is None:
                min_n = sz
        row.append(str(min_n) if min_n is not None else f">{train_sizes[-1]}")
        rows.append(row)

    return make_markdown_table(TableData(
        title="Sample Efficiency: Training samples needed for 70% accuracy",
        headers=headers,
        rows=rows,
    ))


def rotation_equivariance_table(
    results: Dict[str, dict],
    rotation_names: List[str],
) -> str:
    """Generate rotation equivariance comparison table."""
    headers = ["Model", "Orig Acc"] + [f"{r} Acc" for r in rotation_names] + ["Avg Drop", "Robustness"]
    rows = []
    for name, data in results.items():
        orig = data["orig"]
        rot_accs = [data[r] for r in rotation_names]
        avg_drop = orig - sum(rot_accs) / len(rot_accs)
        robustness = (orig - avg_drop) / orig if orig > 0 else 0
        row = [name, f"{orig:.1%}"] + [f"{a:.1%}" for a in rot_accs] + \
              [f"{avg_drop:.1%}", f"{robustness:.2%}"]
        rows.append(row)

    return make_markdown_table(TableData(
        title="Rotation Equivariance: Accuracy drop on rotated test data",
        headers=headers,
        rows=rows,
    ))


def throughput_table(configs: List[dict]) -> str:
    """Generate throughput comparison table."""
    headers = ["Config", "StandardMLP", "CubeMLP", "Ratio", "Winner"]
    rows = []
    for c in configs:
        rows.append([
            c["desc"],
            f"{c['std_ms']:.1f}ms",
            f"{c['cube_ms']:.1f}ms",
            f"{c['ratio']:.2f}x",
            c["winner"],
        ])

    return make_markdown_table(TableData(
        title="Training Throughput at Scale (forward+backward+update)",
        headers=headers,
        rows=rows,
    ))


def convergence_table(histories: Dict[str, List[dict]]) -> str:
    """Generate convergence comparison table."""
    headers = ["Model", "Final Acc", "Time to 60%", "Time to 75%", "Total Time"]
    rows = []

    baseline_t60 = None
    for name, log in histories.items():
        t60 = None
        t75 = None
        for entry in log:
            if t60 is None and entry["test_acc"] >= 0.60:
                t60 = entry["wall_time_s"]
            if t75 is None and entry["test_acc"] >= 0.75:
                t75 = entry["wall_time_s"]
        if baseline_t60 is None:
            baseline_t60 = t60

        t60_str = f"{t60:.1f}s" if t60 else "N/A"
        t75_str = f"{t75:.1f}s" if t75 else "N/A"
        rows.append([
            name,
            f"{log[-1]['test_acc']:.1%}",
            t60_str,
            t75_str,
            f"{log[-1]['wall_time_s']:.1f}s",
        ])

    return make_markdown_table(TableData(
        title="Training Dynamics: Convergence speed comparison",
        headers=headers,
        rows=rows,
    ))


def generate_full_report(
    sample_results: Optional[Dict] = None,
    rotation_results: Optional[Dict] = None,
    throughput_configs: Optional[List] = None,
    histories: Optional[Dict] = None,
) -> str:
    """Generate a complete competition markdown report from all experiment results.

    Args:
        sample_results: {model_name: [acc_at_each_train_size]}
        rotation_results: {model_name: {"orig": acc, "U": acc, ...}}
        throughput_configs: [{"desc": ..., "std_ms": ..., "cube_ms": ..., "ratio": ..., "winner": ...}]
        histories: {model_name: [{"epoch": ..., "test_acc": ..., "wall_time_s": ...}, ...]}

    Returns:
        Complete markdown string.
    """
    sections = ["# Cube Permutation Operator — Validation Results\n"]

    if sample_results is not None:
        sections.append(sample_efficiency_table(
            sample_results["results"],
            sample_results.get("train_sizes", [30, 60, 120, 240, 480, 960]),
        ))

    if rotation_results is not None:
        sections.append(rotation_equivariance_table(
            rotation_results,
            rotation_results.get("rotation_names", ["U", "R", "F"]),
        ))

    if throughput_configs is not None:
        sections.append(throughput_table(throughput_configs))

    if histories is not None:
        sections.append(convergence_table(histories))

    return "\n\n".join(sections)


if __name__ == "__main__":
    # Demo: print empty table structures
    print("Report generator ready.")
    print(convergence_table({"Example": [
        {"epoch": 0, "test_acc": 0.30, "wall_time_s": 1.0},
        {"epoch": 10, "test_acc": 0.65, "wall_time_s": 11.0},
    ]}))
