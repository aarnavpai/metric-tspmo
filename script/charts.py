# ruff: noqa
"""
Benchmark Chart Generator.

This script reads benchmark data using the report.py loader and generates
comprehensive visualizations using matplotlib to analyze solver performance,
accuracy, and behavior.
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Import the load_data function from report.py
sys.path.insert(0, os.path.dirname(__file__))
from report import format_bytes as format_bytes_report
from report import load_data as load_data_from_report

# Color palette for consistent, professional-looking charts
COLORS = {
    "dpll": "#10b981",  # green
    "cdcl": "#3b82f6",  # blue
    "schöning": "#f59e0b",  # amber
    "sat": "#10b981",
    "unsat": "#ef4444",
    "verified": "#10b981",
    "unverified": "#94a3b8",
}

SOLVER_ORDER = ["dpll", "cdcl", "schöning"]


def format_bytes(size: float) -> str:
    """Convert bytes to human-readable format."""
    return format_bytes_report(size)


def is_sat_problem(problem_name: str) -> bool:
    """Determine if a problem is satisfiable based on filename."""
    return problem_name.startswith("uf") and not problem_name.startswith("uuf")


def is_unsat_problem(problem_name: str) -> bool:
    """Determine if a problem is unsatisfiable based on filename."""
    return problem_name.startswith("uuf")


def load_data(result_dir: str) -> List[Dict[str, Any]]:
    """
    Load benchmark results using the report.py loader.

    Args:
        result_dir: Path to the result directory containing benchs/ and memray/

    Returns:
        A flat list of result dictionaries with added problem_category field.
    """
    # Use the load_data function from report.py
    grouped_data = load_data_from_report(result_dir)

    # Flatten the grouped data into a single list
    data = []
    for (problem_type, problem_name), rows in grouped_data.items():
        for row in rows:
            # Add problem category based on problem name
            if is_sat_problem(row["problem"]):
                row["problem_category"] = "SAT"
            elif is_unsat_problem(row["problem"]):
                row["problem_category"] = "UNSAT"
            else:
                row["problem_category"] = "OTHER"

            data.append(row)

    return data


def setup_plot_style():
    """Configure matplotlib style for professional-looking charts."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#f8fafc"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.titlesize"] = 16
    plt.rcParams["figure.titleweight"] = "bold"


def chart_1_wall_time_overview(data: List[Dict], output_dir: str):
    """1. Wall Time Overview - Box plot comparing solver wall times by problem type."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Separate SAT and UNSAT problems
    sat_data = [row for row in data if row["problem_category"] == "SAT"]
    unsat_data = [row for row in data if row["problem_category"] == "UNSAT"]

    # SAT problems
    solver_times_sat = defaultdict(list)
    for row in sat_data:
        solver_times_sat[row["solver"]].append(row["wall_time"])

    solvers_sat = [s for s in SOLVER_ORDER if s in solver_times_sat]
    times_sat = [solver_times_sat[s] for s in solvers_sat]

    bp1 = ax1.boxplot(
        times_sat,
        tick_labels=solvers_sat,
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )

    for patch, solver in zip(bp1["boxes"], solvers_sat):
        patch.set_facecolor(COLORS.get(solver, "#94a3b8"))
        patch.set_alpha(0.7)

    ax1.set_ylabel("Wall Time (seconds)", fontweight="bold")
    ax1.set_xlabel("Solver", fontweight="bold")
    ax1.set_title("SAT Problems (uf*)", fontweight="bold", color=COLORS["sat"])
    ax1.grid(True, alpha=0.3, axis="y")

    # Add statistics
    for i, solver in enumerate(solvers_sat, 1):
        times_list = solver_times_sat[solver]
        median = np.median(times_list)
        mean = np.mean(times_list)
        ax1.text(
            i,
            ax1.get_ylim()[1] * 0.95,
            f"μ={mean:.3f}s\nmed={median:.3f}s",
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # UNSAT problems
    solver_times_unsat = defaultdict(list)
    for row in unsat_data:
        solver_times_unsat[row["solver"]].append(row["wall_time"])

    solvers_unsat = [s for s in SOLVER_ORDER if s in solver_times_unsat]
    times_unsat = [solver_times_unsat[s] for s in solvers_unsat]

    bp2 = ax2.boxplot(
        times_unsat,
        tick_labels=solvers_unsat,
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )

    for patch, solver in zip(bp2["boxes"], solvers_unsat):
        patch.set_facecolor(COLORS.get(solver, "#94a3b8"))
        patch.set_alpha(0.7)

    ax2.set_ylabel("Wall Time (seconds)", fontweight="bold")
    ax2.set_xlabel("Solver", fontweight="bold")
    ax2.set_title("UNSAT Problems (uuf*)", fontweight="bold", color=COLORS["unsat"])
    ax2.grid(True, alpha=0.3, axis="y")

    # Add statistics
    for i, solver in enumerate(solvers_unsat, 1):
        times_list = solver_times_unsat[solver]
        median = np.median(times_list)
        mean = np.mean(times_list)
        ax2.text(
            i,
            ax2.get_ylim()[1] * 0.95,
            f"μ={mean:.3f}s\nmed={median:.3f}s",
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig.suptitle(
        "Solver Wall Time Distribution by Problem Type", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_wall_time_overview.png"), dpi=300)
    plt.close()
    print("✓ Generated: 01_wall_time_overview.png")


def chart_2_memory_overview(data: List[Dict], output_dir: str):
    """2. Memory Usage Overview - Bar chart of average memory by solver (excluding zero values)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    solver_memory = defaultdict(list)
    for row in data:
        # Only include non-zero memory values
        if row["memory"] > 0:
            solver_memory[row["solver"]].append(row["memory"])

    solvers = [
        s for s in SOLVER_ORDER if s in solver_memory and len(solver_memory[s]) > 0
    ]

    if not solvers:
        print("⚠ Warning: No memory data available for chart 2")
        return

    avg_memory = [np.mean(solver_memory[s]) for s in solvers]
    max_memory = [np.max(solver_memory[s]) for s in solvers]
    min_memory = [np.min(solver_memory[s]) for s in solvers]

    x = np.arange(len(solvers))
    width = 0.6

    bars = ax.bar(
        x,
        avg_memory,
        width,
        color=[COLORS.get(s, "#94a3b8") for s in solvers],
        alpha=0.8,
        label="Average",
    )

    # Add error bars showing min/max range
    errors = [
        [avg - min_val for avg, min_val in zip(avg_memory, min_memory)],
        [max_val - avg for max_val, avg in zip(max_memory, avg_memory)],
    ]
    ax.errorbar(
        x, avg_memory, yerr=errors, fmt="none", ecolor="black", capsize=5, alpha=0.5
    )

    ax.set_ylabel("Peak Memory Usage (bytes)", fontweight="bold")
    ax.set_xlabel("Solver", fontweight="bold")
    ax.set_title("Average Peak Memory Usage by Solver", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(solvers)
    ax.grid(True, alpha=0.3, axis="y")

    # Format y-axis to show human-readable memory
    y_ticks = ax.get_yticks()
    ax.set_yticklabels([format_bytes(y) for y in y_ticks])

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, avg_memory)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            format_bytes(val),
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_memory_overview.png"), dpi=300)
    plt.close()
    print("✓ Generated: 02_memory_overview.png")


def chart_3_accuracy_overview(data: List[Dict], output_dir: str):
    """3. Accuracy Overview - Stacked bar showing SAT verification vs UNSAT unverification rates."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Track totals and verified counts by problem type for each solver
    solver_stats = defaultdict(
        lambda: {
            "sat_total": 0,
            "unsat_total": 0,
            "sat_verified": 0,
            "unsat_verified": 0,
        }
    )

    for row in data:
        solver = row["solver"]
        if row["problem_category"] == "SAT":
            solver_stats[solver]["sat_total"] += 1
            if row["verified"]:
                solver_stats[solver]["sat_verified"] += 1
        elif row["problem_category"] == "UNSAT":
            solver_stats[solver]["unsat_total"] += 1
            if row["verified"]:
                solver_stats[solver]["unsat_verified"] += 1

    # Only include solvers present in the data
    solvers = [s for s in SOLVER_ORDER if s in solver_stats]

    # Compute rates (percent): SAT verification and UNSAT unverification per solver
    sat_rates = []
    unsat_unverification_rates = []
    for s in solvers:
        sat_total = solver_stats[s]["sat_total"]
        unsat_total = solver_stats[s]["unsat_total"]
        sat_verified = solver_stats[s]["sat_verified"]
        unsat_verified = solver_stats[s]["unsat_verified"]

        sat_rate = (sat_verified / sat_total * 100) if sat_total > 0 else 0.0
        unsat_verification_rate = (
            (unsat_verified / unsat_total * 100) if unsat_total > 0 else 0.0
        )
        unsat_unverification_rate = (
            100.0 - unsat_verification_rate if unsat_total > 0 else 0.0
        )

        sat_rates.append(sat_rate)
        unsat_unverification_rates.append(unsat_unverification_rate)

    x = np.arange(len(solvers))
    width = 0.6

    # Stacked bars: SAT verification rate on bottom (green), UNSAT unverification rate on top (red)
    ax.bar(
        x,
        sat_rates,
        width,
        label="SAT Verification Rate",
        color=COLORS["sat"],
        alpha=0.85,
    )
    ax.bar(
        x,
        unsat_unverification_rates,
        width,
        bottom=sat_rates,
        label="UNSAT Unverification Rate",
        color=COLORS["unsat"],
        alpha=0.85,
    )

    ax.set_ylabel("Rate (%)", fontweight="bold")
    ax.set_xlabel("Solver", fontweight="bold")
    ax.set_title("SAT Verification vs UNSAT Unverification (stacked)", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(solvers)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 200)  # Max stacked value if both rates reach 100%

    # Add labels showing individual rates and total stacked height
    for i, solver in enumerate(solvers):
        sat_rate = sat_rates[i]
        unsat_unverification_rate = unsat_unverification_rates[i]
        total_stack = sat_rate + unsat_unverification_rate

        # Label above bar with total
        ax.text(
            i,
            total_stack + 2,
            f"Total: {total_stack:.0f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

        # Label inside segments if large enough
        if sat_rate > 5:
            ax.text(
                i,
                sat_rate / 2,
                f"{sat_rate:.0f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        if unsat_unverification_rate > 5:
            ax.text(
                i,
                sat_rate + unsat_unverification_rate / 2,
                f"{unsat_unverification_rate:.0f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_accuracy_overview.png"), dpi=300)
    plt.close()
    print("✓ Generated: 03_accuracy_overview.png")


def chart_4_sat_vs_unsat(data: List[Dict], output_dir: str):
    """4. SAT vs UNSAT Performance - Average solve time comparison."""
    fig, ax = plt.subplots(figsize=(10, 7))

    sat_times = defaultdict(list)
    unsat_times = defaultdict(list)

    for row in data:
        if row["problem_category"] == "SAT":
            sat_times[row["solver"]].append(row["wall_time"])
        elif row["problem_category"] == "UNSAT":
            unsat_times[row["solver"]].append(row["wall_time"])

    solvers = [s for s in SOLVER_ORDER if s in sat_times or s in unsat_times]

    x = np.arange(len(solvers))
    width = 0.35

    sat_means = [np.mean(sat_times[s]) if s in sat_times else 0 for s in solvers]
    unsat_means = [np.mean(unsat_times[s]) if s in unsat_times else 0 for s in solvers]

    bars1 = ax.bar(
        x - width / 2,
        sat_means,
        width,
        label="SAT (uf*)",
        color=COLORS["sat"],
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        unsat_means,
        width,
        label="UNSAT (uuf*)",
        color=COLORS["unsat"],
        alpha=0.8,
    )

    ax.set_ylabel("Average Wall Time (seconds)", fontweight="bold")
    ax.set_xlabel("Solver", fontweight="bold")
    ax.set_title("Average Solving Time: SAT vs UNSAT Problems", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(solvers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bars, means in [(bars1, sat_means), (bars2, unsat_means)]:
        for bar, mean in zip(bars, means):
            if mean > 0:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{mean:.3f}s",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_sat_vs_unsat.png"), dpi=300)
    plt.close()
    print("✓ Generated: 04_sat_vs_unsat.png")


def chart_5_problem_difficulty(data: List[Dict], output_dir: str):
    """5. Problem Difficulty Analysis - Time vs problem with SAT/UNSAT distinction."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Separate SAT and UNSAT
    sat_data = [row for row in data if row["problem_category"] == "SAT"]
    unsat_data = [row for row in data if row["problem_category"] == "UNSAT"]

    # SAT problems
    problem_data_sat = defaultdict(dict)
    for row in sat_data:
        problem_data_sat[row["problem"]][row["solver"]] = row["wall_time"]

    problems_sat = sorted(problem_data_sat.keys())[:50]

    for solver in SOLVER_ORDER:
        times = [problem_data_sat[p].get(solver, None) for p in problems_sat]
        x_vals = [i for i, t in enumerate(times) if t is not None]
        y_vals = [t for t in times if t is not None]

        if y_vals:
            ax1.plot(
                x_vals,
                y_vals,
                marker="o",
                markersize=3,
                alpha=0.7,
                label=solver,
                color=COLORS.get(solver, "#94a3b8"),
                linewidth=1.5,
            )

    ax1.set_xlabel("Problem Instance (SAT)", fontweight="bold")
    ax1.set_ylabel("Wall Time (seconds)", fontweight="bold")
    ax1.set_title(
        "Performance on SAT Problems (uf*)", fontweight="bold", color=COLORS["sat"]
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # UNSAT problems
    problem_data_unsat = defaultdict(dict)
    for row in unsat_data:
        problem_data_unsat[row["problem"]][row["solver"]] = row["wall_time"]

    problems_unsat = sorted(problem_data_unsat.keys())[:50]

    for solver in SOLVER_ORDER:
        times = [problem_data_unsat[p].get(solver, None) for p in problems_unsat]
        x_vals = [i for i, t in enumerate(times) if t is not None]
        y_vals = [t for t in times if t is not None]

        if y_vals:
            ax2.plot(
                x_vals,
                y_vals,
                marker="o",
                markersize=3,
                alpha=0.7,
                label=solver,
                color=COLORS.get(solver, "#94a3b8"),
                linewidth=1.5,
            )

    ax2.set_xlabel("Problem Instance (UNSAT)", fontweight="bold")
    ax2.set_ylabel("Wall Time (seconds)", fontweight="bold")
    ax2.set_title(
        "Performance on UNSAT Problems (uuf*)", fontweight="bold", color=COLORS["unsat"]
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Solver Performance Across Problem Instances", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_problem_difficulty.png"), dpi=300)
    plt.close()
    print("✓ Generated: 05_problem_difficulty.png")


def chart_6_memory_usage_line(data: List[Dict], output_dir: str):
    """6. Memory Usage Across Problems - Line plot showing memory trends."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Filter data with non-zero memory
    data_with_memory = [row for row in data if row["memory"] > 0]

    # Separate SAT and UNSAT
    sat_data = [row for row in data_with_memory if row["problem_category"] == "SAT"]
    unsat_data = [row for row in data_with_memory if row["problem_category"] == "UNSAT"]

    # SAT problems
    problem_memory_sat = defaultdict(dict)
    for row in sat_data:
        problem_memory_sat[row["problem"]][row["solver"]] = row["memory"]

    problems_sat = sorted(problem_memory_sat.keys())[:50]

    for solver in SOLVER_ORDER:
        memory = [problem_memory_sat[p].get(solver, None) for p in problems_sat]
        x_vals = [i for i, m in enumerate(memory) if m is not None]
        y_vals = [m for m in memory if m is not None]

        if y_vals:
            ax1.plot(
                x_vals,
                y_vals,
                marker="o",
                markersize=3,
                alpha=0.7,
                label=solver,
                color=COLORS.get(solver, "#94a3b8"),
                linewidth=1.5,
            )

    ax1.set_xlabel("Problem Instance (SAT)", fontweight="bold")
    ax1.set_ylabel("Peak Memory Usage (bytes)", fontweight="bold")
    ax1.set_title(
        "Memory Usage on SAT Problems (uf*)", fontweight="bold", color=COLORS["sat"]
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Format y-axis
    y_ticks1 = ax1.get_yticks()
    ax1.set_yticklabels([format_bytes(y) for y in y_ticks1])

    # UNSAT problems
    problem_memory_unsat = defaultdict(dict)
    for row in unsat_data:
        problem_memory_unsat[row["problem"]][row["solver"]] = row["memory"]

    problems_unsat = sorted(problem_memory_unsat.keys())[:50]

    for solver in SOLVER_ORDER:
        memory = [problem_memory_unsat[p].get(solver, None) for p in problems_unsat]
        x_vals = [i for i, m in enumerate(memory) if m is not None]
        y_vals = [m for m in memory if m is not None]

        if y_vals:
            ax2.plot(
                x_vals,
                y_vals,
                marker="o",
                markersize=3,
                alpha=0.7,
                label=solver,
                color=COLORS.get(solver, "#94a3b8"),
                linewidth=1.5,
            )

    ax2.set_xlabel("Problem Instance (UNSAT)", fontweight="bold")
    ax2.set_ylabel("Peak Memory Usage (bytes)", fontweight="bold")
    ax2.set_title(
        "Memory Usage on UNSAT Problems (uuf*)",
        fontweight="bold",
        color=COLORS["unsat"],
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Format y-axis
    y_ticks2 = ax2.get_yticks()
    ax2.set_yticklabels([format_bytes(y) for y in y_ticks2])

    fig.suptitle(
        "Peak Memory Usage Across Problem Instances", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_memory_usage_line.png"), dpi=300)
    plt.close()
    print("✓ Generated: 06_memory_usage_line.png")


def chart_7_dpll_statistics(data: List[Dict], output_dir: str):
    """7. DPLL Statistics - Analyze decisions, propagations, and pure literals."""
    dpll_data = [row for row in data if row["solver"] == "dpll"]

    if not dpll_data:
        print("⚠ Skipped: 07_dpll_statistics.png (no DPLL data)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract stats
    decisions = [row["stats"].get("decisions", 0) for row in dpll_data if row["stats"]]
    unit_props = [
        row["stats"].get("unit_propagations", 0) for row in dpll_data if row["stats"]
    ]
    pure_lits = [
        row["stats"].get("pure_literals", 0) for row in dpll_data if row["stats"]
    ]
    wall_times = [row["wall_time"] for row in dpll_data if row["stats"]]

    # Color by problem type
    colors = [
        COLORS["sat"] if is_sat_problem(row["problem"]) else COLORS["unsat"]
        for row in dpll_data
        if row["stats"]
    ]

    # Decisions vs Time
    axes[0, 0].scatter(decisions, wall_times, alpha=0.6, c=colors, s=40)
    axes[0, 0].set_xlabel("Number of Decisions", fontweight="bold")
    axes[0, 0].set_ylabel("Wall Time (seconds)", fontweight="bold")
    axes[0, 0].set_title("Decisions vs Time")
    axes[0, 0].grid(True, alpha=0.3)

    # Unit Propagations vs Time
    axes[0, 1].scatter(unit_props, wall_times, alpha=0.6, c=colors, s=40)
    axes[0, 1].set_xlabel("Unit Propagations", fontweight="bold")
    axes[0, 1].set_ylabel("Wall Time (seconds)", fontweight="bold")
    axes[0, 1].set_title("Unit Propagations vs Time")
    axes[0, 1].grid(True, alpha=0.3)

    # Pure Literals vs Time
    axes[1, 0].scatter(pure_lits, wall_times, alpha=0.6, c=colors, s=40)
    axes[1, 0].set_xlabel("Pure Literals Found", fontweight="bold")
    axes[1, 0].set_ylabel("Wall Time (seconds)", fontweight="bold")
    axes[1, 0].set_title("Pure Literals vs Time")
    axes[1, 0].grid(True, alpha=0.3)

    # Statistics distribution
    stats_data = [decisions, unit_props, pure_lits]
    bp = axes[1, 1].boxplot(
        stats_data,
        tick_labels=["Decisions", "Unit Props", "Pure Lits"],
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["dpll"])
        patch.set_alpha(0.7)
    axes[1, 1].set_ylabel("Count", fontweight="bold")
    axes[1, 1].set_title("DPLL Statistics Distribution")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    # Add legend
    sat_patch = mpatches.Patch(color=COLORS["sat"], label="SAT Problems", alpha=0.6)
    unsat_patch = mpatches.Patch(
        color=COLORS["unsat"], label="UNSAT Problems", alpha=0.6
    )
    axes[0, 0].legend(handles=[sat_patch, unsat_patch], loc="upper left")

    fig.suptitle("DPLL Algorithm Statistics Analysis", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_dpll_statistics.png"), dpi=300)
    plt.close()
    print("✓ Generated: 07_dpll_statistics.png")


def chart_8_solver_speedup(data: List[Dict], output_dir: str):
    """8. Solver Speedup Comparison - Speedup relative to slowest solver."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Group by problem and type
    sat_problem_times = defaultdict(dict)
    unsat_problem_times = defaultdict(dict)

    for row in data:
        if row["problem_category"] == "SAT":
            sat_problem_times[row["problem"]][row["solver"]] = row["wall_time"]
        elif row["problem_category"] == "UNSAT":
            unsat_problem_times[row["problem"]][row["solver"]] = row["wall_time"]

    # Calculate speedup for SAT problems
    sat_speedups = defaultdict(list)
    for problem, solvers in sat_problem_times.items():
        if len(solvers) < 2:
            continue
        max_time = max(solvers.values())
        for solver, time in solvers.items():
            speedup = max_time / time if time > 0 else 1
            sat_speedups[solver].append(speedup)

    solvers_sat = [s for s in SOLVER_ORDER if s in sat_speedups]
    speedup_data_sat = [sat_speedups[s] for s in solvers_sat]

    bp1 = ax1.boxplot(speedup_data_sat, tick_labels=solvers_sat, patch_artist=True)

    for patch, solver in zip(bp1["boxes"], solvers_sat):
        patch.set_facecolor(COLORS.get(solver, "#94a3b8"))
        patch.set_alpha(0.7)

    ax1.axhline(y=1, color="red", linestyle="--", alpha=0.5, label="Baseline (1x)")
    ax1.set_ylabel("Speedup Factor", fontweight="bold")
    ax1.set_xlabel("Solver", fontweight="bold")
    ax1.set_title("SAT Problems", fontweight="bold", color=COLORS["sat"])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Calculate speedup for UNSAT problems
    unsat_speedups = defaultdict(list)
    for problem, solvers in unsat_problem_times.items():
        if len(solvers) < 2:
            continue
        max_time = max(solvers.values())
        for solver, time in solvers.items():
            speedup = max_time / time if time > 0 else 1
            unsat_speedups[solver].append(speedup)

    solvers_unsat = [s for s in SOLVER_ORDER if s in unsat_speedups]
    speedup_data_unsat = [unsat_speedups[s] for s in solvers_unsat]

    bp2 = ax2.boxplot(speedup_data_unsat, tick_labels=solvers_unsat, patch_artist=True)

    for patch, solver in zip(bp2["boxes"], solvers_unsat):
        patch.set_facecolor(COLORS.get(solver, "#94a3b8"))
        patch.set_alpha(0.7)

    ax2.axhline(y=1, color="red", linestyle="--", alpha=0.5, label="Baseline (1x)")
    ax2.set_ylabel("Speedup Factor", fontweight="bold")
    ax2.set_xlabel("Solver", fontweight="bold")
    ax2.set_title("UNSAT Problems", fontweight="bold", color=COLORS["unsat"])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Solver Speedup Relative to Slowest (per problem)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "08_solver_speedup.png"), dpi=300)
    plt.close()
    print("✓ Generated: 08_solver_speedup.png")


def chart_9_cumulative_time(data: List[Dict], output_dir: str):
    """9. Cumulative Time Distribution - CDF of solve times by problem type."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # SAT problems
    for solver in SOLVER_ORDER:
        solver_times = sorted(
            [
                row["wall_time"]
                for row in data
                if row["solver"] == solver and row["problem_category"] == "SAT"
            ]
        )
        if not solver_times:
            continue

        y = np.arange(1, len(solver_times) + 1) / len(solver_times) * 100
        ax1.plot(
            solver_times,
            y,
            label=solver,
            linewidth=2.5,
            color=COLORS.get(solver, "#94a3b8"),
        )

    ax1.set_xlabel("Wall Time (seconds)", fontweight="bold")
    ax1.set_ylabel("Cumulative Percentage (%)", fontweight="bold")
    ax1.set_title("SAT Problems (uf*)", fontweight="bold", color=COLORS["sat"])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim([0, 100])

    # UNSAT problems
    for solver in SOLVER_ORDER:
        solver_times = sorted(
            [
                row["wall_time"]
                for row in data
                if row["solver"] == solver and row["problem_category"] == "UNSAT"
            ]
        )
        if not solver_times:
            continue

        y = np.arange(1, len(solver_times) + 1) / len(solver_times) * 100
        ax2.plot(
            solver_times,
            y,
            label=solver,
            linewidth=2.5,
            color=COLORS.get(solver, "#94a3b8"),
        )

    ax2.set_xlabel("Wall Time (seconds)", fontweight="bold")
    ax2.set_ylabel("Cumulative Percentage (%)", fontweight="bold")
    ax2.set_title("UNSAT Problems (uuf*)", fontweight="bold", color=COLORS["unsat"])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim([0, 100])

    # Add reference lines
    for ax in [ax1, ax2]:
        for pct in [50, 90, 95]:
            ax.axhline(y=pct, color="gray", linestyle=":", alpha=0.3)
            ax.text(
                ax.get_xlim()[1] * 0.02, pct + 2, f"{pct}%", fontsize=8, color="gray"
            )

    fig.suptitle(
        "Cumulative Distribution of Solve Times", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "09_cumulative_time.png"), dpi=300)
    plt.close()
    print("✓ Generated: 09_cumulative_time.png")


def chart_10_performance_heatmap(data: List[Dict], output_dir: str):
    """10. Performance Heatmap - Visual comparison with SAT/UNSAT separation."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    solvers = [s for s in SOLVER_ORDER if any(row["solver"] == s for row in data)]

    # SAT problems heatmap
    sat_data = [row for row in data if row["problem_category"] == "SAT"]
    problems_sat = sorted(list(set(row["problem"] for row in sat_data)))[:40]

    matrix_sat = np.zeros((len(solvers), len(problems_sat)))
    for i, solver in enumerate(solvers):
        for j, problem in enumerate(problems_sat):
            matching = [
                row
                for row in sat_data
                if row["solver"] == solver and row["problem"] == problem
            ]
            if matching:
                matrix_sat[i, j] = matching[0]["wall_time"]
            else:
                matrix_sat[i, j] = np.nan

    matrix_log_sat = np.log10(matrix_sat + 1e-6)
    im1 = ax1.imshow(
        matrix_log_sat, aspect="auto", cmap="Greens", interpolation="nearest"
    )

    ax1.set_xticks(np.arange(0, len(problems_sat), 5))
    ax1.set_xticklabels(
        [problems_sat[i] for i in range(0, len(problems_sat), 5)],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax1.set_yticks(np.arange(len(solvers)))
    ax1.set_yticklabels(solvers)
    ax1.set_xlabel("Problem Instance", fontweight="bold")
    ax1.set_ylabel("Solver", fontweight="bold")
    ax1.set_title(
        "SAT Problems (uf*) - log10 scale",
        fontweight="bold",
        color=COLORS["sat"],
        pad=10,
    )

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("log10(Wall Time)", fontweight="bold")

    # UNSAT problems heatmap
    unsat_data = [row for row in data if row["problem_category"] == "UNSAT"]
    problems_unsat = sorted(list(set(row["problem"] for row in unsat_data)))[:40]

    matrix_unsat = np.zeros((len(solvers), len(problems_unsat)))
    for i, solver in enumerate(solvers):
        for j, problem in enumerate(problems_unsat):
            matching = [
                row
                for row in unsat_data
                if row["solver"] == solver and row["problem"] == problem
            ]
            if matching:
                matrix_unsat[i, j] = matching[0]["wall_time"]
            else:
                matrix_unsat[i, j] = np.nan

    matrix_log_unsat = np.log10(matrix_unsat + 1e-6)
    im2 = ax2.imshow(
        matrix_log_unsat, aspect="auto", cmap="Reds", interpolation="nearest"
    )

    ax2.set_xticks(np.arange(0, len(problems_unsat), 5))
    ax2.set_xticklabels(
        [problems_unsat[i] for i in range(0, len(problems_unsat), 5)],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax2.set_yticks(np.arange(len(solvers)))
    ax2.set_yticklabels(solvers)
    ax2.set_xlabel("Problem Instance", fontweight="bold")
    ax2.set_ylabel("Solver", fontweight="bold")
    ax2.set_title(
        "UNSAT Problems (uuf*) - log10 scale",
        fontweight="bold",
        color=COLORS["unsat"],
        pad=10,
    )

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("log10(Wall Time)", fontweight="bold")

    fig.suptitle("Performance Heatmap by Problem Type", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "10_performance_heatmap.png"), dpi=300)
    plt.close()
    print("✓ Generated: 10_performance_heatmap.png")


def chart_11_nqueens_scaling(data: List[Dict], output_dir: str):
    """11. N-Queens Scaling - Performance vs problem size."""
    nqueens_data = [row for row in data if row.get("type") == "nqueens"]

    if not nqueens_data:
        print("⚠ Skipped: 11_nqueens_scaling.png (no n-queens data)")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Extract N from problem names (assuming format like "04", "08", "12", etc.)
    solver_times = defaultdict(lambda: defaultdict(list))

    for row in nqueens_data:
        try:
            # Problem names are like "04", "08", "12" representing N
            n = int(row["problem"])
            solver_times[row["solver"]][n].append(row["wall_time"])
        except (ValueError, KeyError):
            continue

    # Plot for each solver
    for solver in SOLVER_ORDER:
        if solver not in solver_times:
            continue

        n_values = sorted(solver_times[solver].keys())
        avg_times = [np.mean(solver_times[solver][n]) for n in n_values]

        ax.plot(
            n_values,
            avg_times,
            marker="o",
            markersize=8,
            linewidth=2.5,
            label=solver,
            color=COLORS.get(solver, "#94a3b8"),
            alpha=0.8,
        )

    ax.set_xlabel("N (Board Size)", fontweight="bold")
    ax.set_ylabel("Average Wall Time (seconds)", fontweight="bold")
    ax.set_title("N-Queens Scaling: Performance vs Board Size", pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")  # Log scale for better visualization of exponential growth

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "11_nqueens_scaling.png"), dpi=300)
    plt.close()
    print("✓ Generated: 11_nqueens_scaling.png")


def chart_12_reducible_problems_comparison(data: List[Dict], output_dir: str):
    """12. Reducible Problems - Compare performance across problem types."""
    # Filter for reducible problems
    reducible_types = ["sudoku", "nqueens", "clique"]
    reducible_data = [row for row in data if row.get("type") in reducible_types]

    if not reducible_data:
        print(
            "⚠ Skipped: 12_reducible_problems_comparison.png (no reducible problem data)"
        )
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Group by solver and problem type
    solver_type_times = defaultdict(lambda: defaultdict(list))

    for row in reducible_data:
        solver_type_times[row["solver"]][row["type"]].append(row["wall_time"])

    # Create grouped bar chart
    problem_types = sorted(set(row["type"] for row in reducible_data))
    solvers = [s for s in SOLVER_ORDER if s in solver_type_times]

    x = np.arange(len(problem_types))
    width = 0.25

    for i, solver in enumerate(solvers):
        avg_times = [
            np.mean(solver_type_times[solver][ptype])
            if ptype in solver_type_times[solver]
            else 0
            for ptype in problem_types
        ]

        offset = (i - len(solvers) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            avg_times,
            width,
            label=solver,
            color=COLORS.get(solver, "#94a3b8"),
            alpha=0.8,
        )

    ax.set_xlabel("Problem Type", fontweight="bold")
    ax.set_ylabel("Average Wall Time (seconds)", fontweight="bold")
    ax.set_title("Reducible Problems: Solver Performance Comparison", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([ptype.capitalize() for ptype in problem_types])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "12_reducible_problems_comparison.png"), dpi=300
    )
    plt.close()
    print("✓ Generated: 12_reducible_problems_comparison.png")


def chart_13_sudoku_performance(data: List[Dict], output_dir: str):
    """13. Sudoku Performance - Detailed analysis of sudoku problems."""
    sudoku_data = [row for row in data if row.get("type") == "sudoku"]

    if not sudoku_data:
        print("⚠ Skipped: 13_sudoku_performance.png (no sudoku data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Time distribution by solver
    solver_times = defaultdict(list)
    for row in sudoku_data:
        solver_times[row["solver"]].append(row["wall_time"])

    solvers = [s for s in SOLVER_ORDER if s in solver_times]
    times = [solver_times[s] for s in solvers]

    bp = ax1.boxplot(times, tick_labels=solvers, patch_artist=True)

    for patch, solver in zip(bp["boxes"], solvers):
        patch.set_facecolor(COLORS.get(solver, "#94a3b8"))
        patch.set_alpha(0.7)

    ax1.set_ylabel("Wall Time (seconds)", fontweight="bold")
    ax1.set_xlabel("Solver", fontweight="bold")
    ax1.set_title("Sudoku: Time Distribution", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # Right plot: Success rate by problem
    problem_success = defaultdict(lambda: {"total": 0, "verified": 0})

    for row in sudoku_data:
        problem_success[row["problem"]]["total"] += 1
        if row["verified"]:
            problem_success[row["problem"]]["verified"] += 1

    problems = sorted(problem_success.keys())
    success_rates = [
        100 * problem_success[p]["verified"] / problem_success[p]["total"]
        for p in problems
    ]

    colors = [
        COLORS["verified"] if rate == 100 else COLORS["unverified"]
        for rate in success_rates
    ]

    ax2.bar(range(len(problems)), success_rates, color=colors, alpha=0.8)
    ax2.set_xlabel("Problem Instance", fontweight="bold")
    ax2.set_ylabel("Success Rate (%)", fontweight="bold")
    ax2.set_title("Sudoku: Verification Success by Problem", fontweight="bold")
    ax2.set_xticks(range(len(problems)))
    ax2.set_xticklabels(problems, rotation=45, ha="right", fontsize=8)
    ax2.axhline(y=100, color="green", linestyle="--", alpha=0.3)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim([0, 105])

    fig.suptitle("Sudoku Problem Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "13_sudoku_performance.png"), dpi=300)
    plt.close()
    print("✓ Generated: 13_sudoku_performance.png")


def chart_14_problem_type_overview(data: List[Dict], output_dir: str):
    """14. Problem Type Overview - Distribution and performance across all types."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Problem count by type
    type_counts = defaultdict(int)
    for row in data:
        type_counts[row.get("type", "unknown")] += 1

    types = sorted(type_counts.keys())
    counts = [type_counts[t] for t in types]

    colors_list = [
        COLORS["sat"]
        if t == "cnf"
        else "#9333ea"
        if t == "nqueens"
        else "#f97316"
        if t == "sudoku"
        else "#06b6d4"
        if t == "clique"
        else "#94a3b8"
        for t in types
    ]

    ax1.bar(range(len(types)), counts, color=colors_list, alpha=0.8)
    ax1.set_xlabel("Problem Type", fontweight="bold")
    ax1.set_ylabel("Number of Test Cases", fontweight="bold")
    ax1.set_title("Problem Distribution by Type", fontweight="bold")
    ax1.set_xticks(range(len(types)))
    ax1.set_xticklabels([t.upper() for t in types], rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add count labels on bars
    for i, count in enumerate(counts):
        ax1.text(i, count, str(count), ha="center", va="bottom", fontweight="bold")

    # Right plot: Average time by type
    type_times = defaultdict(list)
    for row in data:
        type_times[row.get("type", "unknown")].append(row["wall_time"])

    avg_times = [np.mean(type_times[t]) for t in types]

    ax2.bar(range(len(types)), avg_times, color=colors_list, alpha=0.8)
    ax2.set_xlabel("Problem Type", fontweight="bold")
    ax2.set_ylabel("Average Wall Time (seconds)", fontweight="bold")
    ax2.set_title("Average Solve Time by Problem Type", fontweight="bold")
    ax2.set_xticks(range(len(types)))
    ax2.set_xticklabels([t.upper() for t in types], rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add time labels on bars
    for i, time in enumerate(avg_times):
        ax2.text(i, time, f"{time:.3f}s", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Problem Type Overview", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "14_problem_type_overview.png"), dpi=300)
    plt.close()
    print("✓ Generated: 14_problem_type_overview.png")


def chart_15_cdcl_statistics(data: List[Dict], output_dir: str):
    """15. CDCL Statistics - Analyze decisions, propagations, conflicts, and learned clauses."""
    cdcl_data = [row for row in data if row["solver"] == "cdcl"]

    if not cdcl_data:
        print("⚠ Skipped: 15_cdcl_statistics.png (no CDCL data)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract stats
    decisions = [row["stats"].get("decisions", 0) for row in cdcl_data if row["stats"]]
    unit_props = [
        row["stats"].get("unit_propagations", 0) for row in cdcl_data if row["stats"]
    ]
    conflicts = [row["stats"].get("conflicts", 0) for row in cdcl_data if row["stats"]]
    learned = [
        row["stats"].get("learned_clauses", 0) for row in cdcl_data if row["stats"]
    ]
    wall_times = [row["wall_time"] for row in cdcl_data if row["stats"]]

    # Color by problem type
    colors = [
        COLORS["sat"] if is_sat_problem(row["problem"]) else COLORS["unsat"]
        for row in cdcl_data
        if row["stats"]
    ]

    # Decisions vs Time
    axes[0, 0].scatter(decisions, wall_times, alpha=0.6, c=colors, s=40)
    axes[0, 0].set_xlabel("Number of Decisions", fontweight="bold")
    axes[0, 0].set_ylabel("Wall Time (seconds)", fontweight="bold")
    axes[0, 0].set_title("Decisions vs Time")
    axes[0, 0].grid(True, alpha=0.3)

    # Conflicts vs Time
    axes[0, 1].scatter(conflicts, wall_times, alpha=0.6, c=colors, s=40)
    axes[0, 1].set_xlabel("Number of Conflicts", fontweight="bold")
    axes[0, 1].set_ylabel("Wall Time (seconds)", fontweight="bold")
    axes[0, 1].set_title("Conflicts vs Time")
    axes[0, 1].grid(True, alpha=0.3)

    # Learned Clauses vs Time
    axes[1, 0].scatter(learned, wall_times, alpha=0.6, c=colors, s=40)
    axes[1, 0].set_xlabel("Learned Clauses", fontweight="bold")
    axes[1, 0].set_ylabel("Wall Time (seconds)", fontweight="bold")
    axes[1, 0].set_title("Learned Clauses vs Time")
    axes[1, 0].grid(True, alpha=0.3)

    # Statistics distribution
    stats_data = [decisions, conflicts, learned]
    bp = axes[1, 1].boxplot(
        stats_data,
        tick_labels=["Decisions", "Conflicts", "Learned"],
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["cdcl"])
        patch.set_alpha(0.7)
    axes[1, 1].set_ylabel("Count", fontweight="bold")
    axes[1, 1].set_title("CDCL Statistics Distribution")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    # Add legend
    sat_patch = mpatches.Patch(color=COLORS["sat"], label="SAT Problems", alpha=0.6)
    unsat_patch = mpatches.Patch(
        color=COLORS["unsat"], label="UNSAT Problems", alpha=0.6
    )
    axes[0, 0].legend(handles=[sat_patch, unsat_patch], loc="upper left")

    fig.suptitle("CDCL Algorithm Statistics Analysis", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "15_cdcl_statistics.png"), dpi=300)
    plt.close()
    print("✓ Generated: 15_cdcl_statistics.png")


def chart_16_hampath_performance(data: List[Dict], output_dir: str):
    """16. Hamiltonian Path Performance - Analysis of graph hampath problems."""
    hampath_data = [row for row in data if row.get("type") == "hampath"]

    if not hampath_data:
        print("⚠ Skipped: 16_hampath_performance.png (no hampath data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Time distribution by solver
    solver_times = defaultdict(list)
    for row in hampath_data:
        solver_times[row["solver"]].append(row["wall_time"])

    solvers = [s for s in SOLVER_ORDER if s in solver_times]
    times = [solver_times[s] for s in solvers]

    bp = ax1.boxplot(times, tick_labels=solvers, patch_artist=True)

    for patch, solver in zip(bp["boxes"], solvers):
        patch.set_facecolor(COLORS.get(solver, "#94a3b8"))
        patch.set_alpha(0.7)

    ax1.set_ylabel("Wall Time (seconds)", fontweight="bold")
    ax1.set_xlabel("Solver", fontweight="bold")
    ax1.set_title("Hamiltonian Path: Time Distribution", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # Right plot: Performance across problem instances
    problem_times = defaultdict(dict)
    for row in hampath_data:
        problem_times[row["problem"]][row["solver"]] = row["wall_time"]

    problems = sorted(problem_times.keys())[:20]  # First 20 problems

    for solver in solvers:
        times_list = [problem_times[p].get(solver, None) for p in problems]
        x_vals = [i for i, t in enumerate(times_list) if t is not None]
        y_vals = [t for t in times_list if t is not None]

        if y_vals:
            ax2.plot(
                x_vals,
                y_vals,
                marker="o",
                markersize=4,
                alpha=0.7,
                label=solver,
                color=COLORS.get(solver, "#94a3b8"),
                linewidth=1.5,
            )

    ax2.set_xlabel("Problem Instance", fontweight="bold")
    ax2.set_ylabel("Wall Time (seconds)", fontweight="bold")
    ax2.set_title("Hamiltonian Path: Performance Across Graphs", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Hamiltonian Path Problem Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "16_hampath_performance.png"), dpi=300)
    plt.close()
    print("✓ Generated: 16_hampath_performance.png")


def chart_17_clique_performance(data: List[Dict], output_dir: str):
    """17. Clique Performance - Analysis of graph clique detection problems."""
    clique_data = [row for row in data if row.get("type") == "clique"]

    if not clique_data:
        print("⚠ Skipped: 17_clique_performance.png (no clique data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Time distribution by solver
    solver_times = defaultdict(list)
    for row in clique_data:
        solver_times[row["solver"]].append(row["wall_time"])

    solvers = [s for s in SOLVER_ORDER if s in solver_times]
    times = [solver_times[s] for s in solvers]

    bp = ax1.boxplot(times, tick_labels=solvers, patch_artist=True)

    for patch, solver in zip(bp["boxes"], solvers):
        patch.set_facecolor(COLORS.get(solver, "#94a3b8"))
        patch.set_alpha(0.7)

    ax1.set_ylabel("Wall Time (seconds)", fontweight="bold")
    ax1.set_xlabel("Solver", fontweight="bold")
    ax1.set_title("Clique: Time Distribution", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # Right plot: Performance across problem instances
    problem_times = defaultdict(dict)
    for row in clique_data:
        problem_times[row["problem"]][row["solver"]] = row["wall_time"]

    problems = sorted(problem_times.keys())[:20]  # First 20 problems

    for solver in solvers:
        times_list = [problem_times[p].get(solver, None) for p in problems]
        x_vals = [i for i, t in enumerate(times_list) if t is not None]
        y_vals = [t for t in times_list if t is not None]

        if y_vals:
            ax2.plot(
                x_vals,
                y_vals,
                marker="o",
                markersize=4,
                alpha=0.7,
                label=solver,
                color=COLORS.get(solver, "#94a3b8"),
                linewidth=1.5,
            )

    ax2.set_xlabel("Problem Instance", fontweight="bold")
    ax2.set_ylabel("Wall Time (seconds)", fontweight="bold")
    ax2.set_title("Clique: Performance Across Graphs", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Clique Detection Problem Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "17_clique_performance.png"), dpi=300)
    plt.close()
    print("✓ Generated: 17_clique_performance.png")


def chart_18_graph_problems_comparison(data: List[Dict], output_dir: str):
    """18. Graph Problems Comparison - Compare hampath vs clique performance."""
    graph_types = ["hampath", "clique"]
    graph_data = [row for row in data if row.get("type") in graph_types]

    if not graph_data:
        print("⚠ Skipped: 18_graph_problems_comparison.png (no graph problem data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Average time comparison
    solver_type_times = defaultdict(lambda: defaultdict(list))

    for row in graph_data:
        solver_type_times[row["solver"]][row["type"]].append(row["wall_time"])

    problem_types = sorted(set(row["type"] for row in graph_data))
    solvers = [s for s in SOLVER_ORDER if s in solver_type_times]

    x = np.arange(len(problem_types))
    width = 0.25

    for i, solver in enumerate(solvers):
        avg_times = [
            np.mean(solver_type_times[solver][ptype])
            if ptype in solver_type_times[solver]
            else 0
            for ptype in problem_types
        ]

        offset = (i - len(solvers) / 2 + 0.5) * width
        ax1.bar(
            x + offset,
            avg_times,
            width,
            label=solver,
            color=COLORS.get(solver, "#94a3b8"),
            alpha=0.8,
        )

    ax1.set_xlabel("Problem Type", fontweight="bold")
    ax1.set_ylabel("Average Wall Time (seconds)", fontweight="bold")
    ax1.set_title("Average Solve Time Comparison", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([ptype.capitalize() for ptype in problem_types])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Right plot: Success rate comparison
    solver_type_success = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "verified": 0})
    )

    for row in graph_data:
        solver_type_success[row["solver"]][row["type"]]["total"] += 1
        if row["verified"]:
            solver_type_success[row["solver"]][row["type"]]["verified"] += 1

    for i, solver in enumerate(solvers):
        success_rates = [
            100
            * solver_type_success[solver][ptype]["verified"]
            / solver_type_success[solver][ptype]["total"]
            if solver_type_success[solver][ptype]["total"] > 0
            else 0
            for ptype in problem_types
        ]

        offset = (i - len(solvers) / 2 + 0.5) * width
        ax2.bar(
            x + offset,
            success_rates,
            width,
            label=solver,
            color=COLORS.get(solver, "#94a3b8"),
            alpha=0.8,
        )

    ax2.set_xlabel("Problem Type", fontweight="bold")
    ax2.set_ylabel("Success Rate (%)", fontweight="bold")
    ax2.set_title("Verification Success Rate", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([ptype.capitalize() for ptype in problem_types])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim([0, 105])

    fig.suptitle("Graph Problems: Hampath vs Clique", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "18_graph_problems_comparison.png"), dpi=300)
    plt.close()
    print("✓ Generated: 18_graph_problems_comparison.png")


def generate_all_charts(result_dir: str, output_dir: str):
    """Generate all benchmark charts."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark Chart Generator")
    print(f"{'=' * 60}\n")
    print(f"Reading data from: {result_dir}")

    data = load_data(result_dir)
    print(f"Loaded {len(data)} benchmark results")

    # Count by type
    sat_count = sum(1 for row in data if row["problem_category"] == "SAT")
    unsat_count = sum(1 for row in data if row["problem_category"] == "UNSAT")
    type_counts = defaultdict(int)
    for row in data:
        type_counts[row.get("type", "unknown")] += 1

    print(f"  - SAT problems (uf*): {sat_count}")
    print(f"  - UNSAT problems (uuf*): {unsat_count}")
    print(f"  - Problem types: {dict(type_counts)}\n")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    setup_plot_style()

    print("Generating charts...\n")

    # Core charts (1-10)
    chart_1_wall_time_overview(data, output_dir)
    chart_2_memory_overview(data, output_dir)
    chart_3_accuracy_overview(data, output_dir)
    chart_4_sat_vs_unsat(data, output_dir)
    chart_5_problem_difficulty(data, output_dir)
    chart_6_memory_usage_line(data, output_dir)
    chart_7_dpll_statistics(data, output_dir)
    chart_8_solver_speedup(data, output_dir)
    chart_9_cumulative_time(data, output_dir)
    chart_10_performance_heatmap(data, output_dir)

    # Reducible problem charts (11-14)
    chart_11_nqueens_scaling(data, output_dir)
    chart_12_reducible_problems_comparison(data, output_dir)
    chart_13_sudoku_performance(data, output_dir)
    chart_14_problem_type_overview(data, output_dir)

    # CDCL statistics chart (15)
    chart_15_cdcl_statistics(data, output_dir)

    # Graph problem charts (16-18)
    chart_16_hampath_performance(data, output_dir)
    chart_17_clique_performance(data, output_dir)
    chart_18_graph_problems_comparison(data, output_dir)

    print(f"\n{'=' * 60}")
    print(f"✓ All charts generated successfully!")
    print(f"{'=' * 60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive charts from benchmark results."
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        default="result",
        help="Path to the result directory containing benchs/ (default: result)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="result/charts",
        help="Output directory for charts (default: result/charts)",
    )

    args = parser.parse_args()

    generate_all_charts(args.result_dir, args.output)


if __name__ == "__main__":
    main()
