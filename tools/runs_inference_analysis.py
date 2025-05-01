# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "seaborn",
# ]
# ///
import argparse
import os
import re
from collections import defaultdict
from datetime import datetime
import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patheffects as path_effects


def find_evaluation_files(dir_path):
    """Find all files ending with _evaluation.txt in a directory."""
    return glob.glob(os.path.join(dir_path, "**/*_evaluation.txt"), recursive=True)


def process_evaluation_file(file_path):
    """Process a single evaluation file and extract metrics."""
    with open(file_path, "r") as f:
        content = f.read().strip()

    # Extract metrics using regex
    rmse_match = re.search(r"RMSE:\s*([\d.]+)", content)
    psnr_match = re.search(r"PSNR:\s*([\d.]+)", content)
    ssim_match = re.search(r"1-SSIM:\s*([\d.]+)", content)

    # Extract dataset name from filename (e.g., fftle01_0000_32_evaluation.txt)
    basename = os.path.basename(file_path)
    dataset_match = re.match(r"([^_]+)_", basename)
    dataset = dataset_match.group(1) if dataset_match else "unknown"

    if rmse_match and psnr_match and ssim_match:
        rmse = float(rmse_match.group(1))
        psnr = float(psnr_match.group(1))
        ssim = 1.0 - float(ssim_match.group(1))  # Convert 1-SSIM to SSIM
        return {
            "rmse": rmse,
            "psnr": psnr,
            "ssim": ssim,
            "file": basename,
            "dataset": dataset,
        }
    return None


def process_directory(dir_path, model_name):
    """Process all evaluation files in a directory and group by dataset."""
    # Initialize with datasets structure
    datasets = defaultdict(lambda: {"rmse": [], "psnr": [], "ssim": [], "files": []})

    eval_files = find_evaluation_files(dir_path)
    print(f"Found {len(eval_files)} evaluation files in {dir_path}")

    if not eval_files:
        print(f"Warning: No evaluation files found in {dir_path}")
        return None

    for file_path in eval_files:
        result = process_evaluation_file(file_path)
        if result:
            dataset = result["dataset"]
            datasets[dataset]["rmse"].append(result["rmse"])
            datasets[dataset]["psnr"].append(result["psnr"])
            datasets[dataset]["ssim"].append(result["ssim"])
            datasets[dataset]["files"].append(result["file"])

    if not datasets:
        print(f"Warning: No valid metrics found in {dir_path}")
        return None

    return {"model": model_name, "datasets": datasets}


def find_outliers(metrics):
    """Find outliers using IQR method."""
    if len(metrics) <= 1:
        return []

    q1 = np.percentile(metrics, 25)
    q3 = np.percentile(metrics, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = [x for x in metrics if x < lower_bound or x > upper_bound]
    return outliers


def calculate_stats(metrics, outliers=None, discard_outliers=False):
    """Calculate statistics for a list of metrics."""
    if outliers is None:
        outliers = []

    filtered_values = (
        metrics if not discard_outliers else [x for x in metrics if x not in outliers]
    )

    # If all values are outliers, use the original list
    if not filtered_values:
        filtered_values = metrics

    return {
        "min": min(filtered_values),
        "max": max(filtered_values),
        "avg": sum(filtered_values) / len(filtered_values),
        "median": np.median(filtered_values),
        "std": np.std(filtered_values),
        "count": len(filtered_values),
        "outliers": outliers if discard_outliers else [],
    }


def set_plot_style():
    """Set the global plotting style."""
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["axes.facecolor"] = "#f8f9fa"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#dddddd"
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "#e0e0e0"
    plt.rcParams["grid.linestyle"] = "--"


def create_box_plots(data_dict, dataset, output_path, discard_outliers=False):
    """Create box plots comparing models for a specific dataset."""
    metric_names = {
        "rmse": "RMSE (lower is better)",
        "psnr": "PSNR (higher is better)",
        "ssim": "SSIM (higher is better)",
    }

    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    # Create box plots for each metric
    for i, (metric, title) in enumerate(metric_names.items()):
        # Prepare data for box plot
        plot_data = []
        labels = []

        for model_name, model_data in data_dict.items():
            if dataset in model_data["datasets"]:
                metrics = model_data["datasets"][dataset][metric]
                if discard_outliers:
                    outliers = find_outliers(metrics)
                    metrics = [m for m in metrics if m not in outliers]
                if metric == "rmse":
                    metrics = [m * 10**4 for m in metrics]
                plot_data.append(metrics)
                labels.append(model_name)

        # Create box plot
        sns.boxplot(data=plot_data, ax=axes[i], palette="viridis")

        # Add swarm plot for individual data points
        sns.swarmplot(data=plot_data, ax=axes[i], color="black", alpha=0.5, size=3)

        # Add mean values as text
        for j, values in enumerate(plot_data):
            if values:
                mean_val = np.mean(values)
                axes[i].text(
                    j,
                    mean_val,
                    f"{mean_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
                )
        # Set title and labels
        ylabel = "RMSE ($\\times 10^{-4}$)" if metric == "rmse" else metric.upper()
        axes[i].set_title(title, fontsize=14, fontweight="bold")
        axes[i].set_ylabel(ylabel, fontsize=12)
        axes[i].set_xticks(range(len(labels)))
        axes[i].set_xticklabels(labels, rotation=0, ha="center")

    # Adjust layout
    plt.tight_layout()
    plt.suptitle(
        f"Dataset: {dataset} - Comparison of Models across Metrics",
        fontsize=16,
        y=1.05,
    )

    # Save figure
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Box plots for dataset {dataset} saved to {output_path}")
    plt.close(fig)


def create_histogram_plots(data_dict, dataset, output_path, discard_outliers=False):
    """Create histogram plots comparing models for a specific dataset."""
    metric_names = {
        "rmse": "RMSE (lower is better)",
        "psnr": "PSNR (higher is better)",
        "ssim": "SSIM (higher is better)",
    }

    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), dpi=150)

    # Create histograms for each metric
    for i, (metric, title) in enumerate(metric_names.items()):
        # Set individual plot
        ax = axes[i]

        # Create histogram for each model
        for model_name, model_data in data_dict.items():
            if dataset in model_data["datasets"]:
                metrics = model_data["datasets"][dataset][metric]
                if discard_outliers:
                    outliers = find_outliers(metrics)
                    metrics = [m for m in metrics if m not in outliers]

                sns.histplot(
                    metrics,
                    ax=ax,
                    label=model_name,
                    kde=True,
                    element="step",
                    alpha=0.6,
                )

        # Set title and labels
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Value", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend()

        # Add mean lines
        for model_name, model_data in data_dict.items():
            if dataset in model_data["datasets"]:
                metrics = model_data["datasets"][dataset][metric]
                if discard_outliers:
                    outliers = find_outliers(metrics)
                    metrics = [m for m in metrics if m not in outliers]

                mean_val = np.mean(metrics)
                ax.axvline(
                    mean_val,
                    color=ax.lines[-1].get_color(),
                    linestyle="--",
                    linewidth=2,
                    label=f"{model_name} Mean: {mean_val:.3f}",
                )

                # Add text annotation for mean
                y_pos = ax.get_ylim()[1] * 0.9
                ax.text(
                    mean_val,
                    y_pos,
                    f"{mean_val:.3f}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color=ax.lines[-1].get_color(),
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1),
                )

    # Adjust layout
    plt.tight_layout()
    plt.suptitle(
        f"Dataset: {dataset} - Distribution of Metrics across Models",
        fontsize=16,
        y=1.0,
    )

    # Save figure
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Histogram plots for dataset {dataset} saved to {output_path}")
    plt.close(fig)


def create_dataset_comparison_plot_summary(
    data_dict,
    metric,
    output_path,
    discard_outliers=False,
    variant_name="variant",
):
    """Create plot comparing performance across datasets."""
    # If summary_mode is True, we'll create a plot with all metrics
    # Get all datasets
    all_datasets = set()
    for model_data in data_dict.values():
        all_datasets.update(model_data["datasets"].keys())
    all_datasets = sorted(all_datasets)

    if not all_datasets:
        print("Warning: No datasets found for comparison")
        return

    # Metric info for all three metrics
    metrics = ["rmse", "psnr", "ssim"]
    metric_names = {"rmse": "RMSE", "psnr": "PSNR", "ssim": "SSIM"}
    better_is_higher = {"rmse": False, "psnr": True, "ssim": True}

    # Create figure with three subplots (one for each metric)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    # Create plots for each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Prepare data for this metric
        baseline_means = []
        variant_means = []

        for dataset in all_datasets:
            baseline_vals = []
            variant_vals = []

            # Get baseline data
            if "Baseline" in data_dict and dataset in data_dict["Baseline"]["datasets"]:
                metrics_data = data_dict["Baseline"]["datasets"][dataset][metric]
                if discard_outliers:
                    outliers = find_outliers(metrics_data)
                    metrics_data = [m for m in metrics_data if m not in outliers]
                if metrics_data:
                    baseline_vals = metrics_data

            # Get variant data
            if (
                variant_name in data_dict
                and dataset in data_dict[variant_name]["datasets"]
            ):
                metrics_data = data_dict[variant_name]["datasets"][dataset][metric]
                if discard_outliers:
                    outliers = find_outliers(metrics_data)
                    metrics_data = [m for m in metrics_data if m not in outliers]
                if metrics_data:
                    variant_vals = metrics_data

            # Calculate means if both have data
            if baseline_vals and variant_vals:
                baseline_mean = np.mean(baseline_vals)
                variant_mean = np.mean(variant_vals)

                baseline_means.append(baseline_mean)
                variant_means.append(variant_mean)
            else:
                # Missing data for one model
                baseline_means.append(
                    np.mean(baseline_vals) if baseline_vals else np.nan
                )
                variant_means.append(np.mean(variant_vals) if variant_vals else np.nan)

        # Plot means
        x = np.arange(len(all_datasets))
        width = 0.25  # Slightly wider bars

        # Bar plot: absolute values
        baseline_bars = ax.bar(
            x - width / 2, baseline_means, width, label="Baseline", color="#1f77b4"
        )
        variant_bars = ax.bar(
            x + width / 2, variant_means, width, label=variant_name, color="#ff7f0e"
        )

        # Format value labels based on metric
        if metric == "rmse":
            # For RMSE values which are small numbers, use scientific notation
            formatter = lambda x: f"{x * 10**4:.2f}×$10^{{-4}}$"
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x * 10**4:.1f}")
            )
            ax.set_ylabel("RMSE ($\\times 10^{-4}$)", fontsize=12)
        else:
            # For PSNR and SSIM, use regular decimal format
            formatter = lambda x: f"{x:.3f}"
            ax.set_ylabel(metric.upper(), fontsize=12)

        # Add value labels on bars
        for bar in baseline_bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(
                    formatter(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        for bar in variant_bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(
                    formatter(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(
            f"{metric_names[metric]} by Dataset", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(all_datasets, rotation=0, ha="center")
        if i == 0:  # Only add legend to first plot
            ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)  # Reduce space between subplots
    plt.suptitle(
        f"Comparison of All Metrics Across Datasets: {variant_name} vs Baseline",
        fontsize=16,
        y=1.02,
    )

    # Save figure
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    time.sleep(1)  # Ensure the file is saved before closing
    print(f"Summary comparison plot saved to {output_path}")
    plt.close(fig)
    return


def create_dataset_comparison_plot(
    data_dict,
    metric,
    output_path,
    discard_outliers=False,
    variant_name="variant",
):
    # Get all datasets
    all_datasets = set()
    for model_data in data_dict.values():
        all_datasets.update(model_data["datasets"].keys())
    all_datasets = sorted(all_datasets)

    if not all_datasets:
        print(f"Warning: No datasets found for comparison")
        return

    # Prepare data for plotting
    baseline_means = []
    variant_means = []
    improvements = []

    # Metric info
    metric_names = {
        "rmse": "RMSE (lower is better)",
        "psnr": "PSNR (higher is better)",
        "ssim": "SSIM (higher is better)",
    }

    title = metric_names[metric]
    better_is_higher = metric != "rmse"  # For PSNR and SSIM, higher is better

    for dataset in all_datasets:
        baseline_vals = []
        variant_vals = []

        # Get baseline data
        if "Baseline" in data_dict and dataset in data_dict["Baseline"]["datasets"]:
            metrics = data_dict["Baseline"]["datasets"][dataset][metric]
            if discard_outliers:
                outliers = find_outliers(metrics)
                metrics = [m for m in metrics if m not in outliers]
            if metrics:
                baseline_vals = metrics

        # Get variant data
        if variant_name in data_dict and dataset in data_dict[variant_name]["datasets"]:
            metrics = data_dict[variant_name]["datasets"][dataset][metric]
            if discard_outliers:
                outliers = find_outliers(metrics)
                metrics = [m for m in metrics if m not in outliers]
            if metrics:
                variant_vals = metrics

        # Calculate means if both have data
        if baseline_vals and variant_vals:
            baseline_mean = np.mean(baseline_vals)
            variant_mean = np.mean(variant_vals)

            if better_is_higher:
                # For PSNR/SSIM: positive improvement if variant > baseline
                improvement = ((variant_mean - baseline_mean) / baseline_mean) * 100
            else:
                # For RMSE: positive improvement if variant < baseline
                improvement = ((baseline_mean - variant_mean) / baseline_mean) * 100

            baseline_means.append(baseline_mean)
            variant_means.append(variant_mean)
            improvements.append(improvement)
        else:
            # Missing data for one model
            baseline_means.append(np.mean(baseline_vals) if baseline_vals else np.nan)
            variant_means.append(np.mean(variant_vals) if variant_vals else np.nan)
            improvements.append(np.nan)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), dpi=150, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Plot means
    x = np.arange(len(all_datasets))
    width = 0.3

    # First plot: absolute values
    baseline_bars = ax1.bar(
        x - width / 2, baseline_means, width, label="Baseline", color="#1f77b4"
    )
    variant_bars = ax1.bar(
        x + width / 2, variant_means, width, label=variant_name, color="#ff7f0e"
    )

    # Format value labels based on metric
    if metric == "rmse":
        # For RMSE values which are small numbers, use scientific notation
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x * 10**4:.1f}")
        )
        ax1.set_ylabel("RMSE ($\\times 10^{-4}$)", fontsize=12)
    else:
        # For PSNR and SSIM, use regular decimal format
        ax1.set_ylabel(metric.upper(), fontsize=12)

    # Add value labels on top of bars
    def add_labels(bars, ax, fmt):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(
                    fmt.format(height * 10**4),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    fmt = "{:.2f}" if metric == "rmse" else "{:.3f}"
    add_labels(baseline_bars, ax1, fmt)
    add_labels(variant_bars, ax1, fmt)

    # Fixed y-limits for RMSE (showing values as ×10⁻⁴)
    if metric == "rmse":
        ax1.set_ylim([0, 0.0030])

    # Fixed y-limits for PSNR
    elif metric == "psnr":
        ax1.set_ylim([30, 45])

    # Fixed y-limits for SSIM
    elif metric == "ssim":
        ax1.set_ylim([0.94, 0.96])  # Narrow range to highlight small differences

    # ax1.set_ylabel(metric.upper(), fontsize=12)
    ax1.set_title(f"{title} by Dataset", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_datasets, rotation=0, ha="center")
    ax1.legend()

    # Second plot: improvement percentages
    colors = ["#2ca02c" if imp > 0 else "#d62728" for imp in improvements]
    ax2.bar(x, improvements, color=colors)

    # Add value labels on top of bars
    for i, v in enumerate(improvements):
        if not np.isnan(v):
            ax2.text(
                i,
                v + (1 if v > 0 else -1),
                f"{v:.1f}%",
                ha="center",
                va="bottom" if v > 0 else "top",
                fontsize=9,
            )

    ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_ylabel("Improvement (%)", fontsize=12)
    ax2.set_title(
        f"Improvement of {variant_name} over Baseline", fontsize=14, fontweight="bold"
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_datasets, rotation=0, ha="center")

    # Add explanation of improvement
    if metric == "rmse":
        ax2.annotate(
            "Positive % = lower RMSE (better)",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
    else:
        ax2.annotate(
            f"Positive % = higher {metric.upper()} (better)",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    # Save figure
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Dataset comparison plot for {metric} saved to {output_path}")
    plt.close(fig)
    time.sleep(1)  # Ensure the file is saved before closing


def generate_dataset_summary(data_dict, dataset, output_file, discard_outliers=False):
    """Generate a summary report for a specific dataset."""
    UP_ARROW = "\u2191"
    DOWN_ARROW = "\u2193"
    EQUAL_ARROW = "\u2194"

    with open(output_file, "a") as f:  # Append to existing file
        f.write(f"\n\n# Dataset: {dataset}\n")
        f.write("=" * 80 + "\n\n")

        # Check which models have data for this dataset
        models_with_data = []
        for model_name, model_data in data_dict.items():
            if (
                dataset in model_data["datasets"]
                and model_data["datasets"][dataset]["rmse"]
            ):
                models_with_data.append(model_name)

        if not models_with_data:
            f.write(f"No data available for dataset {dataset}\n\n")
            return

        # Metrics analysis
        metric_names = {"rmse": "RMSE", "psnr": "PSNR", "ssim": "SSIM"}

        # Identify baseline model (first model in dict)
        baseline_model = next(iter(data_dict.keys()))

        # Analyze each metric
        for metric, metric_name in metric_names.items():
            f.write(f"\n## {metric_name} Analysis\n")
            f.write("-" * 80 + "\n\n")

            better_indicator = DOWN_ARROW if metric == "rmse" else UP_ARROW

            # Table header
            f.write(
                f"{'Model':<20} | {'Min':<10} | {'Max':<10} | {'Avg':<10} | {'Median':<10} | {'StdDev':<10} | {'vs Baseline':<12}\n"
            )
            f.write("-" * 80 + "\n")

            baseline_stats = None

            # Calculate and display stats for each model
            for model_name in models_with_data:
                model_data = data_dict[model_name]

                if dataset in model_data["datasets"]:
                    metrics = model_data["datasets"][dataset][metric]
                    outliers = find_outliers(metrics) if discard_outliers else []
                    stats = calculate_stats(metrics, outliers, discard_outliers)

                    # Store baseline stats for comparison
                    if model_name == baseline_model:
                        baseline_stats = stats

                    # Format values
                    min_val = (
                        f"{stats['min']:.6f}"
                        if metric == "rmse"
                        else f"{stats['min']:.3f}"
                    )
                    max_val = (
                        f"{stats['max']:.6f}"
                        if metric == "rmse"
                        else f"{stats['max']:.3f}"
                    )
                    avg_val = (
                        f"{stats['avg']:.6f}"
                        if metric == "rmse"
                        else f"{stats['avg']:.3f}"
                    )
                    med_val = (
                        f"{stats['median']:.6f}"
                        if metric == "rmse"
                        else f"{stats['median']:.3f}"
                    )
                    std_val = (
                        f"{stats['std']:.6f}"
                        if metric == "rmse"
                        else f"{stats['std']:.3f}"
                    )

                    # Compare with baseline
                    if model_name == baseline_model:
                        comp = "Baseline"
                    else:
                        diff = stats["avg"] - baseline_stats["avg"]
                        pct_diff = (diff / baseline_stats["avg"]) * 100

                        # Determine if better or worse
                        if metric == "rmse":  # For RMSE, lower is better
                            better = (
                                better_indicator
                                if diff < 0
                                else DOWN_ARROW
                                if diff == 0
                                else UP_ARROW
                            )
                            pct_diff = (
                                -pct_diff
                            )  # Invert for RMSE to make positive = better
                        else:  # For PSNR and SSIM, higher is better
                            better = (
                                better_indicator
                                if diff > 0
                                else EQUAL_ARROW
                                if diff == 0
                                else DOWN_ARROW
                            )

                        comp = f"{pct_diff:+.2f}% {better}"

                    f.write(
                        f"{model_name:<20} | {min_val:<10} | {max_val:<10} | {avg_val:<10} | {med_val:<10} | {std_val:<10} | {comp:<12}\n"
                    )

            f.write("\n")

            # Add outliers info if applicable
            if discard_outliers:
                f.write("Outliers found and removed:\n")
                for model_name in models_with_data:
                    model_data = data_dict[model_name]
                    if dataset in model_data["datasets"]:
                        metrics = model_data["datasets"][dataset][metric]
                        outliers = find_outliers(metrics)
                        if outliers:
                            f.write(
                                f"{model_name}: {len(outliers)} outliers out of {len(metrics)} values\n"
                            )
                        else:
                            f.write(f"{model_name}: No outliers found\n")

                f.write("\n")


def generate_summary_report(
    data_dict, output_file, discard_outliers=False, variant_name="variant"
):
    """Generate a summary report comparing models across all datasets."""
    UP_ARROW = "\u2191"
    DOWN_ARROW = "\u2193"
    EQUAL_ARROW = "\u2194"

    with open(output_file, "w") as f:
        f.write("# Inference Results Analysis Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Configuration\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Discard outliers: {discard_outliers}\n\n")

        # Get all datasets
        all_datasets = set()
        for model_data in data_dict.values():
            all_datasets.update(model_data["datasets"].keys())
        all_datasets = sorted(all_datasets)

        # Summary of datasets and models
        f.write("## Summary of Available Data\n")
        f.write("=" * 80 + "\n\n")

        # Table header for datasets summary
        f.write(
            f"{'Dataset':<15} | {'Baseline Files':<15} | {f'{variant_name} Files':<15}\n"
        )
        f.write("-" * 80 + "\n")

        for dataset in all_datasets:
            baseline_count = len(
                data_dict["Baseline"]["datasets"].get(dataset, {}).get("rmse", [])
            )
            variant_count = len(
                data_dict[variant_name]["datasets"].get(dataset, {}).get("rmse", [])
            )

            f.write(f"{dataset:<15} | {baseline_count:<15} | {variant_count:<15}\n")

        f.write("\n\n## Overall Model Comparison\n")
        f.write("=" * 80 + "\n\n")

        # Table for overall metrics
        metric_names = {"rmse": "RMSE", "psnr": "PSNR", "ssim": "SSIM"}

        for metric, metric_name in metric_names.items():
            f.write(f"\n### {metric_name} - Overall Average by Dataset\n")
            f.write("-" * 80 + "\n\n")

            better_indicator = DOWN_ARROW if metric == "rmse" else UP_ARROW

            # Table header
            f.write(
                f"{'Dataset':<15} | {'Baseline':<10} | {variant_name:<10} | {'Diff':<10} | {'% Change':<10} | {'Better?':<8}\n"
            )
            f.write("-" * 80 + "\n")

            for dataset in all_datasets:
                baseline_metrics = (
                    data_dict["Baseline"]["datasets"].get(dataset, {}).get(metric, [])
                )
                variant_metrics = (
                    data_dict[variant_name]["datasets"].get(dataset, {}).get(metric, [])
                )

                if baseline_metrics and variant_metrics:
                    if discard_outliers:
                        baseline_outliers = find_outliers(baseline_metrics)
                        variant_outliers = find_outliers(variant_metrics)
                        baseline_metrics = [
                            m for m in baseline_metrics if m not in baseline_outliers
                        ]
                        variant_metrics = [
                            m for m in variant_metrics if m not in variant_outliers
                        ]

                    baseline_avg = np.mean(baseline_metrics)
                    variant_avg = np.mean(variant_metrics)
                    diff = variant_avg - baseline_avg

                    if metric == "rmse":  # For RMSE, lower is better
                        pct_change = (
                            -100 * diff / baseline_avg
                        )  # Negative diff is improvement
                        better = (
                            better_indicator
                            if diff < 0
                            else (EQUAL_ARROW if diff == 0 else UP_ARROW)
                        )
                    else:  # For PSNR and SSIM, higher is better
                        pct_change = (
                            100 * diff / baseline_avg
                        )  # Positive diff is improvement
                        better = (
                            better_indicator
                            if diff > 0
                            else (EQUAL_ARROW if diff == 0 else DOWN_ARROW)
                        )

                    # Format values
                    baseline_fmt = (
                        f"{baseline_avg:.6f}"
                        if metric == "rmse"
                        else f"{baseline_avg:.3f}"
                    )
                    variant_fmt = (
                        f"{variant_avg:.6f}"
                        if metric == "rmse"
                        else f"{variant_avg:.3f}"
                    )
                    diff_fmt = f"{diff:.6f}" if metric == "rmse" else f"{diff:.3f}"

                    f.write(
                        f"{dataset:<15} | {baseline_fmt:<10} | {variant_fmt:<10} | {diff_fmt:<10} | {pct_change:+.2f}%:<10 | {better:<8}\n"
                    )
                else:
                    # Missing data for one model
                    f.write(
                        f"{dataset:<15} | {'N/A' if not baseline_metrics else '':<10} | {'N/A' if not variant_metrics else '':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<8}\n"
                    )

            f.write("\n")

    # Now add individual dataset summaries
    for dataset in all_datasets:
        generate_dataset_summary(data_dict, dataset, output_file, discard_outliers)

    print(f"Summary report saved to {output_file}")


def main(baseline_dirs, variant_dirs, variant_name, output_dir, discard_outliers):
    """Main function to process directories and generate analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # Process baseline directories
    baseline_data = {"model": "Baseline", "datasets": {}}

    print(f"Processing baseline directories: {baseline_dirs}")
    for dir_path in baseline_dirs:
        result = process_directory(dir_path, "Baseline")
        if result:
            # Merge dataset results
            for dataset, metrics in result["datasets"].items():
                if dataset not in baseline_data["datasets"]:
                    baseline_data["datasets"][dataset] = {
                        "rmse": [],
                        "psnr": [],
                        "ssim": [],
                        "files": [],
                    }
                baseline_data["datasets"][dataset]["rmse"].extend(metrics["rmse"])
                baseline_data["datasets"][dataset]["psnr"].extend(metrics["psnr"])
                baseline_data["datasets"][dataset]["ssim"].extend(metrics["ssim"])
                baseline_data["datasets"][dataset]["files"].extend(metrics["files"])

    # Process variant directories
    variant_data = {"model": variant_name, "datasets": {}}

    print(f"Processing variant directories: {variant_dirs}")
    for dir_path in variant_dirs:
        result = process_directory(dir_path, variant_name)
        if result:
            # Merge dataset results
            for dataset, metrics in result["datasets"].items():
                if dataset not in variant_data["datasets"]:
                    variant_data["datasets"][dataset] = {
                        "rmse": [],
                        "psnr": [],
                        "ssim": [],
                        "files": [],
                    }
                variant_data["datasets"][dataset]["rmse"].extend(metrics["rmse"])
                variant_data["datasets"][dataset]["psnr"].extend(metrics["psnr"])
                variant_data["datasets"][dataset]["ssim"].extend(metrics["ssim"])
                variant_data["datasets"][dataset]["files"].extend(metrics["files"])

    def adjust_dataset_names(data_dict):
        training_datasets = ["fftle0", "fftle1", "taccturb0", "taccturb1"]
        all_datasets = set(data_dict["datasets"].keys())
        for dataset in all_datasets:
            if dataset in training_datasets:
                data_dict["datasets"][dataset + "*"] = data_dict["datasets"].pop(
                    dataset
                )
            else:
                data_dict["datasets"][dataset + "†"] = data_dict["datasets"].pop(
                    dataset
                )

    adjust_dataset_names(baseline_data)
    adjust_dataset_names(variant_data)

    # Create data dictionary for analysis
    data_dict = {"Baseline": baseline_data, variant_name: variant_data}

    # Check if we have data to analyze
    if not baseline_data["datasets"] or not variant_data["datasets"]:
        print("Error: No valid data found to perform analysis.")
        return

    # Get all datasets
    all_datasets = set()
    for model_data in data_dict.values():
        all_datasets.update(model_data["datasets"].keys())
    all_datasets = sorted(all_datasets)

    print(f"Found {len(all_datasets)} datasets: {', '.join(all_datasets)}")

    # Set up plotting style
    set_plot_style()

    # Outliers suffix for file names
    outliers_suffix = "_no_outliers" if discard_outliers else ""

    # Generate plots for each dataset
    for dataset in all_datasets:
        # Box plots
        box_plot_path = os.path.join(
            output_dir, f"{dataset[:-1]}_boxplots{outliers_suffix}.png"
        )
        create_box_plots(data_dict, dataset, box_plot_path, discard_outliers)

        # Histogram plots
        hist_plot_path = os.path.join(
            output_dir, f"{dataset[:-1]}_histograms{outliers_suffix}.png"
        )
        create_histogram_plots(data_dict, dataset, hist_plot_path, discard_outliers)

    # Generate cross-dataset comparison plots
    for metric in ["rmse", "psnr", "ssim"]:
        comparison_path = os.path.join(
            output_dir, f"dataset_comparison_{metric}{outliers_suffix}.png"
        )
        create_dataset_comparison_plot(
            data_dict, metric, comparison_path, discard_outliers, variant_name
        )

    # In the main function, after generating the individual metric plots:
    summary_plot_path = os.path.join(
        output_dir, f"all_metrics_summary{outliers_suffix}.png"
    )
    create_dataset_comparison_plot_summary(
        data_dict,
        None,
        summary_plot_path,
        discard_outliers,
        variant_name,
    )

    # Generate summary report
    summary_file = os.path.join(output_dir, f"summary{outliers_suffix}.txt")
    generate_summary_report(data_dict, summary_file, discard_outliers, variant_name)

    # Save raw data as CSV
    csv_file = os.path.join(output_dir, f"metrics{outliers_suffix}.csv")

    # Create DataFrame from raw data
    rows = []

    for model_name, model_data in data_dict.items():
        for dataset, metrics in model_data["datasets"].items():
            for i in range(len(metrics["rmse"])):
                rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset,
                        "file": metrics["files"][i],
                        "rmse": metrics["rmse"][i],
                        "psnr": metrics["psnr"][i],
                        "ssim": metrics["ssim"][i],
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    print(f"Raw metrics saved to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare inference results between baseline and variant models, broken down by dataset."
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        required=True,
        help="Directories containing baseline inference results",
    )
    parser.add_argument(
        "--variant",
        nargs="+",
        required=True,
        help="Directories containing variant inference results",
    )
    parser.add_argument(
        "--name", default="variant", help="Name of the variant model (default: variant)"
    )
    parser.add_argument(
        "--output",
        default="inference_analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--discard-outliers",
        action="store_true",
        default=False,
        help="Discard outliers from the analysis",
    )

    args = parser.parse_args()
    main(args.baseline, args.variant, args.name, args.output, args.discard_outliers)
