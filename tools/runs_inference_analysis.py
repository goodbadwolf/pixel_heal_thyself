# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "seaborn",
# ]
# ///
import argparse
import glob
import os
import re
import time
from collections import defaultdict
from datetime import datetime

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
    with open(file_path) as f:
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

    return [x for x in metrics if x < lower_bound or x > upper_bound]


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


def set_plot_style() -> None:
    """Set the global plotting style with enhanced aesthetics."""
    # Use a more modern seaborn style
    sns.set_style("whitegrid", {"grid.linestyle": ":"})

    # Set color palette - using a more visually pleasing palette
    sns.set_palette("deep")

    # Font settings
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Roboto", "Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10

    # Figure aesthetics
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#f8f9fa"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 1.0

    # Grid settings
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "#e0e0e0"
    plt.rcParams["grid.linewidth"] = 0.8

    # Legend settings
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.fancybox"] = True
    plt.rcParams["legend.shadow"] = False
    plt.rcParams["legend.framealpha"] = 0.8
    plt.rcParams["legend.edgecolor"] = "#cccccc"


# Enhanced box plot function with prettier aesthetics
def create_box_plots(data_dict, dataset, output_path, discard_outliers=False) -> None:
    return
    """Create box plots comparing models for a specific dataset with enhanced visuals."""
    metric_names = {
        "rmse": "RMSE (lower is better)",
        "psnr": "PSNR (higher is better)",
        "ssim": "SSIM (higher is better)",
    }

    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=180)

    # Custom color palette
    palette = ["#3366CC", "#FF9933"]

    data_ranges = {}

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
                plot_data.append(metrics)
                labels.append(model_name)
                if metric not in data_ranges:
                    data_ranges[metric] = (min(metrics), max(metrics))
                else:
                    data_ranges[metric] = (
                        min(data_ranges[metric][0], *metrics),
                        max(data_ranges[metric][1], *metrics),
                    )
        # Create prettier box plot
        bplot = sns.boxplot(
            data=plot_data,
            ax=axes[i],
            palette=palette,
            width=0.5,
            linewidth=1.2,
            fliersize=4,
            boxprops={"alpha": 0.8},
            medianprops={"color": "black"},
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 8,
            },
        )

        # Add swarm plot for individual data points with better visibility
        sns.swarmplot(
            data=plot_data,
            ax=axes[i],
            color="black",
            alpha=0.5,
            size=4,
            edgecolor="gray",
            linewidth=0.5,
        )

        # Add mean values as text with better styling
        for j, values in enumerate(plot_data):
            if values:
                mean_val = np.mean(values)
                mean_val_str = f"{mean_val:.2f}"
                if metric == "rmse":
                    mean_val_str = f"{mean_val * 10**4:.2f}"
                axes[i].text(
                    j,
                    mean_val,
                    mean_val_str,
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=9,
                    color="#333333",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.9,
                        "edgecolor": "#cccccc",
                        "boxstyle": "round,pad=0.3",
                        "linewidth": 0.5,
                    },
                )

        # Set title and labels with better styling
        ylabel = "RMSE ($\\times 10^{-4}$)" if metric == "rmse" else metric.upper()
        axes[i].set_title(title, fontsize=14, fontweight="bold", pad=15)
        axes[i].set_ylabel(ylabel, fontsize=12, labelpad=10)
        axes[i].set_xticks(range(len(labels)))
        axes[i].set_xticklabels(labels, rotation=0, ha="center", fontweight="semibold")

        # Adjust the data ranges for clean ranges
        if metric == "rmse":
            data_ranges[metric] = (
                max(0, data_ranges[metric][0] - 0.0003),
                data_ranges[metric][1] + 0.0003,
            )
        elif metric == "psnr":
            data_ranges[metric] = (
                max(30, data_ranges[metric][0] - 1),
                data_ranges[metric][1] + 1,
            )
        elif metric == "ssim":
            data_ranges[metric] = (
                max(0.9, data_ranges[metric][0] - 0.01),
                min(1.0, data_ranges[metric][1] + 0.01),
            )

        axes[i].set_ylim(data_ranges[metric][0], data_ranges[metric][1])
        if metric == "rmse":
            # axes[i].set_ylim(0, 0.002)
            axes[i].yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x * 10**4:.2f}"),
            )
        elif metric == "psnr":
            # axes[i].set_ylim(35, 43)
            pass
        elif metric == "ssim":
            # axes[i].set_ylim(0.93, 0.98)
            pass

        # Add a light border to the plot area
        for spine in axes[i].spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(0.8)

        # Add subtle background shading to alternate rows
        if len(plot_data) > 0:
            for j in range(len(plot_data[0])):
                if j % 2 == 0:
                    axes[i].axhspan(j - 0.5, j + 0.5, alpha=0.05, color="gray")

    # Adjust layout
    plt.tight_layout()

    # Add an appealing title
    plt.suptitle(
        f"Dataset: {dataset} - Comparison of Models across Metrics",
        fontsize=16,
        y=1.05,
        fontweight="bold",
        color="#333333",
    )

    # Save figure with high quality and tight border
    plt.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="white")
    time.sleep(1)  # Ensure the file is saved before closing
    print(f"Box plots for dataset {dataset} saved to {output_path}")
    plt.close(fig)


# Enhanced summary comparison function
def create_dataset_comparison_plot_summary(
    data_dict,
    metric,
    output_path,
    discard_outliers=False,
    variant_name="variant",
) -> None:
    """Create plot comparing performance across datasets with improved visuals."""
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

    # Custom colors for more visual appeal
    baseline_color = "#3366CC"  # Blue
    variant_color = "#FF9933"  # Orange

    # Create figure with three subplots (one for each metric)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=180)

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
                    np.mean(baseline_vals) if baseline_vals else np.nan,
                )
                variant_means.append(np.mean(variant_vals) if variant_vals else np.nan)

        # Plot means with enhanced styling
        x = np.arange(len(all_datasets))
        width = 0.3  # Slightly wider bars for better visibility

        # Bar plot: absolute values with enhanced styling
        baseline_bars = ax.bar(
            x - width / 2,
            baseline_means,
            width,
            label="Baseline",
            color=baseline_color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.85,
            zorder=10,
        )
        variant_bars = ax.bar(
            x + width / 2,
            variant_means,
            width,
            label=variant_name,
            color=variant_color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.85,
            zorder=10,
        )

        # Format value labels based on metric
        if metric == "rmse":
            # For RMSE values which are small numbers, use scientific notation
            def formatter(x) -> str:
                return f"{x * 10**4:.2f}×$10^{{-4}}$"
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x * 10**4:.1f}"),
            )
            ax.set_ylabel("RMSE ($\\times 10^{-4}$)", fontsize=12, labelpad=10)
        else:
            # For PSNR and SSIM, use regular decimal format
            def formatter(x) -> str:
                return f"{x:.3f}"
            ax.set_ylabel(metric.upper(), fontsize=12, labelpad=10)

        # Add value labels on bars with enhanced styling
        for bar in baseline_bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(
                    formatter(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#333333",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.8,
                        "edgecolor": "#cccccc",
                        "boxstyle": "round,pad=0.2",
                        "linewidth": 0.5,
                    },
                    zorder=15,
                )

        for bar in variant_bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(
                    formatter(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#333333",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.8,
                        "edgecolor": "#cccccc",
                        "boxstyle": "round,pad=0.2",
                        "linewidth": 0.5,
                    },
                    zorder=15,
                )

        # Add horizontal grid lines for better readability
        ax.yaxis.grid(True, linestyle=":", linewidth=0.8, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)  # Place grid behind all other elements

        # Remove top and right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Enhanced styling for titles and labels
        ax.set_title(
            f"{metric_names[metric]} by Dataset", fontsize=14, fontweight="bold", pad=15,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(all_datasets, rotation=0, ha="center", fontweight="semibold")

        # Set y-axis limits appropriately
        if metric == "rmse":
            y_min = 0
            y_max = (
                max([m for m in baseline_means + variant_means if not np.isnan(m)])
                * 1.15
            )
            ax.set_ylim([y_min, y_max])
        elif metric == "psnr":
            y_min = (
                min([m for m in baseline_means + variant_means if not np.isnan(m)])
                * 0.95
            )
            y_max = (
                max([m for m in baseline_means + variant_means if not np.isnan(m)])
                * 1.05
            )
            ax.set_ylim([y_min, y_max])
        elif metric == "ssim":
            # For SSIM, focus on the relevant range (typically high values)
            min_val = min(
                [m for m in baseline_means + variant_means if not np.isnan(m)],
            )
            y_min = max(0.9 * min_val, 0)
            y_max = 1.0
            ax.set_ylim([y_min, y_max])

        # Enhanced legend with better formatting
        if i == 0:  # Only add legend to first plot
            legend = ax.legend(
                loc="upper right",
                frameon=True,
                fancybox=True,
                framealpha=0.9,
                edgecolor="#cccccc",
                fontsize=10,
            )
            legend.get_frame().set_linewidth(0.8)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)

    # Add a nicer title
    plt.suptitle(
        f"Comparison of All Metrics Across Datasets: {variant_name} vs Baseline",
        fontsize=16,
        y=1.02,
        fontweight="bold",
        color="#333333",
    )

    # Save figure with high quality
    plt.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="white")
    time.sleep(1)  # Ensure the file is saved before closing
    print(f"Summary comparison plot saved to {output_path}")
    plt.close(fig)


def create_dataset_comparison_plot(
    data_dict,
    metric,
    output_path,
    discard_outliers=False,
    variant_name="variant",
) -> None:
    """
    Create a single plot comparing performance across datasets for a specific metric,
    with improvement percentages shown directly on the bars.
    """
    # Get all datasets
    all_datasets = set()
    for model_data in data_dict.values():
        all_datasets.update(model_data["datasets"].keys())
    all_datasets = sorted(all_datasets)

    if not all_datasets:
        print("Warning: No datasets found for comparison")
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

    # Create figure with a single plot for the main comparison
    fig, ax = plt.subplots(figsize=(14, 8), dpi=180)

    # Custom colors for more visual appeal
    baseline_color = "#3366CC"  # Blue
    variant_color = "#FF9933"  # Orange

    # Plot means
    x = np.arange(len(all_datasets))
    width = 0.35  # Wider bars for better visibility

    # Bar plot: absolute values with enhanced styling
    baseline_bars = ax.bar(
        x - width / 2,
        baseline_means,
        width,
        label="Baseline",
        color=baseline_color,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
        zorder=10,
    )
    variant_bars = ax.bar(
        x + width / 2,
        variant_means,
        width,
        label=variant_name,
        color=variant_color,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
        zorder=10,
    )

    # Format value labels based on metric
    if metric == "rmse":
        # Scientific notation for RMSE
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 10**4:.1f}"))
        ax.set_ylabel(
            "RMSE ($\\times 10^{-4}$)", fontsize=12, fontweight="semibold", labelpad=10,
        )
    else:
        # Regular format for PSNR and SSIM
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight="semibold", labelpad=10)

    # Add value labels on baseline bars
    for i, bar in enumerate(baseline_bars):
        height = bar.get_height()
        if not np.isnan(height):
            val = height * 10**4 if metric == "rmse" else height
            ax.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#333333",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "#cccccc",
                    "boxstyle": "round,pad=0.2",
                    "linewidth": 0.5,
                },
                zorder=15,
            )

    # Add value labels on variant bars with improvement percentage
    for i, bar in enumerate(variant_bars):
        height = bar.get_height()
        if (
            not np.isnan(height)
            and i < len(improvements)
            and not np.isnan(improvements[i])
        ):
            val = height * 10**4 if metric == "rmse" else height

            # Determine if improvement is positive or negative for coloring
            imp = improvements[i]
            imp_color = (
                "#2ca02c" if imp > 0 else "#d62728"
            )  # Green if positive, red if negative

            # Create label text with value and improvement percentage
            label_text = f"{val:.2f}\n({imp:+.1f}%)"

            # Add the annotation
            ax.annotate(
                label_text,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, -height * 0.5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=imp_color,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.9,
                    "edgecolor": "#cccccc",
                    "boxstyle": "round,pad=0.3",
                    "linewidth": 0.5,
                },
                zorder=15,
            )

    # Set appropriate y-axis limits based on the metric
    if metric == "rmse":
        ax.set_ylim(
            [
                0,
                max([m for m in baseline_means + variant_means if not np.isnan(m)])
                * 1.15,
            ],
        )
    elif metric == "psnr":
        min_val = (
            min([m for m in baseline_means + variant_means if not np.isnan(m)]) * 0.97
        )
        max_val = (
            max([m for m in baseline_means + variant_means if not np.isnan(m)]) * 1.03
        )
        ax.set_ylim([min_val, max_val])
    elif metric == "ssim":
        min_val = (
            min([m for m in baseline_means + variant_means if not np.isnan(m)]) * 0.995
        )
        ax.set_ylim([min_val, 1.0])

    # Add horizontal grid lines for better readability
    ax.yaxis.grid(True, linestyle=":", linewidth=0.8, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)  # Place grid behind all other elements

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Enhanced title and axis labels
    # ax.set_title(f"{title}", fontsize=15, fontweight="bold", pad=15, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(all_datasets, rotation=0, ha="center", fontweight="semibold")

    # Enhanced legend
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        fontsize=10,
    )
    legend.get_frame().set_linewidth(0.8)

    # Add explanation of improvement with enhanced styling
    explanation = (
        "Positive % = lower RMSE (better)"
        if metric == "rmse"
        else f"Positive % = higher {metric.upper()} (better)"
    )
    ax.annotate(
        explanation,
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=10,
        fontweight="semibold",
        bbox={
            "boxstyle": "round,pad=0.4", "fc": "white", "ec": "#cccccc", "alpha": 0.9, "linewidth": 0.8,
        },
        zorder=15,
    )

    # Adjust layout for better spacing
    plt.tight_layout()

    # Add overall figure title with enhanced styling
    plt.suptitle(
        f"{metric_names[metric]}: {variant_name} vs Baseline Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.98,
        color="#333333",
    )

    # Save figure with high quality
    plt.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="white")
    time.sleep(1)  # Ensure the file is saved before closing
    print(f"Dataset comparison plot for {metric} saved to {output_path}")
    plt.close(fig)


def generate_dataset_summary(data_dict, dataset, output_file, discard_outliers=False) -> None:
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
                f"{'Model':<20} | {'Min':<10} | {'Max':<10} | {'Avg':<10} | {'Median':<10} | {'StdDev':<10} | {'vs Baseline':<12}\n",
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
                        f"{model_name:<20} | {min_val:<10} | {max_val:<10} | {avg_val:<10} | {med_val:<10} | {std_val:<10} | {comp:<12}\n",
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
                                f"{model_name}: {len(outliers)} outliers out of {len(metrics)} values\n",
                            )
                        else:
                            f.write(f"{model_name}: No outliers found\n")

                f.write("\n")


def generate_summary_report(
    data_dict, output_file, discard_outliers=False, variant_name="variant",
) -> None:
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
            f"{'Dataset':<15} | {'Baseline Files':<15} | {f'{variant_name} Files':<15}\n",
        )
        f.write("-" * 80 + "\n")

        for dataset in all_datasets:
            baseline_count = len(
                data_dict["Baseline"]["datasets"].get(dataset, {}).get("rmse", []),
            )
            variant_count = len(
                data_dict[variant_name]["datasets"].get(dataset, {}).get("rmse", []),
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
                f"{'Dataset':<15} | {'Baseline':<10} | {variant_name:<10} | {'Diff':<10} | {'% Change':<10} | {'Better?':<8}\n",
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
                        f"{dataset:<15} | {baseline_fmt:<10} | {variant_fmt:<10} | {diff_fmt:<10} | {pct_change:+00.2f}% {'':<6} | {better:<8}\n",
                    )
                else:
                    # Missing data for one model
                    f.write(
                        f"{dataset:<15} | {'N/A' if not baseline_metrics else '':<10} | {'N/A' if not variant_metrics else '':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<8}\n",
                    )

            f.write("\n")

    # Now add individual dataset summaries
    for dataset in all_datasets:
        generate_dataset_summary(data_dict, dataset, output_file, discard_outliers)

    print(f"Summary report saved to {output_file}")


def main(baseline_dirs, variant_dirs, variant_name, output_dir, discard_outliers) -> None:
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

    def adjust_dataset_names(data_dict) -> None:
        training_datasets = ["fftle0", "fftle1", "taccturb0", "taccturb1"]
        all_datasets = set(data_dict["datasets"].keys())
        for dataset in all_datasets:
            if dataset in training_datasets:
                data_dict["datasets"][dataset + "*"] = data_dict["datasets"].pop(
                    dataset,
                )
            else:
                data_dict["datasets"][dataset + "†"] = data_dict["datasets"].pop(
                    dataset,
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
            output_dir, f"{dataset[:-1]}_boxplots{outliers_suffix}.png",
        )

        create_box_plots(data_dict, dataset, box_plot_path, discard_outliers)

    # Generate cross-dataset comparison plots
    for metric in ["rmse", "psnr", "ssim"]:
        comparison_path = os.path.join(
            output_dir, f"dataset_comparison_{metric}{outliers_suffix}.png",
        )
        create_dataset_comparison_plot(
            data_dict, metric, comparison_path, discard_outliers, variant_name,
        )

    # In the main function, after generating the individual metric plots:
    summary_plot_path = os.path.join(
        output_dir, f"all_metrics_summary{outliers_suffix}.png",
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
                    },
                )

    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    print(f"Raw metrics saved to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare inference results between baseline and variant models, broken down by dataset.",
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
        "--name", default="variant", help="Name of the variant model (default: variant)",
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
