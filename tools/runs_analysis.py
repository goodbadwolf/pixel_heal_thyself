import argparse
import os
import re
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patheffects as path_effects


def find_runs_dirs(root_folder):
    r"""Find all folders matching run\d+ pattern."""
    run_folders = []
    run_pattern = re.compile(r"run\d+$")

    for dirpath, dirnames, _ in os.walk(root_folder):
        for dirname in dirnames:
            if run_pattern.match(dirname):
                run_folders.append(os.path.join(dirpath, dirname))

    return run_folders


def process_folder(folder_path, overrides_to_names_map):
    """Process a single run folder and extract metric data."""
    overrides_path = os.path.join(folder_path, ".hydra", "overrides.yaml")
    eval_path = os.path.join(folder_path, "evaluation.txt")

    if not os.path.exists(overrides_path) or not os.path.exists(eval_path):
        return None, None, None, None

    with open(eval_path, "r") as f:
        eval_lines = [line.strip() for line in f.readlines() if line.strip()]
        if len(eval_lines) < 20:
            return None, None, None, None

    allowed_overrides = [r"data\.", r"trainer\."]
    with open(overrides_path, "r") as f:
        filtered_lines = []
        for line in f.readlines():
            line = line.strip()
            for pattern in allowed_overrides:
                if re.search(pattern, line):
                    filtered_lines.append(line)
                    break
        overrides_content = "||".join(filtered_lines)

    if overrides_content not in overrides_to_names_map:
        print(f"Skipping unknown configuration: {overrides_content}")
        return None, None, None, None
    overrides_content = overrides_to_names_map[overrides_content]

    epoch_mrses = {}
    epoch_psnrs = {}
    epoch_ssims = {}
    for line in eval_lines:
        epoch_match = re.search(r"Validation:\s*(\d+)", line)
        msre_match = re.search(r"Avg MRSE:\s*([\d.]+)", line)
        psnr_match = re.search(r"Avg PSNR:\s*([\d.]+)", line)
        ssim_match = re.search(r"Avg 1-SSIM:\s*([\d.]+)", line)

        if epoch_match and psnr_match and msre_match and ssim_match:
            epoch = int(epoch_match.group(1))
            msre = float(msre_match.group(1))
            psnr = float(psnr_match.group(1))
            ssim = 1.0 - float(ssim_match.group(1))

            if epoch not in epoch_mrses:
                epoch_mrses[epoch] = []
            epoch_mrses[epoch].append(msre)
            if epoch not in epoch_psnrs:
                epoch_psnrs[epoch] = []
            epoch_psnrs[epoch].append(psnr)
            if epoch not in epoch_ssims:
                epoch_ssims[epoch] = []
            epoch_ssims[epoch].append(ssim)

    return overrides_content, epoch_mrses, epoch_psnrs, epoch_ssims


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


def calculate_stats(metrics, outliers):
    """Calculate statistics excluding outliers."""
    filtered_values = [x for x in metrics if x not in outliers]

    # If all values are outliers, use the original list
    if not filtered_values:
        filtered_values = metrics

    return {
        "min": min(filtered_values),
        "max": max(filtered_values),
        "avg": sum(filtered_values) / len(filtered_values),
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


def save_current_plot(filename):
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {filename}")


def plot_metric(df, metric_type, colors, markers):
    """Plot a specific metric."""
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    # Get data for this metric
    df_subset = df[df["metric_type"] == metric_type]

    # Get all unique epochs and sort them
    all_epochs = sorted(df_subset["epoch"].unique())

    # Plot each configuration
    for _, (override, color, marker) in enumerate(
        zip(df_subset["overrides"].unique(), colors, markers)
    ):
        subset = df_subset[df_subset["overrides"] == override].sort_values("epoch")

        # Plot average line
        _ = ax.plot(
            subset["epoch"],
            subset["avg"],
            marker=marker,
            markersize=8,
            markeredgewidth=1.5,
            markeredgecolor="white",
            linewidth=2.5,
            color=color,
            label=override,
            alpha=0.9,
            path_effects=[
                path_effects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
                path_effects.Normal(),
            ],
        )[0]

        # Add shaded area for min/max range
        ax.fill_between(
            subset["epoch"],
            subset["min"],
            subset["max"],
            color=color,
            alpha=0.2,
        )

    # Set up x-axis with proper ticks
    ax.set_xticks(all_epochs)
    ax.set_xticklabels([str(int(epoch)) for epoch in all_epochs])

    # Choose appropriate title and labels
    metric_name = {"mrses": "MRSE", "psnrs": "PSNR", "ssims": "SSIM"}[metric_type]
    metric_full = {
        "mrses": "Mean Reciprocal Square Error",
        "psnrs": "Peak Signal-to-Noise Ratio",
        "ssims": "Structural Similarity Index",
    }[metric_type]

    # Set title and labels
    ax.set_title(
        f"{metric_full} ({metric_name})\nComparison Across Different Configurations",
        fontsize=16,
        pad=20,
        fontweight="bold",
    )
    ax.set_xlabel("Epoch", fontsize=12, labelpad=10, fontweight="semibold")
    ax.set_ylabel(
        f"{metric_name} Value", fontsize=12, labelpad=10, fontweight="semibold"
    )

    # For MRSE specifically, set log scale on y-axis
    if metric_type == "mrses":
        # Set specific y-limits to show wider range
        ax.set_ylim(bottom=0, top=0.01)  # Adjust these values as needed

    # Style the ticks
    ax.tick_params(axis="both", which="major", labelsize=10, pad=6)
    ax.tick_params(axis="x", which="both", length=4, width=1)

    # Rotate labels if needed
    if len(all_epochs) > 8:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.subplots_adjust(bottom=0.2)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Position legend appropriately
    if len(df_subset["overrides"].unique()) > 6:
        _ = ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=10,
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            edgecolor="#cccccc",
            title="Configurations",
            title_fontsize=12,
            shadow=True,
        )
    else:
        ax.legend(
            loc="best",
            fontsize=10,
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            edgecolor="#cccccc",
            title="Configurations",
            title_fontsize=12,
            shadow=True,
        )

    # Add annotations and styling
    plt.figtext(
        0.02,
        0.02,
        "Shaded areas represent min-max range (excluding outliers)",
        fontsize=9,
        style="italic",
        alpha=0.7,
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(1.0)

    plt.figtext(
        0.99,
        0.01,
        f"Generated on {datetime.now().strftime('%Y-%m-%d')}",
        fontsize=8,
        ha="right",
        alpha=0.5,
    )

    # Layout adjustments
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    return fig


def create_summary_plot(df, colors, markers):
    """Create a summary plot with all metrics side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    metric_types = ["mrses", "psnrs", "ssims"]
    metric_names = {"mrses": "MRSE", "psnrs": "PSNR", "ssims": "SSIM"}

    # Store the legend for placing outside plots
    # lgd = None

    for ax, metric_type in zip(axes, metric_types):
        df_subset = df[df["metric_type"] == metric_type]
        all_epochs = sorted(df_subset["epoch"].unique())

        # Set up x-axis
        ax.set_xticks(all_epochs)
        ax.set_xticklabels([str(int(epoch)) for epoch in all_epochs])

        # Rotate labels if needed
        if len(all_epochs) > 6:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Plot each configuration (limit to 5 for clarity)
        unique_overrides = df_subset["overrides"].unique()
        max_configs = min(5, len(unique_overrides))

        for i, (override, color, marker) in enumerate(
            zip(
                unique_overrides[:max_configs],
                colors[:max_configs],
                markers[:max_configs],
            )
        ):
            subset = df_subset[df_subset["overrides"] == override].sort_values("epoch")

            # Plot line
            ax.plot(
                subset["epoch"],
                subset["avg"],
                marker=marker,
                markersize=6,
                linewidth=2,
                color=color,
                label=override,
            )

        ax.set_title(f"{metric_names[metric_type]}", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)

        if metric_type == "mrses":
            ax.set_ylabel("Average Value", fontsize=12)

        if metric_type == "psnrs":
            # Only add shared legend on middle plot
            handles, labels = ax.get_legend_handles_labels()
            _ = ax.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=max_configs,
            )

    plt.suptitle("Comparison of All Metrics", fontsize=16, y=1.05, fontweight="bold")
    plt.tight_layout()

    return fig


def generate_metrics_summary(
    df, plot_filters, output_file, tail_epochs=5, best_performer=False, verbose=False
):
    with open(output_file, "w") as f:
        f.write("# Metrics Summary Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Config\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"tail_epochs: {tail_epochs}\n")
        f.write(f"best_performer: {best_performer}\n\n")

        for filter_name, configurations in plot_filters.items():
            f.write(f"\n## Filter: {filter_name}\n")
            f.write("=" * 80 + "\n\n")

            # Filter data to only include these configurations
            filtered_df = df[df["overrides"].isin(configurations)]

            # Get the metrics we're working with
            metric_types = ["mrses", "psnrs", "ssims"]
            metric_full_names = {
                "mrses": "MRSE",
                "psnrs": "PSNR",
                "ssims": "SSIM",
            }

            # For each metric type
            for metric_type in metric_types:
                f.write(f"\n### {metric_full_names[metric_type]}\n")
                f.write("-" * 80 + "\n\n")

                metric_df = filtered_df[filtered_df["metric_type"] == metric_type]

                # Get the last `tail_epochs` epochs for each configuration
                all_epochs = sorted(metric_df["epoch"].unique())
                if len(all_epochs) > tail_epochs:
                    last_epochs = all_epochs[-tail_epochs:]
                else:
                    last_epochs = all_epochs

                # Only keep data for the last epochs
                last_epochs_df = metric_df[metric_df["epoch"].isin(last_epochs)]

                # Calculate the average of the last `tail_epochs` epochs for each configuration
                avg_results = {}

                for config in configurations:
                    config_data = last_epochs_df[last_epochs_df["overrides"] == config]
                    if not config_data.empty:
                        avg_value = config_data["avg"].mean()
                        avg_results[config] = avg_value

                # Create a baseline for comparison (the first configuration in the list)
                baseline_config = configurations[0]

                if baseline_config in avg_results:
                    baseline_value = avg_results[baseline_config]

                    if verbose:
                        f.write(
                            f"Average metrics over last {len(last_epochs)} epochs ({', '.join(map(str, last_epochs))})\n\n"
                        )

                    # Table header
                    f.write(
                        f"{'Configuration':<30} | {'Avg Value':<10} | {'Abs Diff':<10} | {'% Diff':<10} | {'% Trend':<5}\n"
                    )
                    f.write("-" * 80 + "\n")

                    # List results for each configuration
                    for config in configurations:
                        if config in avg_results:
                            config_value = avg_results[config]
                            abs_diff = config_value - baseline_value

                            # For PSNR and SSIM, higher is better; for MRSE, lower is better
                            if metric_type == "mrses":
                                pct_diff = (
                                    (baseline_value - config_value)
                                    / baseline_value
                                    * 100
                                )
                                better = (
                                    "↓"
                                    if abs_diff < 0
                                    else ("=" if abs_diff == 0 else "↑")
                                )
                            else:
                                pct_diff = (
                                    (config_value - baseline_value)
                                    / baseline_value
                                    * 100
                                )
                                better = (
                                    "↑"
                                    if abs_diff > 0
                                    else ("=" if abs_diff == 0 else "↓")
                                )

                            # Format the values with appropriate precision
                            if metric_type == "mrses":
                                value_str = f"{config_value:.6f}"
                                abs_diff_str = f"{abs_diff:.6f}"
                            else:
                                value_str = f"{config_value:.3f}"
                                abs_diff_str = f"{abs_diff:.3f}"

                            # Skip baseline comparison with itself (diff would be 0)
                            if config == baseline_config:
                                pct_diff_str = "baseline"
                            else:
                                pct_diff_str = f"{pct_diff:.2f}"

                            f.write(
                                f"{config:<30} | {value_str:<10} | {abs_diff_str:<10} | {pct_diff_str:<10} | {better:<5}\n"
                            )
                        else:
                            f.write(
                                f"{config:<30} | {'No data':<10} | {'N/A':<10} | {'N/A':<10}\n"
                            )

                # f.write(f"Comparing to baseline configuration: {baseline_config}\n")
                # f.write(
                #     "Last {len(last_epochs)} epochs analyzed: {', '.join(map(str, last_epochs))}\n\n"
                # )

                if best_performer:
                    # Add some statistics about which configuration performed best
                    f.write("\nBest performing configurations:\n")

                    best_configs = {}
                    for epoch in last_epochs:
                        epoch_data = last_epochs_df[last_epochs_df["epoch"] == epoch]

                        if not epoch_data.empty:
                            if metric_type == "mrses":
                                # For MRSE, lower is better
                                best_idx = epoch_data["avg"].idxmin()
                            else:
                                # For PSNR and SSIM, higher is better
                                best_idx = epoch_data["avg"].idxmax()

                            best_config = epoch_data.loc[best_idx]["overrides"]
                            best_value = epoch_data.loc[best_idx]["avg"]

                            if best_config not in best_configs:
                                best_configs[best_config] = 0
                            best_configs[best_config] += 1

                            if metric_type == "mrses":
                                f.write(
                                    f"Epoch {epoch}: {best_config} (MRSE: {best_value:.6f})\n"
                                )
                            else:
                                f.write(
                                    f"Epoch {epoch}: {best_config} ({metric_type[:-1].upper()}: {best_value:.3f})\n"
                                )

                    # Summary of which configuration was best most often
                    f.write("\nConfiguration frequency as best performer:\n")
                    for config, count in sorted(
                        best_configs.items(), key=lambda x: x[1], reverse=True
                    ):
                        f.write(f"{config}: {count}/{len(last_epochs)} epochs\n")

                    f.write("\n")  # Extra line for readability

            f.write("\n\n")  # Extra line between filter sections


overrides_to_names_map = {
    "- data.resize=0.5||- trainer.curve_order=raster||- trainer.epochs=20": "baseline",
    "- data.resize=0.5||- trainer.curve_order=hilbert||- trainer.epochs=20": "baseline+hilbert",
    "- data.resize=0.5||- trainer.curve_order=zorder||- trainer.epochs=20": "baseline+zorder",
    "- data.resize=0.5||- trainer.epochs=20||- trainer.curve_order=raster||- trainer.use_multiscale_discriminator=false||- trainer.use_film=true": "baseline+film",
    "- data.resize=0.5||- trainer.epochs=20||- trainer.curve_order=raster||- trainer.use_multiscale_discriminator=true||- trainer.use_film=false": "baseline+multiscale",
    "- data.resize=0.5||- trainer.epochs=20||- trainer.curve_order=raster||- trainer.use_multiscale_discriminator=true||- trainer.use_film=true": "baseline+multiscale+film",
    "- data.resize=0.5||- trainer.epochs=20||- trainer.curve_order=zorder||- trainer.use_multiscale_discriminator=false||- trainer.use_film=true": "baseline+zorder+film",
    "- data.resize=0.5||- trainer.epochs=20||- trainer.curve_order=zorder||- trainer.use_multiscale_discriminator=true||- trainer.use_film=false": "baseline+zorder+multiscale",
    "- data.resize=0.5||- trainer.epochs=20||- trainer.curve_order=zorder||- trainer.use_multiscale_discriminator=true||- trainer.use_film=true": "baseline+zorder+multiscale+film",
    "- data.resize=0.5||- trainer.epochs=20||- trainer.curve_order=hilbert||- trainer.use_multiscale_discriminator=false||- trainer.use_film=true": "baseline+hilbert+film",
    "- data.resize=0.5||- trainer.epochs=20||- trainer.curve_order=hilbert||- trainer.use_multiscale_discriminator=true||- trainer.use_film=false": "baseline+hilbert+multiscale",
    "- data.resize=0.5||- trainer.epochs=20||- trainer.curve_order=hilbert||- trainer.use_multiscale_discriminator=true||- trainer.use_film=true": "baseline+hilbert+multiscale+film",
}

plot_filters = {
    "baseline.curve_order": ["baseline", "baseline+zorder", "baseline+hilbert"],
    "baseline.multiscale": [
        "baseline",
        "baseline+multiscale",
        "baseline+zorder+multiscale",
        "baseline+hilbert+multiscale",
    ],
    "baseline.film": [
        "baseline",
        "baseline+film",
        "baseline+zorder+film",
        "baseline+hilbert+film",
    ],
    "baseline.multiscale.film": [
        "baseline",
        "baseline+multiscale+film",
        "baseline+zorder+multiscale+film",
        "baseline+hilbert+multiscale+film",
    ],
}


def main(root_folder):
    """Main function to process folders and generate plots."""
    analysis_root = os.path.join(root_folder, "analysis")
    os.makedirs(analysis_root, exist_ok=True)

    run_folders = find_runs_dirs(root_folder)
    print(f"Found {len(run_folders)} run folders")

    # Dictionary to store all data
    all_data = defaultdict(
        lambda: {
            "mrses": defaultdict(list),
            "psnrs": defaultdict(list),
            "ssims": defaultdict(list),
        }
    )

    # Process each folder
    for folder in run_folders:
        overrides_content, epoch_mrses, epoch_psnrs, epoch_ssims = process_folder(
            folder, overrides_to_names_map
        )
        if not all([overrides_content, epoch_mrses, epoch_psnrs, epoch_ssims]):
            continue

        for epoch, mrses in epoch_mrses.items():
            all_data[overrides_content]["mrses"][epoch].extend(mrses)
        for epoch, psnrs in epoch_psnrs.items():
            all_data[overrides_content]["psnrs"][epoch].extend(psnrs)
        for epoch, ssims in epoch_ssims.items():
            all_data[overrides_content]["ssims"][epoch].extend(ssims)

    # Filter configurations with enough data points
    filtered_data = {}
    for overrides, data_dict in all_data.items():
        has_enough_datapoints = True
        for epoch, mrses in data_dict["mrses"].items():
            if len(mrses) < 4:
                has_enough_datapoints = False
                break
        if has_enough_datapoints:
            filtered_data[overrides] = data_dict

    print(f"After filtering, {len(filtered_data)} configurations remain")

    # Calculate stats for each metric
    stats_data = []
    for overrides, data_dict in filtered_data.items():
        for metric_type, epochs in data_dict.items():
            for epoch, metrics in epochs.items():
                outliers = find_outliers(metrics)
                stats = calculate_stats(metrics, outliers)
                stats_data.append(
                    {
                        "overrides": overrides,
                        "epoch": epoch,
                        "metric_type": metric_type,
                        "min": stats["min"],
                        "max": stats["max"],
                        "avg": stats["avg"],
                        "outliers": outliers,
                        "values": metrics,
                    }
                )
        print(f"Processed {overrides} with {len(metrics)} metrics")

    csv_filename = os.path.join(analysis_root, "metrics.csv")
    # Create DataFrame and save to CSV
    df = pd.DataFrame(stats_data)
    df.to_csv(csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")

    # Set up plotting style
    set_plot_style()

    # Create high-contrast color palette
    # Using a combination of 'tab10' and custom colors for maximum contrast
    base_colors = sns.color_palette("tab10", 10)
    additional_colors = [
        "#880E4F",  # Deep pink
        "#004D40",  # Dark teal
        "#3E2723",  # Dark brown
        "#FF6F00",  # Amber
        "#1A237E",  # Indigo
        "#1B5E20",  # Dark green
        "#B71C1C",  # Dark red
        "#4A148C",  # Deep purple
    ]
    colors = base_colors + additional_colors

    # Distinctive markers
    markers = ["o", "s", "D", "^", "v", ">", "<", "p", "*", "X", "P", "d"]

    # Make sure we have enough markers and colors
    max_overrides = len(df["overrides"].unique())
    while len(markers) < max_overrides:
        markers = markers * 2
    while len(colors) < max_overrides:
        colors = colors * 2

    for filter_name, filter in plot_filters.items():
        filtered_df = df[df["overrides"].isin(filter)]
        for metric_type in ["mrses", "psnrs", "ssims"]:
            fig = plot_metric(filtered_df, metric_type, colors, markers)
            plot_filename = os.path.join(
                analysis_root, f"{filter_name}.{metric_type}.png"
            )
            save_current_plot(plot_filename)
            plt.close(fig)

        fig = create_summary_plot(filtered_df, colors, markers)
        summary_filename = os.path.join(analysis_root, f"{filter_name}.summary.png")
        save_current_plot(summary_filename)
        plt.close(fig)

    summary_file = os.path.join(analysis_root, "summary.txt")
    generate_metrics_summary(
        df, plot_filters, summary_file, tail_epochs=3, best_performer=False
    )
    print(f"Metrics summary saved to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process run folders and analyze metric data."
    )
    parser.add_argument("root_folder", help="The root folder to search for run folders")

    args = parser.parse_args()
    main(args.root_folder)
