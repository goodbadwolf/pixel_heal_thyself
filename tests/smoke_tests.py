"""Smoke tests for PHT - quick sanity checks for the codebase."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Callable, TextIO

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


MRSE_THRESHOLD = 0.1
PSNR_THRESHOLD = 15.0
SSIM_1MINUS_THRESHOLD = 0.5

MIN_EPOCHS_FOR_VALIDATION = 2
EXPECTED_EPOCHS = 5
FINAL_LOSS_THRESHOLD = 0.1
SEPARATOR_LENGTH = 100

IS_CI = os.getenv("CI", "false").lower() == "true"
IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
DEFAULT_TIMEOUT = 300 if IS_CI else 60  # 5 minutes for CI, 1 minute for local


@dataclass
class TestConfig:
    name: str
    overrides: list[str] = field(default_factory=list)
    skip: bool = field(default=False)


@dataclass
class TestResult:
    test: TestConfig
    success: bool
    duration: float
    error: str = None
    files_created: dict = field(default_factory=dict)
    metrics_valid: bool = False
    metrics_error: str = None
    training_valid: bool = False
    training_error: str = None


@dataclass
class CommandResult:
    success: bool
    output: str
    duration: float


def log(message: str, color: str = "") -> None:
    if color:
        print(f"{color}{message}{Colors.ENDC}")
    else:
        print(message)


def log_github_action(
    level: str,
    message: str,
    file: str | None = None,
    line: int | None = None,
) -> None:
    """Log a message with GitHub Actions annotations."""
    if IS_GITHUB_ACTIONS:
        params = []
        if file:
            params.append(f"file={file}")
        if line:
            params.append(f"line={line}")
        param_str = "," + ",".join(params) if params else ""
        print(f"::{level}{param_str}::{message}")
    else:
        color_map = {
            "error": Colors.RED,
            "warning": Colors.YELLOW,
            "notice": Colors.BLUE,
        }
        color = color_map.get(level, "")
        log(f"[{level.upper()}] {message}", color)


def github_group_start(name: str) -> None:
    """Start a collapsible group in GitHub Actions."""
    if IS_GITHUB_ACTIONS:
        print(f"::group::{name}")


def github_group_end() -> None:
    """End a collapsible group in GitHub Actions."""
    if IS_GITHUB_ACTIONS:
        print("::endgroup::")


def log_resource_usage() -> None:
    """Log current resource usage (memory, disk)."""
    if IS_CI and HAS_PSUTIL:
        try:
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            log(
                f"Resources - Memory: {mem.percent:.1f}% used ({mem.used / 1e9:.1f}GB/{mem.total / 1e9:.1f}GB), "
                f"Disk: {disk.percent:.1f}% used ({disk.used / 1e9:.1f}GB/{disk.total / 1e9:.1f}GB)",
            )

            if mem.percent > 90:
                log_github_action("warning", f"High memory usage: {mem.percent:.1f}%")
            if disk.percent > 90:
                log_github_action("warning", f"High disk usage: {disk.percent:.1f}%")
        except Exception as e:
            log(f"Could not get resource usage: {e}", Colors.YELLOW)


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0) -> Callable:
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN002
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        if IS_CI:
                            log(
                                f"Attempt {attempt + 1} failed: {e!s}, retrying in {delay}s...",
                                Colors.YELLOW,
                            )
                        time.sleep(delay)
                    else:
                        raise
            raise last_error

        return wrapper

    return decorator


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


class SmokeTestsRunner:
    MRSE_RE = re.compile(r"Avg MRSE:\s*([0-9.]+)")
    PSNR_RE = re.compile(r"Avg PSNR:\s*([0-9.]+)")
    SSIM_RE = re.compile(r"Avg 1-SSIM:\s*([0-9.]+)")

    def __init__(self, output_dir: Path, verbose: bool = False) -> None:
        self.verbose = verbose
        self.output_dir = output_dir
        self.test_results: list[TestResult] = []
        self.start_time = None

    def _run_command(
        self,
        cmd: list[str],
        timeout: int = DEFAULT_TIMEOUT,
    ) -> CommandResult:
        start = time.time()
        success = False
        log_msg = ""
        log_color = None
        if self.verbose:
            log(f"Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            success = True
            output = result.stdout
            log_msg = f"Command completed successfully: {output}"
        except subprocess.TimeoutExpired:
            success = False
            output = f"Command timed out after {timeout}s"
            log_msg = output
            log_color = Colors.YELLOW
        except subprocess.CalledProcessError as e:
            success = False
            output = f"Command failed with exit code {e.returncode}\n"
            if e.stdout:
                output += f"\n=== STDOUT ===\n{e.stdout}\n"
            if e.stderr:
                output += f"\n=== STDERR ===\n{e.stderr}\n"
            if IS_CI:
                output += f"\n=== TRACEBACK ===\n{traceback.format_exc()}\n"
            log_msg = f"Command failed: Exit code {e.returncode}"
            log_color = Colors.RED
        finally:
            duration = time.time() - start
            if self.verbose and log_msg:
                log(log_msg, log_color)
        return CommandResult(success=success, output=output, duration=duration)

    @retry_on_failure(max_attempts=3, delay=0.5)
    def check_output_files(self, run_dir: Path) -> dict[str, bool]:
        expected_files = {
            "train.log": run_dir / "train.log",
            "train_loss.txt": run_dir / "train_loss.txt",
            "evaluation.txt": run_dir / "evaluation.txt",
            "config.yaml": run_dir / ".hydra" / "config.yaml",
        }

        model_checkpoints = list(run_dir.glob("model_epoch*"))
        expected_files["model_checkpoints"] = (
            model_checkpoints[0] if model_checkpoints else None
        )

        return {
            name: (path.exists() if path else False)
            for name, path in expected_files.items()
        }

    @retry_on_failure(max_attempts=2, delay=0.5)
    def validate_file(
        self,
        file_path: Path,
        parse_fn: Callable,
    ) -> tuple[bool, str]:
        if not file_path.exists():
            return False, f"{file_path.name} not found"
        try:
            with open(file_path) as f:
                return parse_fn(f)
        except Exception as e:
            return False, f"Error reading {file_path.name}: {e}"

    def validate_metrics(self, run_dir: Path) -> tuple[bool, str]:
        eval_file = run_dir / "evaluation.txt"

        def parse_metrics(f: TextIO) -> tuple[bool, str]:
            content = f.read()
            required_metrics = ["MRSE", "PSNR", "SSIM"]
            missing = [m for m in required_metrics if m not in content]
            if missing:
                return False, f"Missing metrics: {', '.join(missing)}"
            if "nan" in content.lower() or "inf" in content.lower():
                return False, "Found NaN or Inf values in metrics"

            mrse_match = self.MRSE_RE.search(content)
            psnr_match = self.PSNR_RE.search(content)
            ssim_match = self.SSIM_RE.search(content)
            if not all([mrse_match, psnr_match, ssim_match]):
                return False, "Could not parse all metrics"

            mrse = float(mrse_match.group(1))
            psnr = float(psnr_match.group(1))
            ssim_1minus = float(ssim_match.group(1))

            success, error = False, ""
            if mrse > MRSE_THRESHOLD:
                error = f"MRSE too high: {mrse:.4f} (expected < {MRSE_THRESHOLD})"
            elif psnr < PSNR_THRESHOLD:
                error = f"PSNR too low: {psnr:.2f} (expected > {PSNR_THRESHOLD})"
            elif ssim_1minus > SSIM_1MINUS_THRESHOLD:
                error = f"1-SSIM too high: {ssim_1minus:.4f} (expected < {SSIM_1MINUS_THRESHOLD})"
            else:
                success = True
                error = f"Metrics OK - MRSE: {mrse:.4f}, PSNR: {psnr:.2f}, 1-SSIM: {ssim_1minus:.4f}"

            return success, error

        return self.validate_file(eval_file, parse_metrics)

    def validate_training(self, run_dir: Path) -> tuple[bool, str]:
        loss_file = run_dir / "train_loss.txt"

        def parse_training(f: TextIO) -> tuple[bool, str]:
            lines = f.readlines()
            success, error = False, ""
            if len(lines) < MIN_EPOCHS_FOR_VALIDATION:
                error = "Not enough epochs in train_loss.txt"
            elif len(lines) < EXPECTED_EPOCHS:
                success = False
                error = f"Training OK but only {len(lines)} epochs completed (expected {EXPECTED_EPOCHS})"

            g_losses = []
            for line in lines:
                g_loss_match = re.search(r"G loss:\s*([0-9.]+)", line)
                if g_loss_match:
                    g_losses.append(float(g_loss_match.group(1)))
            if len(g_losses) < MIN_EPOCHS_FOR_VALIDATION:
                error = "Could not parse G losses"
            elif g_losses[1] >= g_losses[0]:
                error = f"G loss not decreasing: {g_losses[0]:.4f} -> {g_losses[1]:.4f}"
            elif g_losses[-1] > FINAL_LOSS_THRESHOLD:
                error = f"Final G loss too high: {g_losses[-1]:.4f} (expected < {FINAL_LOSS_THRESHOLD})"
            else:
                success = True
                error = f"Training OK - G loss: {g_losses[0]:.4f} -> {g_losses[-1]:.4f}"
            return success, error

        return self.validate_file(loss_file, parse_training)

    def run_test(  # noqa: PLR0912
        self,
        test: TestConfig,
    ) -> TestResult:
        result: TestResult | None = None
        command_result: CommandResult | None = None
        if test.skip:
            log(f"Skipping {test.name}...", Colors.YELLOW)
            result = TestResult(test=test, success=True, duration=0, error=None)
            command_result = CommandResult(success=True, output="", duration=0)
        else:
            log(f"\n{Colors.BOLD}Running {test.name}...{Colors.ENDC}")
            github_group_start(f"Test: {test.name}")
            log_resource_usage()

            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "pht.train",
                "-cn",
                "smoke_tests",
            ]
            cmd.extend(test.overrides)

            command_result = self._run_command(cmd)

            result = TestResult(
                test=test,
                success=command_result.success,
                duration=command_result.duration,
                error=None if command_result.success else command_result.output,
            )

        if result.success:
            if self.output_dir.exists():
                run_dirs = sorted(
                    self.output_dir.glob("**/run*"),
                    key=lambda p: p.stat().st_mtime,
                )
                if run_dirs:
                    latest_run = run_dirs[-1]
                    result.files_created = self.check_output_files(latest_run)
                    result.metrics_valid, result.metrics_error = self.validate_metrics(
                        latest_run,
                    )
                    result.training_valid, result.training_error = (
                        self.validate_training(latest_run)
                    )
                else:
                    result.error = f"No run directories found in {self.output_dir}"
            elif test.skip:
                result.error = f"Test {test.name} was skipped"
                result.metrics_valid = True
                result.training_valid = True
            else:
                result.error = f"{self.output_dir} directory not created"

        if result.success and result.metrics_valid and result.training_valid:
            log(f"✓ {test.name} passed ({command_result.duration:.1f}s)", Colors.GREEN)
        else:
            log(f"✗ {test.name} failed ({command_result.duration:.1f}s)", Colors.RED)
            log_github_action("error", f"Test {test.name} failed")
            if result.success:
                if not result.metrics_valid:
                    log(
                        f"  Metrics: {result.metrics_error}",
                        Colors.RED,
                    )
                    log_github_action(
                        "error",
                        f"Metrics validation failed: {result.metrics_error}",
                    )
                if not result.training_valid:
                    log(
                        f"  Training: {result.training_error}",
                        Colors.RED,
                    )
                    log_github_action(
                        "error",
                        f"Training validation failed: {result.training_error}",
                    )

        github_group_end()
        self.test_results.append(result)
        return result

    def run_all(self, tests_to_skip: list[str] | None = None) -> bool:
        if tests_to_skip is None:
            tests_to_skip = []
        self.test_results = []
        self.start_time = time.time()

        tests = [
            TestConfig(name="afgsa_baseline", overrides=["model=afgsa"]),
            TestConfig(name="mamba_baseline", overrides=["model=mamba"]),
        ]

        for test in tests:
            test.skip = test.name in tests_to_skip

        # TODO: Fix these tests - they complete but don't produce expected outputs
        """[
            # TODO: Fix these tests - they complete but don't produce expected outputs
            # ("AFGSA with LPIPS", "afgsa", ["model.losses.use_lpips_loss=true"]),
            # ("AFGSA with SSIM", "afgsa", ["model.losses.use_ssim_loss=true"]),
            # ("AFGSA Hilbert Curve", "afgsa", ["model.curve_order=hilbert"]),
            # ("Mamba Z-order Curve", "mamba", ["model.curve_order=zorder"]),
            # ("AFGSA with FiLM", "afgsa", ["model.use_film=true"]),
            # ("AFGSA MultiScale Disc", "afgsa", ["model.discriminator.use_multiscale_discriminator=true"]),
        ]"""

        for test in tests:
            if self.output_dir.exists():
                remove_dir(self.output_dir)
                log(
                    f"Cleaned output directory from previous test: {self.output_dir}",
                    Colors.YELLOW,
                )
            self.run_test(test)

        log_resource_usage()
        self.print_summary()

        failed_count = len(
            [
                r
                for r in self.test_results
                if not (r.success and r.metrics_valid and r.training_valid)
            ],
        )

        return failed_count == 0

    def _count_tests_status(self, success: bool) -> int:
        return sum(
            1
            for r in self.test_results
            if (r.success == success) and r.metrics_valid and r.training_valid
        )

    def _print_failed_tests(self) -> None:
        log(f"\n{Colors.BOLD}Failed Tests:{Colors.ENDC}")
        for result in self.test_results:
            if not result.success:
                log(f"  - {result.test.name}:", Colors.RED)
                first_line = (
                    result.error.splitlines()[0] if result.error else "Unknown error"
                )
                log(f"    {first_line}", Colors.RED)
                if IS_CI and result.error:
                    github_group_start(f"Full error: {result.test.name}")
                    log(result.error)
                    github_group_end()

    def _print_output_files_status(self) -> None:
        log(f"\n{Colors.BOLD}Output Files:{Colors.ENDC}")
        for result in self.test_results:
            if result.success and result.files_created:
                missing = [
                    f for f, exists in result.files_created.items() if not exists
                ]
                if missing:
                    log(
                        f"  - {result.test.name}: Missing {', '.join(missing)}",
                        Colors.YELLOW,
                    )
                else:
                    log(f"  - {result.test.name}: All files created ✓", Colors.GREEN)

    def _print_validation_status(
        self,
        validation_type: str,
        validation_key: str,
        error_key: str,
    ) -> None:
        log(f"\n{Colors.BOLD}{validation_type}:{Colors.ENDC}")
        for result in self.test_results:
            if result.success:
                if getattr(result, validation_key):
                    log(
                        f"  - {result.test.name}: {getattr(result, error_key, 'Unknown')}",
                        Colors.GREEN,
                    )
                else:
                    log(
                        f"  - {result.test.name}: {getattr(result, error_key, 'Unknown')}",
                        Colors.RED,
                    )

    def print_summary(self) -> None:
        total_time = time.time() - self.start_time
        passed = self._count_tests_status(True)
        failed = self._count_tests_status(False)

        github_group_start("Test Summary")
        log(f"\n{Colors.BOLD}Summary{Colors.ENDC}")
        log("=" * SEPARATOR_LENGTH)

        log(f"Total tests: {len(self.test_results)}")
        log(f"Passed: {passed} {Colors.GREEN}✓{Colors.ENDC}")
        log(f"Failed: {failed} {Colors.RED}✗{Colors.ENDC}")
        log(f"Total time: {total_time:.1f}s")

        if failed > 0:
            self._print_failed_tests()

        self._print_output_files_status()
        self._print_validation_status(
            "Metrics Validation",
            "metrics_valid",
            "metrics_error",
        )
        self._print_validation_status(
            "Training Progress",
            "training_valid",
            "training_error",
        )
        github_group_end()

    def save_results(self, output_format: str, output_file: str) -> None:
        if output_format == "json":
            with open(output_file, "w") as f:

                def result_to_dict(r: TestResult) -> dict:
                    d = r.__dict__.copy()
                    d["test"] = r.test.__dict__
                    return d

                json.dump([result_to_dict(r) for r in self.test_results], f, indent=2)
            log(f"Results saved to {output_file}")
        else:
            log(f"Unsupported format: {output_format}", Colors.RED)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PHT smoke tests")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Show detailed output",
    )
    parser.add_argument(
        "--clean-pre-run",
        action="store_true",
        default=False,
        help="Clean output directory before running tests",
    )
    parser.add_argument(
        "--clean-post-run",
        action="store_true",
        default=False,
        help="Clean output directory after running tests",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="outputs_smoke_tests",
        help="Outputs directory (default: outputs_smoke_tests)",
    )
    parser.add_argument(
        "-r",
        "--results",
        type=str,
        default="smoke_tests_results.json",
        help="Results file (default: smoke_tests_results.json)",
    )
    parser.add_argument(
        "-f",
        "--save-results-format",
        type=str,
        choices=["json"],
        help="Save results in specified format",
    )
    parser.add_argument(
        "--skip-ci-tests",
        default=["mamba_baseline"],
        nargs="+",
        choices=["afgsa_baseline", "mamba_baseline"],
        help="Skip tests that are hard to install and run on CI. Only applies when running on CI.",
    )
    args = parser.parse_args()

    log(f"{Colors.BOLD}PHT Smoke Tests{Colors.ENDC}")
    log("=" * SEPARATOR_LENGTH)

    if args.clean_pre_run and args.output.exists():
        log(f"Cleaning smoke tests outputs directory... {args.output}")
        remove_dir(args.output)

    tester = SmokeTestsRunner(
        verbose=args.verbose,
        output_dir=args.output,
    )

    if IS_CI:
        tests_to_skip = args.skip_ci_tests
        log(f"Will skip tests: {', '.join(tests_to_skip)}")
    else:
        tests_to_skip = []

    all_passed = tester.run_all(tests_to_skip=tests_to_skip)
    clean_post_run = args.clean_post_run
    if not all_passed:
        clean_post_run = False
        log("Some tests failed. Outputs dir not cleaned.", Colors.YELLOW)

    if clean_post_run:
        log(f"Cleaning smoke tests outputs directory... {args.output}")
        remove_dir(args.output)

    if args.save_results_format:
        tester.save_results(args.save_results_format, args.results)

    log("-" * SEPARATOR_LENGTH)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
