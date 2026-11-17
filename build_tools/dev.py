"""Set up a checkout and run the fork's existing development checks.

Run ``python build_tools/dev.py --help`` from any directory. Each command uses the
checkout containing this script, including when it lives in a linked worktree.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _environment(root: Path) -> dict[str, str]:
    """Return a process environment selecting this checkout's isolated virtualenv."""
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("SHAPIQ_RUN_HEAVY_TESTS", None)
    env["UV_PROJECT_ENVIRONMENT"] = str(root / ".venv")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    return env


def _executable(name: str) -> str:
    """Find a prerequisite executable, failing with an actionable message."""
    executable = shutil.which(name)
    if executable is None:
        msg = f"Install {name} and make it available on PATH before running this command."
        raise RuntimeError(msg)
    return executable


def _run(command: list[str], *, root: Path = ROOT) -> None:
    """Run a development command in its checkout, propagating failures."""
    subprocess.run(command, cwd=root, env=_environment(root), check=True)  # noqa: S603


def _python(root: Path = ROOT) -> str:
    """Return the interpreter of an initialized checkout."""
    executable = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not executable.exists():
        msg = f"Missing environment in {root}. Run python build_tools/dev.py setup first."
        raise RuntimeError(msg)
    return str(executable)


def _uv(*arguments: str, root: Path = ROOT) -> None:
    """Run uv without depending on a caller's activated environment."""
    _run([_executable("uv"), *arguments], root=root)


def _configure_git() -> None:
    """Configure fork-only defaults in the repository's shared local Git config."""
    git = _executable("git")
    result = subprocess.run(  # noqa: S603
        [git, "remote", "get-url", "origin"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    origin = result.stdout.strip().removesuffix(".git")
    if origin not in {
        "https://github.com/smtyyy/shapiq-probly",
        "git@github.com:smtyyy/shapiq-probly",
        "ssh://git@github.com/smtyyy/shapiq-probly",
    }:
        msg = "Setup is restricted to the smtyyy/shapiq-probly fork; check the origin remote."
        raise RuntimeError(msg)
    for key, value in {
        "remote.pushDefault": "origin",
        "push.default": "simple",
        "pull.ff": "only",
        "fetch.prune": "true",
    }.items():
        _run([git, "config", "--local", key, value])


def _check_openmp() -> None:
    """Check the static OpenMP prerequisite used by the unchanged macOS build."""
    if sys.platform != "darwin":
        return
    candidates = [Path("/opt/homebrew/opt/libomp"), Path("/usr/local/opt/libomp")]
    if prefix := os.environ.get("LIBOMP_PREFIX"):
        candidates.insert(0, Path(prefix))
    if not any(
        (prefix / "include/omp.h").is_file() and (prefix / "lib/libomp.a").is_file()
        for prefix in candidates
    ):
        msg = (
            "The macOS source build needs include/omp.h and lib/libomp.a. "
            "Run brew install libomp, or set LIBOMP_PREFIX to an existing installation."
        )
        raise RuntimeError(msg)


def setup(*, docs: bool = False) -> None:
    """Install locked dependencies and hooks backed by the durable main checkout.

    Args:
        docs: Also install the existing documentation dependency group.
    """
    _configure_git()
    _check_openmp()
    arguments = ["sync", "--locked", "--python", "3.12", "--all-extras"]
    if docs:
        arguments.extend(["--group", "docs"])
    _uv(*arguments)
    result = subprocess.run(  # noqa: S603
        [_executable("git"), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    main = Path(result.stdout.strip()).parent
    # Linked worktrees share hooks. Never pin their launcher to a disposable .venv.
    if main != ROOT:
        _uv(
            "sync",
            "--locked",
            "--python",
            "3.12",
            "--only-group",
            "lint",
            "--inexact",
            root=main,
        )
    _run(
        [
            _python(main),
            "-m",
            "pre_commit",
            "install",
            "--hook-type",
            "pre-commit",
            "--hook-type",
            "pre-push",
        ],
        root=main,
    )


def quality() -> None:
    """Run all existing commit-stage checks, excluding the push gate itself."""
    _run(
        [
            _python(),
            "-m",
            "pre_commit",
            "run",
            "--all-files",
            "--hook-stage",
            "pre-commit",
            "--show-diff-on-failure",
        ],
    )


def run_tests(*, suite: str = "all", workers: int = 2) -> None:
    """Run the existing regular tests with bounded local parallelism.

    Args:
        suite: ``all``, ``core``, ``games``, or ``benchmark``.
        workers: Number of core pytest workers; games and benchmarks run serially.
    """
    if suite in {"all", "core"}:
        _run(
            [
                _python(),
                "-m",
                "pytest",
                "tests/shapiq",
                "--cov=shapiq",
                "--cov-report=term-missing:skip-covered",
                "--cov-report=xml",
                "-n",
                str(workers),
            ],
        )
    for name in ("games", "benchmark"):
        if suite in {"all", name}:
            _run([_python(), "-m", "pytest", f"tests/shapiq_{name}"])


def build_docs() -> None:
    """Build the existing Sphinx documentation, treating warnings as failures."""
    _run(
        [
            _python(),
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            "html",
            "docs/source",
            "docs/build/html",
        ],
    )


def main() -> int:
    """Parse the requested development task and return its process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup_parser = commands.add_parser("setup", help="Install locked dependencies and Git hooks")
    setup_parser.add_argument("--docs", action="store_true", help="Also install the docs group")
    commands.add_parser("quality", help="Run all commit-stage quality checks")
    test_parser = commands.add_parser("test", help="Run existing regular tests (no heavy opt-in)")
    test_parser.add_argument(
        "--suite", choices=("all", "core", "games", "benchmark"), default="all"
    )
    test_parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    commands.add_parser("check", help="Run quality checks and all regular tests before pushing")
    commands.add_parser(
        "docs", help="Build Sphinx HTML with warnings as errors (needs setup --docs)"
    )
    args = parser.parse_args()
    try:
        if args.command == "setup":
            setup(docs=args.docs)
        elif args.command == "quality":
            quality()
        elif args.command == "test":
            run_tests(suite=args.suite, workers=args.workers)
        elif args.command == "check":
            quality()
            run_tests()
        else:
            build_docs()
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    except RuntimeError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
