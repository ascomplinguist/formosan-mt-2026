from pathlib import Path
import subprocess


def run_git(args, repo_dir="."):
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.strip()


def ensure_files_clean(files, repo_dir="."):
    """
    Raise RuntimeError if any listed file has uncommitted changes.

    Checks:
    - modified files
    - deleted files
    - staged changes
    - unstaged changes

    Note: untracked files are also reported if they are in `files`.
    """
    repo_dir = Path(repo_dir)

    files = [Path(f).as_posix() for f in files]

    # --porcelain gives stable machine-readable output
    # -- limits the check to the given files
    output = run_git(
        ["status", "--porcelain", "--", *files],
        repo_dir=repo_dir,
    )

    if output:
        raise RuntimeError(
            "Some files have uncommitted changes:\n"
            + output
        )


def get_current_commit_hash(repo_dir="."):
    """Return the current HEAD commit hash."""
    return run_git(["rev-parse", "HEAD"], repo_dir=repo_dir)
