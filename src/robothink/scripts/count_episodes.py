"""
count_episodes.py
=================
Counts the total number of collected episodes across all HDF5 (.hdf5 / .h5)
files found under a given directory (searched recursively).

Supports the robosuite / robomimic HDF5 dataset convention where episodes
are stored as groups directly under the ``data`` top-level group:

    file.hdf5
    └── data/
        ├── demo_0/
        ├── demo_1/
        └── ...

If a file does not contain a ``data`` group, the script falls back to
counting the number of top-level groups in the file, and warns the user.

Usage
-----
    python count_episodes.py <directory> [options]

Arguments
---------
    directory           Path to the directory to search (required).

Options
-------
    -r, --no-recurse    Do NOT search sub-directories (default: recurse).
    -v, --verbose       Print per-file episode counts in addition to totals.
    -h, --help          Show this help message and exit.

Examples
--------
    # Count all episodes under ./datasets, recursively
    python count_episodes.py ./datasets

    # Count only in the top-level directory, with per-file breakdown
    python count_episodes.py ./datasets --no-recurse --verbose
"""

import argparse
import os
import sys

import h5py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_hdf5_files(root: str, recurse: bool) -> list[str]:
    """Return sorted list of .hdf5 / .h5 file paths under *root*."""
    matches = []
    if recurse:
        for dirpath, _dirs, filenames in os.walk(root):
            for fname in filenames:
                if fname.endswith(".hdf5") or fname.endswith(".h5"):
                    matches.append(os.path.join(dirpath, fname))
    else:
        for fname in os.listdir(root):
            if fname.endswith(".hdf5") or fname.endswith(".h5"):
                full = os.path.join(root, fname)
                if os.path.isfile(full):
                    matches.append(full)
    return sorted(matches)


def count_episodes_in_file(path: str) -> tuple[int, str]:
    """Return (episode_count, method_description) for a single HDF5 file.

    The function first looks for the robosuite/robomimic ``data`` group.
    If absent it falls back to counting top-level groups and notes this.
    """
    with h5py.File(path, "r") as f:
        if "data" in f:
            n = len(f["data"].keys())
            return n, "data"
        else:
            # Fallback: count top-level groups
            n = sum(1 for k in f.keys() if isinstance(f[k], h5py.Group))
            return n, "fallback-top-level-groups"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Count episodes in HDF5 demonstration files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "directory",
        help="Root directory to search for HDF5 files.",
    )
    parser.add_argument(
        "-r", "--no-recurse",
        action="store_true",
        default=False,
        help="Do NOT recurse into sub-directories (default: recurse).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Print per-file episode counts.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.directory)
    if not os.path.isdir(root):
        print(f"[ERROR] '{root}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    recurse = not args.no_recurse
    files = find_hdf5_files(root, recurse=recurse)

    if not files:
        print(f"No HDF5 files found under '{root}'.")
        sys.exit(0)

    total = 0
    fallback_files = []

    for path in files:
        try:
            count, method = count_episodes_in_file(path)
        except Exception as exc:
            print(f"[WARNING] Could not read '{path}': {exc}", file=sys.stderr)
            continue

        total += count
        rel = os.path.relpath(path, root)

        if method != "data":
            fallback_files.append(rel)
            flag = "  [no 'data' group — counted top-level groups]"
        else:
            flag = ""

        if args.verbose:
            print(f"  {count:>6} episode(s)  |  {rel}{flag}")

    if args.verbose:
        print()

    print(f"Files found   : {len(files)}")
    print(f"Total episodes: {total}")

    if fallback_files:
        print(
            "\n[WARNING] The following file(s) did not contain a 'data' group.\n"
            "          Episode counts for these files are based on top-level groups\n"
            "          and may not reflect actual episode counts:",
            file=sys.stderr,
        )
        for f in fallback_files:
            print(f"          - {f}", file=sys.stderr)


if __name__ == "__main__":
    main()