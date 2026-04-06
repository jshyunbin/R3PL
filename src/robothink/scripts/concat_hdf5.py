"""
concat_hdf5.py
==============
Concatenates multiple robosuite/robomimic HDF5 demonstration files found
under a given directory into a single output file.

Before concatenating, the script validates that every file shares identical
``env_args`` settings (environment name, robot, controller, etc.).  If any
file differs, the script aborts and reports which files are inconsistent.

Expected HDF5 layout (robosuite / robomimic convention)
-------------------------------------------------------
    file.hdf5
    └── data/                        ← top-level group
        ├── (attr) env_args          ← JSON string with environment settings
        ├── (attr) total             ← total number of samples (optional)
        ├── demo_0/
        │   ├── (attr) num_samples
        │   ├── actions              ← dataset
        │   ├── rewards              ← dataset
        │   ├── dones                ← dataset
        │   ├── states               ← dataset
        │   └── obs/                 ← group (optional)
        ├── demo_1/
        └── ...

The output file follows the same layout.  Episodes are renamed
sequentially (demo_0, demo_1, …) across all source files.
The ``env_args`` attribute is copied from the first (reference) file.
The ``total`` attribute is recomputed as the sum of all samples.

Usage
-----
    python concat_hdf5.py <directory> <output_file> [options]

Arguments
---------
    directory       Directory to search for HDF5 files (recursively by default).
    output_file     Path for the merged output HDF5 file.

Options
-------
    -r, --no-recurse    Do NOT recurse into sub-directories.
    -v, --verbose       Print per-file and per-episode detail.
    --ignore-mismatch   Skip files whose env_args differ instead of aborting.
    -h, --help          Show this help message and exit.

Examples
--------
    python concat_hdf5.py ./datasets merged.hdf5
    python concat_hdf5.py ./datasets merged.hdf5 --verbose
    python concat_hdf5.py ./datasets merged.hdf5 --no-recurse --ignore-mismatch
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_hdf5_files(root: str, recurse: bool) -> list[str]:
    """Return sorted list of .hdf5 / .h5 paths under *root*."""
    matches = []
    if recurse:
        for dirpath, _dirs, filenames in os.walk(root):
            for fname in sorted(filenames):
                if fname.endswith(".hdf5") or fname.endswith(".h5"):
                    matches.append(os.path.join(dirpath, fname))
    else:
        for fname in sorted(os.listdir(root)):
            full = os.path.join(root, fname)
            if os.path.isfile(full) and (fname.endswith(".hdf5") or fname.endswith(".h5")):
                matches.append(full)
    return matches


# ---------------------------------------------------------------------------
# Settings comparison
# ---------------------------------------------------------------------------

def load_env_args(path: str) -> dict | None:
    """Read and parse the ``data/env_args`` attribute from *path*.

    Returns a parsed dict, or None if the attribute is absent.
    """
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise ValueError(f"'{path}' has no top-level 'data' group.")
        data_grp = f["data"]
        if "env_args" not in data_grp.attrs:
            return None
        raw = data_grp.attrs["env_args"]
        # The attribute may be a JSON string or already a dict-like object.
        if isinstance(raw, (bytes, np.bytes_)):
            raw = raw.decode("utf-8")
        return json.loads(raw)


def env_args_equal(a: dict | None, b: dict | None) -> bool:
    """Return True when two env_args dicts are considered equivalent."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def validate_settings(files: list[str], ignore_mismatch: bool, verbose: bool) -> list[str]:
    """Check that all files share the same env_args.

    Returns the list of files that passed validation.  If *ignore_mismatch*
    is False and any mismatch is found, the function prints an error and
    calls ``sys.exit(1)``.
    """
    reference_path = files[0]
    try:
        reference_args = load_env_args(reference_path)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    if reference_args is None:
        print(
            f"[WARNING] '{reference_path}' has no 'env_args' attribute. "
            "Settings comparison will be skipped.",
            file=sys.stderr,
        )

    valid_files = [reference_path]
    mismatched = []

    for path in files[1:]:
        try:
            args = load_env_args(path)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            if ignore_mismatch:
                continue
            sys.exit(1)

        if not env_args_equal(reference_args, args):
            mismatched.append(path)
            if verbose:
                print(f"  [MISMATCH] {os.path.relpath(path)}")
                print(f"    expected: {json.dumps(reference_args, sort_keys=True)}")
                print(f"    got     : {json.dumps(args, sort_keys=True)}")
        else:
            valid_files.append(path)

    if mismatched:
        msg = (
            f"\n[ERROR] {len(mismatched)} file(s) have different 'env_args' "
            f"settings compared to the reference file:\n"
            f"  Reference: {reference_path}\n"
        )
        for p in mismatched:
            msg += f"  Mismatch : {p}\n"

        if ignore_mismatch:
            msg += "[WARNING] --ignore-mismatch is set; skipping those file(s).\n"
            print(msg, file=sys.stderr)
        else:
            msg += "Aborting.  Use --ignore-mismatch to skip inconsistent files.\n"
            print(msg, file=sys.stderr)
            sys.exit(1)

    return valid_files


# ---------------------------------------------------------------------------
# HDF5 group deep-copy helper
# ---------------------------------------------------------------------------

def copy_group(src_group: h5py.Group, dst_group: h5py.Group) -> None:
    """Recursively copy all datasets and sub-groups from *src_group* to *dst_group*.

    Attributes on sub-groups and datasets are preserved.
    """
    for key in src_group.keys():
        item = src_group[key]
        if isinstance(item, h5py.Dataset):
            dst_group.create_dataset(key, data=item[()], compression=item.compression)
            # Copy dataset-level attributes
            for attr_key, attr_val in item.attrs.items():
                dst_group[key].attrs[attr_key] = attr_val
        elif isinstance(item, h5py.Group):
            new_grp = dst_group.require_group(key)
            # Copy group-level attributes
            for attr_key, attr_val in item.attrs.items():
                new_grp.attrs[attr_key] = attr_val
            copy_group(item, new_grp)


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------

def concatenate(files: list[str], output_path: str, verbose: bool) -> None:
    """Merge all episodes from *files* into a single HDF5 at *output_path*."""

    total_episodes = 0
    total_samples = 0

    with h5py.File(output_path, "w") as out_f:
        out_data = out_f.create_group("data")

        # Copy env_args and other root-level attributes from the first file
        with h5py.File(files[0], "r") as ref_f:
            for attr_key, attr_val in ref_f["data"].attrs.items():
                if attr_key != "total":  # recomputed below
                    out_data.attrs[attr_key] = attr_val

        # Iterate source files
        for file_idx, path in enumerate(files):
            rel = os.path.relpath(path)
            if verbose:
                print(f"\n  Processing [{file_idx + 1}/{len(files)}]: {rel}")

            with h5py.File(path, "r") as src_f:
                src_data = src_f["data"]
                demo_keys = sorted(
                    (k for k in src_data.keys() if k.startswith("demo_")),
                    key=lambda k: int(k.split("_")[1]),
                )

                file_samples = 0
                for demo_key in demo_keys:
                    new_demo_name = f"demo_{total_episodes}"
                    src_demo = src_data[demo_key]

                    # Create the destination demo group
                    dst_demo = out_data.create_group(new_demo_name)

                    # Copy group-level attributes (e.g. num_samples, model_file)
                    for attr_key, attr_val in src_demo.attrs.items():
                        dst_demo.attrs[attr_key] = attr_val

                    # Deep-copy all datasets / sub-groups
                    copy_group(src_demo, dst_demo)

                    n_samples = src_demo.attrs.get("num_samples", 0)
                    file_samples += n_samples
                    total_samples += n_samples
                    total_episodes += 1

                    if verbose:
                        print(
                            f"    {demo_key} → {new_demo_name}  "
                            f"({n_samples} samples)"
                        )

                if verbose:
                    print(
                        f"  Finished {rel}: "
                        f"{len(demo_keys)} episode(s), {file_samples} sample(s)"
                    )

        # Write the recomputed total
        out_data.attrs["total"] = total_samples

    print(
        f"\nDone.\n"
        f"  Source files : {len(files)}\n"
        f"  Total episodes: {total_episodes}\n"
        f"  Total samples : {total_samples}\n"
        f"  Output        : {os.path.abspath(output_path)}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate settings and concatenate multiple robosuite HDF5 files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("directory",    help="Root directory to search for HDF5 files.")
    parser.add_argument("output_file",  help="Path for the merged output HDF5 file.")
    parser.add_argument(
        "-r", "--no-recurse",
        action="store_true", default=False,
        help="Do NOT recurse into sub-directories (default: recurse).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true", default=False,
        help="Print per-file and per-episode detail.",
    )
    parser.add_argument(
        "--ignore-mismatch",
        action="store_true", default=False,
        help="Skip files with mismatched env_args instead of aborting.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.directory)
    if not os.path.isdir(root):
        print(f"[ERROR] '{root}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output_file
    if os.path.exists(output_path):
        print(
            f"[ERROR] Output file '{output_path}' already exists. "
            "Please choose a different path or remove the existing file.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Discover files
    files = find_hdf5_files(root, recurse=not args.no_recurse)
    if not files:
        print(f"No HDF5 files found under '{root}'.")
        sys.exit(0)
    print(f"Found {len(files)} HDF5 file(s) under '{root}'.")

    # 2. Validate settings
    print("\nValidating env_args settings...")
    valid_files = validate_settings(files, args.ignore_mismatch, args.verbose)
    print(f"  {len(valid_files)}/{len(files)} file(s) passed validation.")

    if not valid_files:
        print("[ERROR] No valid files to concatenate.", file=sys.stderr)
        sys.exit(1)

    # 3. Concatenate
    print(f"\nConcatenating into '{output_path}'...")
    concatenate(valid_files, output_path, args.verbose)


if __name__ == "__main__":
    main()