"""
Automatic fixer for the GMDH Python script.

What it changes
---------------
1. Adds Pillow imports required for safe TIFF writing.
2. Replaces save_plot_as_tiff() with a robust implementation.
3. Preserves TIFF as the final output format.
4. Keeps the original DPI value, including 1200 dpi.
5. Uses a temporary PNG internally, converts it to RGB, then saves the final
   TIFF with Adobe Deflate compression instead of problematic TIFF-LZW.
6. Reopens and fully decodes the TIFF before accepting it.
7. Creates a backup of the original Python file.

Usage
-----
Command line:
    python fix_gmdh_tiff_grey_lines.py "C:\\path\\to\\your_original_code.py"

Alternatively, run without arguments to enter a source-file path interactively.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path


NEW_FUNCTION = r'''def save_plot_as_tiff(fig, output_path, dpi=FIGURE_DPI):
    """
    Save a verified high-resolution RGB TIFF without random grey/black strips.

    Why this implementation is used:
    - TIFF-LZW can produce corrupted strips in very large 1200-dpi figures.
    - A temporary PNG is rendered first, then converted to RGB.
    - The final TIFF uses Adobe Deflate compression.
    - The final TIFF is reopened and fully decoded before it replaces the output.

    Only the final .tif file remains in the output folder.
    """
    output_path = Path(output_path).with_suffix(".tif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unique_tag = uuid4().hex[:10]
    temporary_png = output_path.parent / f".__gmdh_render_{unique_tag}.png"
    temporary_tiff = output_path.parent / f".__gmdh_verify_{unique_tag}.tif"

    try:
        # Render safely first. The PNG is temporary and is removed below.
        fig.savefig(
            temporary_png,
            dpi=dpi,
            bbox_inches="tight",
            format="png",
            facecolor="white",
            edgecolor="white",
            transparent=False,
        )

        # Convert to RGB so no alpha-channel/viewer artifacts remain.
        with Image.open(temporary_png) as source_image:
            source_image.load()

            if source_image.mode in ("RGBA", "LA"):
                rgba_image = source_image.convert("RGBA")
                rgb_image = Image.new("RGB", rgba_image.size, "white")
                rgb_image.paste(rgba_image, mask=rgba_image.getchannel("A"))
            else:
                rgb_image = source_image.convert("RGB")

            rgb_image.save(
                temporary_tiff,
                format="TIFF",
                compression="tiff_adobe_deflate",
                dpi=(float(dpi), float(dpi)),
            )

        # Fully decode the generated TIFF. Image.verify() alone is insufficient
        # because strip corruption may only be detected during image.load().
        with Image.open(temporary_tiff) as verification_image:
            verification_image.load()

            if verification_image.mode != "RGB":
                raise OSError(
                    f"Unexpected TIFF mode after conversion: {verification_image.mode}"
                )

        # Atomic-style final replacement after successful verification.
        if output_path.exists():
            output_path.unlink()
        temporary_tiff.replace(output_path)

        return output_path

    except Exception as exc:
        raise RuntimeError(
            f"Could not save a valid TIFF file: {output_path}\\n"
            f"Original error: {exc}"
        ) from exc

    finally:
        plt.close(fig)

        for temporary_file in (temporary_png, temporary_tiff):
            if temporary_file.exists():
                try:
                    temporary_file.unlink()
                except OSError:
                    pass
'''


def ensure_imports(source: str) -> str:
    """Add the required imports after the initial import block."""
    additions = []

    if not re.search(r"^from\s+PIL\s+import\s+Image\s*$", source, flags=re.MULTILINE):
        additions.append("from PIL import Image")

    if not re.search(r"^from\s+uuid\s+import\s+uuid4\s*$", source, flags=re.MULTILINE):
        additions.append("from uuid import uuid4")

    if not additions:
        return source

    lines = source.splitlines()
    insertion_index = 0

    # Skip a shebang and encoding declaration when present.
    if lines and lines[0].startswith("#!"):
        insertion_index = 1
    if insertion_index < len(lines) and "coding" in lines[insertion_index]:
        insertion_index += 1

    # Insert after the continuous top-level import block.
    seen_import = False
    for index in range(insertion_index, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            seen_import = True
            insertion_index = index + 1
            continue
        if seen_import and stripped == "":
            insertion_index = index + 1
            continue
        if seen_import:
            break

    new_lines = (
        lines[:insertion_index]
        + additions
        + [""]
        + lines[insertion_index:]
    )
    return "\n".join(new_lines) + ("\n" if source.endswith("\n") else "")


def replace_top_level_function(source: str, function_name: str, replacement: str) -> str:
    """Replace one top-level function while preserving the rest of the script."""
    pattern = re.compile(
        rf"^def\s+{re.escape(function_name)}\s*\(",
        flags=re.MULTILINE,
    )
    match = pattern.search(source)
    if match is None:
        raise ValueError(f"Function {function_name}() was not found in the source file.")

    start = match.start()

    # Locate the next top-level def/class or the next major section comment.
    next_pattern = re.compile(
        r"^(?:def\s+|class\s+|#\s*=+)",
        flags=re.MULTILINE,
    )
    next_match = next_pattern.search(source, match.end())
    end = next_match.start() if next_match else len(source)

    prefix = source[:start].rstrip() + "\n\n"
    suffix = source[end:].lstrip("\n")
    return prefix + replacement.rstrip() + "\n\n" + suffix


def choose_output_path(original_path: Path) -> Path:
    return original_path.with_name(
        original_path.stem + "_FIXED_NO_GREY_TIFF_LINES.py"
    )


def choose_backup_path(original_path: Path) -> Path:
    base = original_path.with_name(
        original_path.stem + "_BACKUP_BEFORE_TIFF_FIX.py"
    )
    if not base.exists():
        return base

    counter = 2
    while True:
        candidate = original_path.with_name(
            original_path.stem + f"_BACKUP_BEFORE_TIFF_FIX_{counter}.py"
        )
        if not candidate.exists():
            return candidate
        counter += 1


def patch_file(original_path: Path) -> tuple[Path, Path]:
    original_path = original_path.expanduser().resolve()

    if not original_path.exists():
        raise FileNotFoundError(f"Python file does not exist: {original_path}")
    if not original_path.is_file():
        raise ValueError(f"The supplied path is not a file: {original_path}")
    if original_path.suffix.lower() != ".py":
        raise ValueError("Select a Python source file (.py).")

    try:
        source = original_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = original_path.read_text(encoding="utf-8-sig")

    if "def save_plot_as_tiff" not in source:
        raise ValueError(
            "The file does not contain def save_plot_as_tiff(...). "
            "Select the GMDH source file that contains the plotting functions."
        )

    patched = ensure_imports(source)
    patched = replace_top_level_function(
        patched,
        "save_plot_as_tiff",
        NEW_FUNCTION,
    )

    # Remove the exact problematic LZW instruction if it remains anywhere else.
    patched = patched.replace(
        'pil_kwargs={"compression": "tiff_lzw"},',
        'pil_kwargs={"compression": "tiff_adobe_deflate"},',
    )
    patched = patched.replace(
        "pil_kwargs={'compression': 'tiff_lzw'},",
        "pil_kwargs={'compression': 'tiff_adobe_deflate'},",
    )

    backup_path = choose_backup_path(original_path)
    output_path = choose_output_path(original_path)

    shutil.copy2(original_path, backup_path)
    output_path.write_text(patched, encoding="utf-8", newline="\n")

    return output_path, backup_path


def get_input_path() -> Path:
    if len(sys.argv) >= 2:
        return Path(sys.argv[1].strip().strip('"'))

    entered = input("Enter path to GMDH Python source (.py): ").strip()
    return Path(entered.strip('"'))


def main() -> None:
    print("=" * 78)
    print("GMDH TIFF GREY-LINE FIXER")
    print("=" * 78)

    try:
        original_path = get_input_path()
        output_path, backup_path = patch_file(original_path)

        print("\nFIX COMPLETED SUCCESSFULLY")
        print(f"Backup file : {backup_path}")
        print(f"Fixed file  : {output_path}")
        print("\nRun the FIXED file, not the old file.")
        print("Delete the previously corrupted TIFF files before rerunning the analysis.")
        print("The new final outputs remain TIFF only; temporary PNG files are deleted.")

    except Exception as exc:
        print("\nFIX FAILED")
        print(str(exc))
        raise SystemExit(1)


if __name__ == "__main__":
    main()