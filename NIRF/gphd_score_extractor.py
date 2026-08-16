"""
NIRF GPHD score extractor from scores images.

This reads each institute's scores image and extracts only the GPHD value from
the score table at the bottom of the image. It avoids full-row OCR because the
older extractor could mis-order or drop values when reading all 17 columns.

Default output: gphd_from_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


YEARS = (2023, 2024, 2025)
GPHD_COLUMN_INDEX = 11

TESSERACT_WINDOWS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if Path(TESSERACT_WINDOWS_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_WINDOWS_PATH


def _group_peaks(peaks: list[tuple[int, int]], max_gap: int = 2) -> list[int]:
    groups: list[list[tuple[int, int]]] = []
    for position, score in peaks:
        if not groups or position - groups[-1][-1][0] > max_gap:
            groups.append([(position, score)])
        else:
            groups[-1].append((position, score))

    return [max(group, key=lambda item: item[1])[0] for group in groups]


def _find_table_grid(gray: np.ndarray) -> tuple[list[int], list[int]]:
    """Return horizontal and vertical table grid lines inferred from pixels."""
    height, width = gray.shape

    row_counts = (gray < 250).sum(axis=1)
    row_peaks = [
        (y, int(row_counts[y]))
        for y in range(int(height * 0.30), int(height * 0.80))
        if row_counts[y] > width * 0.45
    ]
    horizontal_lines = _group_peaks(row_peaks)[-5:]
    if len(horizontal_lines) != 5:
        raise ValueError(f"expected 5 table horizontal lines, found {len(horizontal_lines)}")

    table_top = horizontal_lines[0]
    table_bottom = horizontal_lines[-1]
    table = gray[table_top : table_bottom + 1]
    table_height = table_bottom - table_top + 1

    col_counts = (table < 245).sum(axis=0)
    col_peaks = [
        (x, int(col_counts[x]))
        for x in range(width)
        if col_counts[x] > table_height * 0.65
    ]
    vertical_lines = _group_peaks(col_peaks)
    if len(vertical_lines) != 19:
        raise ValueError(f"expected 19 table vertical lines, found {len(vertical_lines)}")

    return horizontal_lines, vertical_lines


def _normalise_number(token: str) -> float | None:
    value = float(token)
    if 0 <= value <= 20:
        return value

    # Tesseract sometimes drops decimal points in values like 7.58 -> 758.
    if "." not in token and token.isdigit() and len(token) in {3, 4}:
        value = int(token) / 100
        if 0 <= value <= 20:
            return value

    return None


def _ocr_number(cell: Image.Image) -> float:
    attempts: list[Image.Image] = []
    for border in (5, 10, 20):
        base = ImageOps.expand(cell, border=border, fill="white")
        for scale in (3, 4, 6):
            raw = base.resize((base.width * scale, base.height * scale), Image.Resampling.LANCZOS)
            attempts.append(raw)
            attempts.append(ImageEnhance.Contrast(raw).enhance(2.0))
            attempts.append(raw.filter(ImageFilter.SHARPEN))

    valid_values: list[float] = []
    raw_texts: list[str] = []
    config = "--psm 7 -c tessedit_char_whitelist=0123456789."
    for attempt in attempts:
        text = pytesseract.image_to_string(attempt, config=config).strip()
        if text:
            raw_texts.append(text)

        for token in re.findall(r"\d+(?:\.\d+)?", text):
            value = _normalise_number(token)
            if value is None:
                continue
            if "." in token:
                return value
            valid_values.append(value)

    if valid_values:
        return valid_values[0]

    raise ValueError(f"OCR did not return a valid GPHD number: {raw_texts!r}")


def extract_gphd_from_image(image_path: str | Path) -> float:
    """Extract the GPHD score from one scores image."""
    image_path = Path(image_path)
    gray_image = Image.open(image_path).convert("L")
    gray = np.array(gray_image)

    horizontal_lines, vertical_lines = _find_table_grid(gray)

    # Score row is between the parameter-header line and the score/total line.
    y1 = horizontal_lines[2] + 2
    y2 = horizontal_lines[3] - 2

    # Column 0 is SS. There is one left row-label column before SS.
    x1 = vertical_lines[GPHD_COLUMN_INDEX + 1] + 2
    x2 = vertical_lines[GPHD_COLUMN_INDEX + 2] - 2

    cell = gray_image.crop((x1, y1, x2, y2))
    return _ocr_number(cell)


def _load_meta(college_dir: Path) -> tuple[str, str, int | str]:
    meta_path = college_dir / "meta.json"
    institute_id = college_dir.name
    institute_name = college_dir.name
    rank: int | str = ""

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        institute_id = meta.get("iid", institute_id)
        institute_name = meta.get("name", institute_name)
        rank = meta.get("rank", rank)

    return institute_id, institute_name, rank


def _score_images_for_year(root: Path, year: int) -> list[Path]:
    year_dir = root / str(year)
    if not year_dir.exists():
        return []

    images: list[Path] = []
    for college_dir in sorted(path for path in year_dir.iterdir() if path.is_dir()):
        candidates = sorted(college_dir.glob("scores.*"))
        candidates = [path for path in candidates if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if candidates:
            images.append(candidates[0])
    return images


def run_batch(
    root_dir: str | Path = ".",
    output_csv: str | Path = "gphd_from_scores.csv",
    years: tuple[int, ...] = YEARS,
) -> list[dict[str, object]]:
    root = Path(root_dir)
    records: list[dict[str, object]] = []
    failures: list[str] = []

    for year in years:
        image_paths = _score_images_for_year(root, year)
        print(f"\n[{year}] Found {len(image_paths)} score images...")

        for image_path in image_paths:
            college_dir = image_path.parent
            try:
                institute_id, institute_name, rank = _load_meta(college_dir)
                gphd = extract_gphd_from_image(image_path)
                records.append(
                    {
                        "institute_id": institute_id,
                        "institute_name": institute_name,
                        "year": year,
                        "rank": rank,
                        "GPHD": f"{gphd:.2f}",
                        "image_path": str(image_path),
                        "notes": "",
                    }
                )
                print(f"  OK  [{year}] {institute_name[:55]:<55} GPHD={gphd:.2f}")
            except Exception as exc:
                failures.append(f"{year}/{college_dir.name}: {exc}")
                records.append(
                    {
                        "institute_id": college_dir.name,
                        "institute_name": college_dir.name,
                        "year": year,
                        "rank": "",
                        "GPHD": "",
                        "image_path": str(image_path),
                        "notes": str(exc),
                    }
                )
                print(f"  ERR [{year}] {college_dir.name[:55]:<55} {exc}")

    fieldnames = ["institute_id", "institute_name", "year", "rank", "GPHD", "image_path", "notes"]
    with Path(output_csv).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved {len(records)} records -> {output_csv}")
    if failures:
        print(f"\n{len(failures)} failures:")
        for failure in failures[:20]:
            print(f"  {failure}")

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract GPHD scores from NIRF scores images.")
    parser.add_argument("--root", default=".", help="Root folder containing year folders.")
    parser.add_argument("--output", default="gphd_from_scores.csv", help="CSV output path.")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(YEARS),
        help="Years to process, for example: --years 2024 2025",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if not Path(pytesseract.pytesseract.tesseract_cmd).exists() and not shutil.which("tesseract"):
        raise SystemExit("Tesseract was not found. Install it or update pytesseract.pytesseract.tesseract_cmd.")

    args = parse_args()
    run_batch(args.root, args.output, tuple(args.years))
