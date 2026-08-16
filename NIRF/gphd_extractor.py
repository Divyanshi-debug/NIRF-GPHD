"""
NIRF GPHD Raw Feature Extractor
================================
Extracts PhD graduation counts from NIRF PDFs for all years (2023, 2024, 2025).

Fields extracted per institution per year:
  - institute_id
  - institute_name
  - year
  - ft_year1, pt_year1  : Full-time / Part-time PhD graduates, most recent year
  - ft_year2, pt_year2  : Full-time / Part-time PhD graduates, 2nd year
  - ft_year3, pt_year3  : Full-time / Part-time PhD graduates, 3rd year
  - year1_label, year2_label, year3_label : Academic year labels (e.g. "2023-24")
  - total_year1, total_year2, total_year3 : FT + PT totals per year
  - Nphd                : Average total PhD graduates over 3 years
  - GPHD_actual         : Official GPHD score from meta.json (not available at sub-param level, left blank)
  - notes               : Any extraction issues

Output: gphd_raw_features.csv
"""

import pdfplumber
import re
import json
import csv
import os
from pathlib import Path


# ── HELPERS ───────────────────────────────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        return ""


def parse_int(val: str) -> int | None:
    """Clean and parse an integer from a string."""
    val = val.strip().replace(",", "")
    if val.isdigit():
        return int(val)
    return None


# ── GPHD EXTRACTION ───────────────────────────────────────────────────────────

def extract_gphd_features(pdf_path: str, institute_id: str, year: int) -> dict:
    """
    Extract PhD graduation counts from a single PDF.

    The PDF contains a section like:
        No. of Ph.D students graduated (including Integrated Ph.D)
        2023-24    2022-23    2021-22
        Full Time  289        278        231
        Part Time  25         3          0
    """
    text = extract_text(pdf_path)

    record = {
        "institute_id": institute_id,
        "year": year,
        "year1_label": None, "year2_label": None, "year3_label": None,
        "ft_year1": None, "ft_year2": None, "ft_year3": None,
        "pt_year1": None, "pt_year2": None, "pt_year3": None,
        "total_year1": None, "total_year2": None, "total_year3": None,
        "Nphd": None,
        "notes": "",
    }

    if not text.strip():
        record["notes"] = "EMPTY PDF"
        return record

    # Find the PhD graduated section
    # Anchor: "Ph.D students graduated (including Integrated Ph.D)"
    section_match = re.search(
        r"Ph\.D students graduated.*?\n(.*?)\n(.*?)\n(.*?)(?:\n|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if not section_match:
        record["notes"] = "SECTION NOT FOUND"
        return record

    # Extract the 3 lines after the section header
    lines_block = text[section_match.start():]

    # Pull year labels line: e.g. "2023-24 2022-23 2021-22"
    year_label_match = re.search(
        r"(\d{4}-\d{2,4})\s+(\d{4}-\d{2,4})\s+(\d{4}-\d{2,4})",
        lines_block,
    )
    if year_label_match:
        record["year1_label"] = year_label_match.group(1)
        record["year2_label"] = year_label_match.group(2)
        record["year3_label"] = year_label_match.group(3)

    # Pull Full Time row
    ft_match = re.search(
        r"Full\s*Time\s+(\d+)\s+(\d+)\s+(\d+)",
        lines_block,
        re.IGNORECASE,
    )
    if ft_match:
        record["ft_year1"] = int(ft_match.group(1))
        record["ft_year2"] = int(ft_match.group(2))
        record["ft_year3"] = int(ft_match.group(3))

    # Pull Part Time row
    pt_match = re.search(
        r"Part\s*Time\s+(\d+)\s+(\d+)\s+(\d+)",
        lines_block,
        re.IGNORECASE,
    )
    if pt_match:
        record["pt_year1"] = int(pt_match.group(1))
        record["pt_year2"] = int(pt_match.group(2))
        record["pt_year3"] = int(pt_match.group(3))

    # Compute totals and Nphd
    missing = []

    for i, suffix in enumerate(["year1", "year2", "year3"], start=1):
        ft = record[f"ft_{suffix}"]
        pt = record[f"pt_{suffix}"]
        if ft is not None and pt is not None:
            record[f"total_{suffix}"] = ft + pt
        elif ft is not None:
            record[f"total_{suffix}"] = ft
            missing.append(f"pt_year{i} missing, used FT only")
        else:
            missing.append(f"year{i} totals missing")

    totals = [record[f"total_year{i}"] for i in range(1, 4)]
    valid_totals = [t for t in totals if t is not None]

    if len(valid_totals) == 3:
        record["Nphd"] = round(sum(valid_totals) / 3, 4)
    elif len(valid_totals) > 0:
        record["Nphd"] = round(sum(valid_totals) / len(valid_totals), 4)
        missing.append(f"Nphd averaged over {len(valid_totals)} years only")
    else:
        missing.append("Nphd could not be computed")

    if missing:
        record["notes"] = "; ".join(missing)

    return record


# ── BATCH RUNNER ──────────────────────────────────────────────────────────────

def run_batch(root_dir: str, output_csv: str) -> list[dict]:
    root = Path(root_dir)
    records = []
    errors = []

    for year in [2023, 2024, 2025]:
        year_dir = root / str(year)
        if not year_dir.exists():
            print(f"  ⚠ Missing year folder: {year_dir}")
            continue

        college_dirs = sorted(year_dir.iterdir())
        print(f"\n[{year}] Processing {len(college_dirs)} institutions...")

        for college_dir in college_dirs:
            if not college_dir.is_dir():
                continue

            pdfs = list(college_dir.glob("IR-*.pdf"))
            if not pdfs:
                errors.append(f"{year}/{college_dir.name}: no PDF")
                continue

            pdf_path = str(pdfs[0])
            institute_id = pdfs[0].stem

            # Load name from meta.json
            meta_path = college_dir / "meta.json"
            name = college_dir.name
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    name = meta.get("name", name)
                except Exception:
                    pass

            try:
                rec = extract_gphd_features(pdf_path, institute_id, year)
                rec["institute_name"] = name
                records.append(rec)

                status = "✓" if not rec["notes"] else f"⚠  {rec['notes']}"
                nphd_str = f"Nphd={rec['Nphd']:.1f}" if rec["Nphd"] is not None else "Nphd=?"
                print(f"  {status}  [{year}] {name[:50]:<50}  {nphd_str}")

            except Exception as e:
                errors.append(f"{year}/{name}: {str(e)[:80]}")
                print(f"  ✗  [{year}] {name[:50]} — {str(e)[:60]}")

    # Write CSV
    cols = [
        "institute_id", "institute_name", "year",
        "year1_label", "ft_year1", "pt_year1", "total_year1",
        "year2_label", "ft_year2", "pt_year2", "total_year2",
        "year3_label", "ft_year3", "pt_year3", "total_year3",
        "Nphd", "notes",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"\n✓ Saved {len(records)} records → {output_csv}")

    if errors:
        print(f"\n⚠ {len(errors)} errors:")
        for e in errors[:20]:
            print(f"  {e}")

    return records


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_batch(".", "gphd_raw_features.csv")
