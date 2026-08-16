"""
NIRF Scores Extractor
=====================
Reads scores images from each college folder and extracts all sub-parameter scores.

- 2023: scores.png  → Tesseract OCR (PNG, clean)
- 2024: scores.jpg  → EasyOCR (JPEG)
- 2025: scores.jpg  → EasyOCR (JPEG)

Column order (0-indexed):
  0:SS  1:FSR  2:FQE  3:FRU  4:PU  5:QP  6:IPR  7:FPPP
  8:GPH  9:GUE  10:MS  11:GPHD  12:RD  13:WD  14:ESCS  15:PCS  16:PR

Output: scores_extracted.csv
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import re
import csv
import json
import glob
from pathlib import Path
from PIL import Image, ImageEnhance

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# EasyOCR loaded once (slow to init)
_easyocr_reader = None

def get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return _easyocr_reader


# ── CONFIG ────────────────────────────────────────────────────────────────────

YEARS = [2023, 2024, 2025]

COLUMNS = ['SS', 'FSR', 'FQE', 'FRU', 'PU', 'QP', 'IPR', 'FPPP',
           'GPH', 'GUE', 'MS', 'GPHD', 'RD', 'WD', 'ESCS', 'PCS', 'PR']

# Table crop ratios by year (fraction of height from top)
TABLE_TOP_RATIO = {
    2023: 0.62,
    2024: 0.35,   # EasyOCR works on full image crop
    2025: 0.35,
}


# ── TESSERACT (2023 PNG) ──────────────────────────────────────────────────────

def extract_tesseract(img_path: str) -> list[float] | None:
    """Extract score row using Tesseract. Best for 2023 PNGs."""
    img = Image.open(img_path)
    w, h = img.size
    table = img.crop((0, int(h * 0.62), w, h))
    table = table.convert('RGB')
    table = ImageEnhance.Contrast(table).enhance(2.0)
    table = table.resize((table.width * 2, table.height * 2), Image.LANCZOS)
    table = table.convert('L')

    text = pytesseract.image_to_string(table, config='--psm 6')
    return _parse_score_row(text.split('\n'))


# ── EASYOCR (2024/2025 JPEG) ──────────────────────────────────────────────────

def extract_easyocr(img_path: str) -> list[float] | None:
    """Extract score row using EasyOCR. Best for 2024/2025 JPEGs."""
    import numpy as np
    reader = get_easyocr()
    img = Image.open(img_path)
    w, h = img.size

    # Crop to table area
    table = img.crop((0, int(h * 0.35), w, h))
    results = reader.readtext(np.array(table))

    # Collect all detected text items
    all_text = [text for (_, text, conf) in results if conf > 0.4]

    # Find Score row: sequence of 17 decimal numbers after 'Score'
    score_vals = []
    in_score = False
    for token in all_text:
        token = token.strip()
        if re.match(r'^[Ss]core$', token):
            in_score = True
            score_vals = []
            continue
        if re.match(r'^[Tt]otal$', token) and in_score:
            break
        if in_score:
            # Accept decimal numbers
            m = re.match(r'^(\d+\.\d+)$', token)
            if m:
                score_vals.append(float(m.group(1)))

    if len(score_vals) == 17:
        return score_vals

    # Fallback: find any line/sequence with 17 decimal numbers
    all_nums = []
    for token in all_text:
        m = re.match(r'^(\d+\.\d+)$', token.strip())
        if m:
            all_nums.append(float(m.group(1)))
        else:
            if len(all_nums) == 17:
                break
            if len(all_nums) > 0 and not re.match(r'.*\d.*', token):
                continue
            all_nums = []

    # Try sliding window of 17
    for i in range(len(all_nums) - 16):
        window = all_nums[i:i+17]
        # Score row: last value should be ~100 (PR), total max is 100
        if 90 <= window[-1] <= 100:
            return window

    return None


# ── PARSER HELPER ─────────────────────────────────────────────────────────────

def _parse_score_row(lines: list[str]) -> list[float] | None:
    """Find the Score row (17 decimal numbers, not all round) from text lines."""
    for line in lines:
        nums = re.findall(r'\d+\.\d+', line)
        if len(nums) == 17:
            non_round = sum(1 for n in nums if not n.endswith('.00'))
            if non_round >= 3:
                return [float(n) for n in nums]

    # Fallback: first line with 17 numbers
    for line in lines:
        nums = re.findall(r'\d+\.\d+', line)
        if len(nums) == 17:
            return [float(n) for n in nums]

    return None


# ── MAIN EXTRACTOR ────────────────────────────────────────────────────────────

def extract_score_row(img_path: str, year: int) -> dict | None:
    """Extract all sub-parameter scores from a scores image."""
    try:
        if year == 2023:
            vals = extract_tesseract(img_path)
        else:
            vals = extract_easyocr(img_path)

        if vals is None or len(vals) != 17:
            return None

        return {col: val for col, val in zip(COLUMNS, vals)}

    except Exception as e:
        return None


# ── BATCH RUNNER ──────────────────────────────────────────────────────────────

def run_batch(output_csv: str = 'scores_extracted.csv'):
    records = []
    errors = []

    for year in YEARS:
        ext = 'png' if year == 2023 else 'jpg'
        pattern = f'{year}/*/scores.{ext}'
        image_paths = sorted(glob.glob(pattern))

        print(f'\n[{year}] Found {len(image_paths)} score images...')

        for img_path in image_paths:
            college_dir = Path(img_path).parent

            meta_path = college_dir / 'meta.json'
            institute_id = college_dir.name
            institute_name = college_dir.name

            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    institute_id = meta.get('iid', institute_id)
                    institute_name = meta.get('name', institute_name)
                except Exception:
                    pass

            scores = extract_score_row(img_path, year)

            if scores:
                rec = {'institute_id': institute_id,
                       'institute_name': institute_name,
                       'year': year}
                rec.update(scores)
                records.append(rec)
                print(f'  ✓  [{year}] {institute_name[:50]:<50} GPHD={scores["GPHD"]}')
            else:
                errors.append(f'{year}/{institute_name}')
                print(f'  ✗  [{year}] {institute_name[:50]}')

    if records:
        fieldnames = ['institute_id', 'institute_name', 'year'] + COLUMNS
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(records)
        print(f'\n✓ Saved {len(records)} records → {output_csv}')
    else:
        print('\n✗ No records extracted.')

    if errors:
        print(f'\n⚠ {len(errors)} failures:')
        for e in errors[:10]:
            print(f'  {e}')

    return records


# ── SINGLE TEST ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if '--batch' in sys.argv:
        run_batch()
    else:
        print('=== TEST: IIT Madras across all years ===\n')
        tests = [
            (2023, '2023/001_Indian_Institute_of_Technology_Madras/scores.png'),
            (2024, '2024/001_Indian_Institute_of_Technology_Madras/scores.jpg'),
            (2025, '2025/001_Indian_Institute_of_Technology_Madras/scores.jpg'),
        ]
        for year, path in tests:
            scores = extract_score_row(path, year)
            if scores:
                print(f'[{year}] GPHD={scores["GPHD"]}  SS={scores["SS"]}  PR={scores["PR"]}')
            else:
                print(f'[{year}] FAILED')
