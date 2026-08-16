#!/usr/bin/env python3
"""
NIRF Engineering Rankings — Mass Downloader v2
Downloads PDFs + score images for top-100 colleges across 3 years (2023-2025).

  - 2023 images are .png  |  2024/2025 images are .jpg  (auto-detected from page)
  - Retries failed downloads up to MAX_RETRIES times with exponential backoff
  - 404 responses are never retried (file genuinely absent on server)
  - Skips already-downloaded files unless --force is used

Output:
  nirf_data/
  ├── all_rankings_meta.csv
  ├── 2025/
  │   └── 001_Indian_Institute_of_Technology_Madras/
  │       ├── IR-E-U-0456.pdf
  │       ├── scores.jpg        ← .jpg for 2024/2025
  │       └── meta.json
  ├── 2024/ ...
  └── 2023/
      └── 001_.../
          ├── IR-E-U-0456.pdf
          ├── scores.png        ← .png for 2023
          └── meta.json

Usage:
  pip install beautifulsoup4
  python nirf_downloader.py                  # normal run
  python nirf_downloader.py --force          # re-download + overwrite everything
  python nirf_downloader.py --retry-failed   # only fix incomplete folders
"""

import re, json, time, ssl, csv, argparse, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

# ── CONFIG ────────────────────────────────────────────────────────────────
YEARS         = [2025, 2024, 2023]
BASE_OUT      = Path("nirf_data")
MAX_WORKERS   = 6       # parallel threads
REQUEST_DELAY = 0.15    # seconds between downloads
MAX_RETRIES   = 3       # per-file retry attempts
RETRY_BACKOFF = 1.5     # seconds; doubles each attempt

RANKING_URL = "https://www.nirfindia.org/Rankings/{year}/EngineeringRanking.html"
PDF_TMPL    = "https://www.nirfindia.org/nirfpdfcdn/{year}/pdf/Engineering/{iid}.pdf"

# ── SSL / HTTP ────────────────────────────────────────────────────────────
SSL_CTX = ssl._create_unverified_context()

def http_get(url, timeout=30):
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (NIRF research)"}
    )
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as r:
        return r.read()

def sanitize(name):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name)
    return re.sub(r'\s+', '_', name.strip())[:80]

# ── DOWNLOAD WITH RETRY ───────────────────────────────────────────────────
def download_file(url, dest, label="", force=False):
    """
    Download url to dest with retry + exponential backoff.
    Returns: "ok" | "skip" | "no_file" | "error:<msg>"
    """
    if not force and dest.exists() and dest.stat().st_size > 500:
        return "skip"

    delay    = RETRY_BACKOFF
    last_err = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = http_get(url, timeout=60)
            if len(data) < 100:
                raise ValueError("Suspiciously small response (%d bytes)" % len(data))
            dest.write_bytes(data)
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "no_file"          # 404 = file absent, never retry
            last_err = "HTTP %d" % e.code
        except Exception as e:
            last_err = str(e)[:80]

        if attempt < MAX_RETRIES:
            print("    ↻  retry %d/%d [%s] — %s" % (attempt, MAX_RETRIES, label, last_err))
            time.sleep(delay)
            delay *= 2

    return "error:" + last_err

# ── PAGE PARSER ───────────────────────────────────────────────────────────
IMAGE_EXTS = (".jpg", ".png")

def parse_page(html, year):
    """
    Parse NIRF ranking HTML → list of college records.
    Uses recursive=False on row.find_all('td') to avoid pollution from
    the nested hidden sub-score tables inside each row.
    Auto-detects .jpg (2024/25) vs .png (2023) image links from the page.
    """
    soup  = BeautifulSoup(html, "html.parser")
    outer = soup.select_one("table.striped") or soup.find("table")
    if not outer:
        return []

    colleges = []
    for row in outer.select("tbody > tr"):
        cells = row.find_all("td", recursive=False)   # direct children only
        if len(cells) < 6:
            continue

        iid = cells[0].get_text(strip=True)
        if not iid.startswith("IR-"):
            continue

        # First real text node in name cell
        name = next(
            (s.strip() for s in cells[1].strings
             if s.strip() and s.strip() not in ("More Details", "Close", "|")),
            iid
        )

        links = cells[1].find_all("a", href=True)

        pdf_url = next(
            (a["href"] for a in links if ".pdf" in a["href"].lower()),
            PDF_TMPL.format(year=year, iid=iid)
        )

        # Image URL — handles both .jpg (2024/25) and .png (2023)
        img_url = next(
            (a["href"] for a in links
             if any(a["href"].lower().endswith(ext) for ext in IMAGE_EXTS)),
            None
        )

        # Sub-metric scores from the hidden expandable table
        sub = {}
        hidden = cells[1].find("div", class_="tbl_hidden")
        if hidden:
            ths = [t.get_text(strip=True) for t in hidden.find_all("th")]
            tds = [t.get_text(strip=True) for t in hidden.find_all("td")]
            for h, v in zip(ths, tds):
                try:    sub[h] = float(v)
                except: sub[h] = v

        try:    score = float(cells[4].get_text(strip=True))
        except: score = 0.0
        try:    rank  = int(cells[5].get_text(strip=True))
        except: rank  = 0

        colleges.append({
            "year":        year,
            "rank":        rank,
            "iid":         iid,
            "name":        name,
            "city":        cells[2].get_text(strip=True),
            "state":       cells[3].get_text(strip=True),
            "total_score": score,
            "TLR":         sub.get("TLR (100)", ""),
            "RP":          sub.get("RPC (100)", ""),
            "GO":          sub.get("GO (100)", ""),
            "OI":          sub.get("OI (100)", ""),
            "PR":          sub.get("PERCEPTION (100)", ""),
            "pdf_url":     pdf_url,
            "img_url":     img_url,   # None only if page has no image link at all
        })

    return colleges

# ── PER-COLLEGE DOWNLOADER ────────────────────────────────────────────────
def download_college(college, force=False):
    year  = college["year"]
    rank  = college["rank"]
    iid   = college["iid"]
    name  = college["name"]

    folder = BASE_OUT / str(year) / ("%03d_%s" % (rank, sanitize(name)))
    folder.mkdir(parents=True, exist_ok=True)

    # Always overwrite meta.json (zero cost, always up-to-date)
    meta = {k: v for k, v in college.items() if k not in ("pdf_url", "img_url")}
    (folder / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # PDF
    pdf_result = download_file(
        college["pdf_url"],
        folder / (iid + ".pdf"),
        label="[%s#%d] PDF" % (year, rank),
        force=force,
    )
    time.sleep(REQUEST_DELAY)

    # Image — preserve original extension (.jpg or .png)
    img_result = "no_file"
    if college["img_url"]:
        raw_ext  = Path(college["img_url"].split("?")[0]).suffix.lower() or ".jpg"
        img_dest = folder / ("scores" + raw_ext)
        img_result = download_file(
            college["img_url"],
            img_dest,
            label="[%s#%d] IMG" % (year, rank),
            force=force,
        )
        time.sleep(REQUEST_DELAY)

    pdf_ok = pdf_result in ("ok", "skip")
    img_ok = img_result in ("ok", "skip")
    img_na = img_result == "no_file"

    if   pdf_ok and img_ok:  status = "ok"
    elif pdf_ok and img_na:  status = "ok-no-img"
    elif pdf_ok:             status = "partial-pdf"
    elif img_ok:             status = "partial-img"
    else:                    status = "failed"

    return dict(meta, pdf_result=pdf_result, img_result=img_result, status=status)

# ── HELPERS FOR --retry-failed ────────────────────────────────────────────
def is_incomplete(c):
    folder   = BASE_OUT / str(c["year"]) / ("%03d_%s" % (c["rank"], sanitize(c["name"])))
    pdf_path = folder / (c["iid"] + ".pdf")
    pdf_ok   = pdf_path.exists() and pdf_path.stat().st_size > 500
    has_img  = any(folder.glob("scores.*"))
    img_ok   = (not c["img_url"]) or has_img
    return not (pdf_ok and img_ok)

# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="NIRF Engineering Rankings downloader — 2023/2024/2025"
    )
    ap.add_argument("--force",        action="store_true",
                    help="Re-download and overwrite all existing files")
    ap.add_argument("--retry-failed", action="store_true",
                    help="Only process colleges with missing or incomplete files")
    args = ap.parse_args()

    BASE_OUT.mkdir(exist_ok=True)
    all_records = []

    print("=" * 72)
    print("  NIRF Engineering Rankings Downloader — 2023 · 2024 · 2025")
    if args.force:
        print("  Mode: FORCE — re-downloading and overwriting everything")
    elif args.retry_failed:
        print("  Mode: RETRY-FAILED — only fixing incomplete folders")
    else:
        print("  Mode: NORMAL — skipping already-downloaded files")
    print("=" * 72)

    # ── Step 1: Scrape ranking pages ──────────────────────────────────────
    for year in YEARS:
        url = RANKING_URL.format(year=year)
        print("\n[%d] %s" % (year, url))
        try:
            html     = http_get(url).decode("utf-8", errors="replace")
            colleges = parse_page(html, year)
            imgs     = sum(1 for c in colleges if c["img_url"])
            print("     → %d colleges  |  image links found: %d/%d"
                  % (len(colleges), imgs, len(colleges)))
            all_records.extend(colleges)
        except Exception as e:
            print("     ✗ Failed: %s" % e)

    if not all_records:
        print("\nNo records scraped. Exiting.")
        return

    # ── Step 2: Master CSV ────────────────────────────────────────────────
    csv_path = BASE_OUT / "all_rankings_meta.csv"
    csv_cols = ["year","rank","iid","name","city","state",
                "total_score","TLR","RP","GO","OI","PR"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_records)
    print("\nMaster CSV → %s  (%d rows)" % (csv_path, len(all_records)))

    # ── Step 3: Filter if --retry-failed ──────────────────────────────────
    to_process = all_records
    if args.retry_failed:
        to_process = [c for c in all_records if is_incomplete(c)]
        done       = len(all_records) - len(to_process)
        print("\n%d colleges need work  (%d already complete)" % (len(to_process), done))

    # ── Step 4: Download in parallel ──────────────────────────────────────
    force_flag = args.force or args.retry_failed
    print("\n⬇%d colleges × up to 2 files  |  workers=%d  retries=%d"
          % (len(to_process), MAX_WORKERS, MAX_RETRIES))
    print("-" * 72)

    counts = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(download_college, c, force_flag): c for c in to_process}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                s = r["status"]
                icon  = "✓" if s.startswith("ok") else ("⚠" if s.startswith("partial") else "✗")
                label = {
                    "ok":           "✓",
                    "ok-no-img":    "✓ (no img on server)",
                    "partial-pdf":  "⚠ pdf-only",
                    "partial-img":  "⚠ img-only",
                    "failed":       "✗ both failed",
                }.get(s, s)
                print("  %s  [%s] #%03d  %-46s  %s"
                      % (icon, r["year"], r["rank"], r["name"][:46], label))
                counts[s] = counts.get(s, 0) + 1
            except Exception as e:
                counts["failed"] = counts.get("failed", 0) + 1
                print("  ✗  Unexpected error: %s" % e)

    # ── Step 5: Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Done.")
    print("  ✓ full        : %d" % counts.get("ok", 0))
    print("  ✓ no-img      : %d  (server has no image for these)" % counts.get("ok-no-img", 0))
    print("  ⚠ partial-pdf : %d" % counts.get("partial-pdf", 0))
    print("  ⚠ partial-img : %d" % counts.get("partial-img", 0))
    print("  ✗ failed      : %d" % counts.get("failed", 0))
    print("  Output : %s" % BASE_OUT.resolve())
    print("  CSV    : %s" % csv_path.resolve())
    print("=" * 72)

if __name__ == "__main__":
    main()