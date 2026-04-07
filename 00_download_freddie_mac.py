"""
=============================================================================
Mortgage Credit Risk Modelling  |  Freddie Mac Dataset Downloader
=============================================================================
Script  : 00_download_freddie_mac.py
Purpose : Authenticate with Freddie Mac's portal and download single-family
          loan performance data for the specified origination years.

Usage
-----
  1. Register at https://www.freddiemac.com/research/datasets
  2. Set USERNAME, PASSWORD, START_YEAR, END_YEAR below
  3. python 00_download_freddie_mac.py

Output
------
  data/raw/freddie_mac/
    sample_orig_YYYY.txt   — origination attributes (pipe-delimited, latin-1)
    sample_svcg_YYYY.txt   — monthly servicer updates (pipe-delimited, latin-1)

Design Notes
------------
  - Sequential (not concurrent) downloads — Freddie Mac rate-limits parallel
    connections and concurrent requests corrupt zip archives.
  - Retry logic with exponential backoff on transient network failures.
  - Files kept as pipe-delimited .txt (not converted to CSV) because seller /
    servicer names contain commas; converting breaks those fields downstream.
  - Already-extracted years are skipped so interrupted runs resume cleanly.
  - Zip integrity is validated before extraction.

Next Step
---------
  python 01_data_preprocessing.py
=============================================================================
"""

from __future__ import annotations

import os
import sys
import time
import zipfile
import logging
import requests
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION  —  edit these before running
# =============================================================================

USERNAME   = "your_email@example.com"
PASSWORD   = "your_password"
START_YEAR = 2000
END_YEAR   = 2020

# Output directories
RAW_DIR = Path("data/raw/freddie_mac")   # final .txt files
ZIP_DIR = Path("data/raw/zips")          # temporary zip downloads

# Retry settings
MAX_RETRIES = 3
RETRY_WAIT  = 5   # seconds between attempts

# Freddie Mac endpoints
_BASE        = "https://freddiemac.embs.com/FLoan"
LOGIN_URL    = f"{_BASE}/secure/auth.php"
DOWNLOAD_URL = f"{_BASE}/Data/download.php"
SAMPLE_URL   = f"{_BASE}/Data/sample_{{year}}.zip"

# Browser-like headers — Freddie Mac returns 403 without a recognised UA
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("download.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# AUTHENTICATION
# =============================================================================

def login(session: requests.Session) -> None:
    """
    Authenticate with Freddie Mac.

    Flow:
      1. GET the login page with browser-like headers to obtain a PHPSESSID
         cookie.  Without step 1 the POST returns 403.
      2. POST credentials.  Freddie Mac returns HTTP 200 even on bad passwords
         so we inspect the response body for error text.
      3. POST to accept the Terms & Conditions on the download page.
    """
    session.headers.update(_HEADERS)

    # Step 1 — seed the session cookie
    log.info("Connecting to Freddie Mac …")
    resp = session.get(LOGIN_URL, timeout=30)
    if resp.status_code == 403:
        raise RuntimeError(
            "403 on login GET.  Your IP may be blocked or Freddie Mac has "
            "changed their authentication flow.  Try logging in via browser "
            "and downloading files manually."
        )
    resp.raise_for_status()
    php_cookie = session.cookies.get("PHPSESSID", "")

    # Step 2 — post credentials
    login_resp = session.post(
        LOGIN_URL,
        data={"username": USERNAME, "password": PASSWORD, "cookie": php_cookie},
        timeout=30,
    )
    login_resp.raise_for_status()

    body_lower = login_resp.text.lower()
    if "invalid" in body_lower or "incorrect" in body_lower or "failed" in body_lower:
        raise RuntimeError(
            "Login failed — verify USERNAME and PASSWORD in the script."
        )

    # Step 3 — accept Terms & Conditions
    session.post(
        DOWNLOAD_URL,
        data={
            "accept": "Yes",
            "action": "acceptTandC",
            "acceptSubmit": "Continue",
            "cookie": php_cookie,
        },
        timeout=30,
    )
    log.info("Authenticated successfully.")


# =============================================================================
# DOWNLOAD  &  EXTRACTION
# =============================================================================

def download_zip(session: requests.Session, year: int, zip_path: Path) -> bool:
    """
    Download one year's zip archive.

    Returns True on success, False after MAX_RETRIES failures.
    Uses streaming to avoid holding the entire zip in memory.
    Validates the zip magic bytes before returning.
    """
    url = SAMPLE_URL.format(year=year)
    wait = RETRY_WAIT

    for attempt in range(1, MAX_RETRIES + 1):
        log.info("  [%d] Downloading (attempt %d/%d) …", year, attempt, MAX_RETRIES)
        try:
            with session.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=256 * 1024):
                        f.write(chunk)

            if zipfile.is_zipfile(zip_path):
                size_mb = zip_path.stat().st_size / 1_048_576
                log.info("  [%d] ✓ Downloaded (%.1f MB)", year, size_mb)
                return True

            log.warning("  [%d] Corrupt zip — retrying …", year)
            zip_path.unlink(missing_ok=True)

        except requests.HTTPError as exc:
            log.warning("  [%d] HTTP %s — retrying in %ds …", year, exc.response.status_code, wait)
            zip_path.unlink(missing_ok=True)
        except Exception as exc:
            log.warning("  [%d] %s — retrying in %ds …", year, exc, wait)
            zip_path.unlink(missing_ok=True)

        time.sleep(wait)
        wait *= 2   # exponential backoff

    log.error("  [%d] Failed after %d attempts — skipping.", year, MAX_RETRIES)
    return False


def extract_year(zip_path: Path, year: int, out_dir: Path) -> None:
    """
    Extract origination and servicer files for the given origination year.

    Handles both flat and subdirectory zip layouts produced by different
    Freddie Mac archive generations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [f"sample_orig_{year}.txt", f"sample_svcg_{year}.txt"]

    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_names = zf.namelist()
        for target in targets:
            match = next((n for n in zip_names if n.endswith(target)), None)
            if match is None:
                log.warning("  [%d] %s not found in archive.", year, target)
                log.debug("  Archive contents: %s", zip_names)
                continue

            zf.extract(match, path=out_dir)
            extracted = out_dir / match
            final = out_dir / target
            if extracted != final:
                extracted.rename(final)
            log.info("  [%d] ✓ Extracted %s", year, target)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    start_time = datetime.now()
    log.info("=" * 60)
    log.info("Freddie Mac Sample Data Downloader")
    log.info("Years : %d – %d", START_YEAR, END_YEAR)
    log.info("Output: %s", RAW_DIR.resolve())
    log.info("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ZIP_DIR.mkdir(parents=True, exist_ok=True)

    success: list[int] = []
    skipped: list[int] = []
    failed:  list[int] = []

    with requests.Session() as sess:
        login(sess)

        for year in range(START_YEAR, END_YEAR + 1):
            orig_path = RAW_DIR / f"sample_orig_{year}.txt"
            svcg_path = RAW_DIR / f"sample_svcg_{year}.txt"

            if orig_path.exists() and svcg_path.exists():
                log.info("  [%d] Already extracted — skipping.", year)
                skipped.append(year)
                continue

            zip_path = ZIP_DIR / f"sample_{year}.zip"

            if not download_zip(sess, year, zip_path):
                failed.append(year)
                continue

            extract_year(zip_path, year, RAW_DIR)
            zip_path.unlink(missing_ok=True)   # reclaim disk space
            success.append(year)

            time.sleep(1)   # be courteous to the server

    elapsed = (datetime.now() - start_time).seconds
    log.info("")
    log.info("=" * 60)
    log.info("Download Summary  (%dm %ds)", elapsed // 60, elapsed % 60)
    log.info("  ✓ Downloaded : %s", success  or "—")
    log.info("  — Skipped   : %s", skipped  or "—")
    log.info("  ✗ Failed    : %s", failed   or "—")
    log.info("=" * 60)

    if failed:
        log.warning(
            "Some years failed.  Re-run the script to retry; "
            "already-downloaded years will be skipped automatically."
        )

    if success or skipped:
        log.info("Next step: python 01_data_preprocessing.py")


if __name__ == "__main__":
    main()
