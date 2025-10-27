import asyncio
import json
import os
import re
import time
import schedule
from datetime import datetime, timezone
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# ==============================================================
# 🧩 STEP 1 — Parse METAR and TAF from raw text
# ==============================================================
def parse_metars_tafs(text: str) -> dict:
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    data = {}

    # METAR extraction
    for m in re.finditer(r"METAR\s+(V[A-Z]{3,4})\s+([^=]*=)", text, re.I):
        data.setdefault(m.group(1).upper(), {})["METAR"] = re.sub(r"\s+", " ", m.group(2).strip())
    for m in re.finditer(r"(^|\n)\s*(V[A-Z]{3,4})\s+(\d{6}Z\b[^=]*=)", text, re.I | re.M):
        stn = m.group(2).upper()
        if "METAR" not in data.get(stn, {}):
            data.setdefault(stn, {})["METAR"] = re.sub(r"\s+", " ", m.group(3).strip())

    # TAF extraction
    for m in re.finditer(r"TAF\s+(V[A-Z]{3,4})\s+([^=]*=)", text, re.I):
        data.setdefault(m.group(1).upper(), {})["TAF"] = re.sub(r"\s+", " ", m.group(2).strip())
    for m in re.finditer(r"(^|\n)\s*(V[A-Z]{3,4})\s+(\d{6}Z\s+\d{4}/\d{4}[^=]*=)", text, re.I | re.M):
        stn = m.group(2).upper()
        if "TAF" not in data.get(stn, {}):
            data.setdefault(stn, {})["TAF"] = re.sub(r"\s+", " ", m.group(3).strip())

    # Handle “No data for ...”
    for m in re.finditer(r"No data for\s+(V[A-Z]{3,4})", text, re.I):
        stn = m.group(1).upper()
        data.setdefault(stn, {}).setdefault("METAR", None)
        data.setdefault(stn, {}).setdefault("TAF", None)

    return data


# ==============================================================
# 🌐 STEP 2 — Scrape one URL
# ==============================================================
async def scrape_url(url: str) -> dict:
    print(f"🌍 Scraping: {url}")
    source_label = urlparse(url).netloc.replace('.', '_')
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(4000)
            html = await page.content()
            await browser.close()
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return {source_label: {}}

    soup = BeautifulSoup(html, "lxml")
    for s in soup(["script", "style"]):
        s.extract()
    text = soup.get_text("\n")
    parsed = parse_metars_tafs(text)
    return {source_label: parsed}


# ==============================================================
# ⚙️ STEP 3 — Classify data consistency
# ==============================================================
def classify_status(base_value, comparisons):
    values = [v for v in comparisons.values() if v]
    if base_value is None:
        return "no_data"
    if not values:
        return "partial_match"
    if all(v == base_value for v in values):
        return "match"
    if any(v == base_value for v in values):
        return "partial_match"
    return "no_match"


# ==============================================================
# 🔁 STEP 4 — Scrape all URLs and combine results
# ==============================================================
async def scrape_all(sources_file="metar.txt"):
    with open(sources_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    base_metar_url, base_taf_url, urls = None, None, []
    for line in lines:
        if line.startswith("BASE_METAR="):
            base_metar_url = line.split("=", 1)[1].strip()
        elif line.startswith("BASE_TAF="):
            base_taf_url = line.split("=", 1)[1].strip()
        else:
            urls.append(line)

    print(f"\n🌐 Base METAR URL: {base_metar_url}")
    print(f"🌐 Base TAF URL:   {base_taf_url}\n")

    base_metar_data = await scrape_url(base_metar_url)
    base_taf_data = await scrape_url(base_taf_url)

    base_metar_stations = list(base_metar_data.values())[0]
    base_taf_stations = list(base_taf_data.values())[0]

    combined = {}

    for url in urls:
        parsed = await scrape_url(url)
        for source, stations in parsed.items():
            for stn, obs in stations.items():
                base_m = base_metar_stations.get(stn, {}).get("METAR")
                base_t = base_taf_stations.get(stn, {}).get("TAF")

                combined.setdefault(stn, {}).setdefault("METAR", {"base": base_m, "comparisons": {}})
                combined.setdefault(stn, {}).setdefault("TAF", {"base": base_t, "comparisons": {}})

                combined[stn]["METAR"]["comparisons"][source] = obs.get("METAR")
                combined[stn]["TAF"]["comparisons"][source] = obs.get("TAF")

    # Assign match status
    for stn, types in combined.items():
        for dtype, info in types.items():
            info["status"] = classify_status(info["base"], info["comparisons"])

    return combined


# ==============================================================
# 💾 STEP 5 — Save data (append mode)
# ==============================================================
def append_results(data: dict):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Prepare new entry
    metar_data = {
        "timestamp": timestamp,
        "data": {stn: info["METAR"] for stn, info in data.items()},
    }
    taf_data = {
        "timestamp": timestamp,
        "data": {stn: info["TAF"] for stn, info in data.items()},
    }

    # Append to existing JSON file
    for filename, new_entry in [("metar.json", metar_data), ("taf.json", taf_data)]:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                try:
                    old_data = json.load(f)
                except json.JSONDecodeError:
                    old_data = []
        else:
            old_data = []

        if not isinstance(old_data, list):
            old_data = [old_data]

        old_data.append(new_entry)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(old_data, f, indent=2)

    print(f"✅ [{timestamp}] Appended new data → metar.json & taf.json")


# ==============================================================
# 🕒 STEP 6 — Scheduled Run
# ==============================================================
async def run_scraper():
    print("\n🕒 Scraping started...\n")
    try:
        results = await scrape_all()
        append_results(results)
        print("✅ Job completed successfully.\n")
    except Exception as e:
        print(f"❌ Error in run_scraper(): {e}")


def job():
    asyncio.run(run_scraper())


# ==============================================================
# 🚀 STEP 7 — Main Loop
# ==============================================================
if __name__ == "__main__":
    job()  # Run immediately once
    schedule.every(30).minutes.do(job)
    print("🗓️ Scheduler started — scraping every 30 minutes...\n")
    while True:
        schedule.run_pending()
        time.sleep(1)
