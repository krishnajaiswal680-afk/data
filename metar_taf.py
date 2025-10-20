
metar.py

import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import re, json
from urllib.parse import urlparse
from datetime import datetime, timezone

# -------------------- STEP 1: Parse --------------------
def parse_metars_tafs(text: str) -> dict:
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    data = {}

    # METAR
    for m in re.finditer(r"METAR\s+(V[A-Z]{3,4})\s+([^=]*=)", text, re.I):
        data.setdefault(m.group(1).upper(), {})["METAR"] = re.sub(r"\s+", " ", m.group(2).strip())
    for m in re.finditer(r"(^|\n)\s*(V[A-Z]{3,4})\s+(\d{6}Z\b[^=]*=)", text, re.I | re.M):
        s = m.group(2).upper()
        if "METAR" not in data.get(s, {}):
            data.setdefault(s, {})["METAR"] = re.sub(r"\s+", " ", m.group(3).strip())

    # TAF
    for m in re.finditer(r"TAF\s+(V[A-Z]{3,4})\s+([^=]*=)", text, re.I):
        data.setdefault(m.group(1).upper(), {})["TAF"] = re.sub(r"\s+", " ", m.group(2).strip())
    for m in re.finditer(r"(^|\n)\s*(V[A-Z]{3,4})\s+(\d{6}Z\s+\d{4}/\d{4}[^=]*=)", text, re.I | re.M):
        s = m.group(2).upper()
        if "TAF" not in data.get(s, {}):
            data.setdefault(s, {})["TAF"] = re.sub(r"\s+", " ", m.group(3).strip())

    # No data
    for m in re.finditer(r"No data for\s+(V[A-Z]{3,4})", text, re.I):
        s = m.group(1).upper()
        data.setdefault(s, {}).setdefault("METAR", None)
        data.setdefault(s, {}).setdefault("TAF", None)

    return data

# -------------------- STEP 2: Scrape URL --------------------
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

# -------------------- STEP 3: Classify --------------------
def classify_status(base_value, comparisons):
    values = [v for v in comparisons.values() if v is not None]
    total = len(comparisons)
    present = len(values)

    if base_value is None:
        return "no_data"
    if present == 0:
        return "partial_match"
    elif all(v == base_value for v in values) and present == total:
        return "match"
    elif any(v == base_value for v in values) and not all(v == base_value for v in values):
        return "partial_match"
    elif all(v != base_value for v in values):
        return "no_match"
    else:
        return "partial_match"

# -------------------- STEP 4: Scrape all and compare --------------------
async def scrape_all(sources_file=r"C:\Users\krishna.jaiswal\Downloads\web\metar.txt"):
    with open(sources_file) as f:
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

    for stn, types in combined.items():
        for dtype, info in types.items():
            info["status"] = classify_status(info["base"], info["comparisons"])

    # Instead of saving here, just return structured result for scheduler
    return {"metar": combined, "taf": combined}  # you can adjust keys as needed

# -------------------- STEP 5: Main --------------------
async def main():
    results = await scrape_all()
    return results  # return results for scheduler to append

# -------------------- STEP 6: Run standalone --------------------
if __name__ == "__main__":
    import asyncio
    res = asyncio.run(main())
    print("Scrape complete. Results ready for appending.")







shedule.py











import asyncio
import schedule
import time
import json
from datetime import datetime, timezone
from metar_taf import main  # updated async scraping function

METAR_FILE = "metar.json"
TAF_FILE = "taf.json"

async def run_scraper():
    print("\n🕒 Scraping started...\n")
    result = await main()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if result is None:
        print("⚠️ No data returned from scraper.")
        return

    # Load existing METAR history or create empty list
    try:
        with open(METAR_FILE, "r", encoding="utf-8") as f:
            metar_history = json.load(f)
            # If old format is dict, convert to list
            if isinstance(metar_history, dict):
                metar_history = [metar_history]
    except (FileNotFoundError, json.JSONDecodeError):
        metar_history = []

    # Load existing TAF history or create empty list
    try:
        with open(TAF_FILE, "r", encoding="utf-8") as f:
            taf_history = json.load(f)
            if isinstance(taf_history, dict):
                taf_history = [taf_history]
    except (FileNotFoundError, json.JSONDecodeError):
        taf_history = []

    # Append new scrape with timestamp
    metar_history.append({"timestamp": timestamp, "data": result.get("metar", {})})
    taf_history.append({"timestamp": timestamp, "data": result.get("taf", {})})

    # Save back to JSON
    with open(METAR_FILE, "w", encoding="utf-8") as f:
        json.dump(metar_history, f, indent=2)

    with open(TAF_FILE, "w", encoding="utf-8") as f:
        json.dump(taf_history, f, indent=2)

    print(f"✅ [ {timestamp} ] METAR & TAF appended successfully!\n")


def job():
    try:
        asyncio.run(run_scraper())
    except Exception as e:
        print(f"❌ Error during scraping: {e}")


# Run immediately
job()

# Schedule every 30 minutes (adjust interval if needed)
schedule.every(30).minutes.do(job)

print("🗓️ Scheduler started — scraping every 30 minutes...\n")

while True:
    schedule.run_pending()
    time.sleep(1)







