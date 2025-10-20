

# #---------------------------------------------------

# import asyncio
# from playwright.async_api import async_playwright
# from bs4 import BeautifulSoup
# import re, json, os
# from urllib.parse import urlparse
# from datetime import datetime


# # ------------------------------------------------------------
# # STEP 1: Parse METAR and TAF strings from scraped text
# # ------------------------------------------------------------
# def parse_metars_tafs(text: str) -> dict:
#     """
#     Parse METAR and TAF strings into structured dictionary:
#     { 'VABB': {'METAR': '251800Z ... =', 'TAF': '251730Z ... ='} }
#     """
#     text = text.replace('\xa0', ' ').replace('\u200b', '')
#     data = {}

#     # ----- METAR -----
#     for m in re.finditer(r"METAR\s+(V[A-Z]{3,4})\s+([^=]*=)", text, re.I):
#         data.setdefault(m.group(1).upper(), {})["METAR"] = re.sub(r"\s+", " ", m.group(2).strip())

#     for m in re.finditer(r"(^|\n)\s*(V[A-Z]{3,4})\s+(\d{6}Z\b[^=]*=)", text, re.I | re.M):
#         station = m.group(2).upper()
#         if "METAR" not in data.get(station, {}):
#             data.setdefault(station, {})["METAR"] = re.sub(r"\s+", " ", m.group(3).strip())

#     # ----- TAF -----
#     for m in re.finditer(r"TAF\s+(V[A-Z]{3,4})\s+([^=]*=)", text, re.I):
#         data.setdefault(m.group(1).upper(), {})["TAF"] = re.sub(r"\s+", " ", m.group(2).strip())

#     for m in re.finditer(r"(^|\n)\s*(V[A-Z]{3,4})\s+(\d{6}Z\s+\d{4}/\d{4}[^=]*=)", text, re.I | re.M):
#         station = m.group(2).upper()
#         if "TAF" not in data.get(station, {}):
#             data.setdefault(station, {})["TAF"] = re.sub(r"\s+", " ", m.group(3).strip())

#     # ----- No data -----
#     for m in re.finditer(r"No data for\s+(V[A-Z]{3,4})", text, re.I):
#         station = m.group(1).upper()
#         data.setdefault(station, {}).setdefault("METAR", None)
#         data.setdefault(station, {}).setdefault("TAF", None)

#     return data


# # ------------------------------------------------------------
# # STEP 2: Scrape URL dynamically using Playwright
# # ------------------------------------------------------------
# async def scrape_url(url: str) -> dict:
#     print(f"🌍 Scraping: {url}")
#     source_label = urlparse(url).netloc.replace('.', '_')
#     try:
#         async with async_playwright() as p:
#             browser = await p.chromium.launch(headless=True)
#             page = await browser.new_page()
#             await page.goto(url, wait_until="domcontentloaded", timeout=60000)
#             await page.wait_for_timeout(4000)
#             html = await page.content()
#             await browser.close()
#     except Exception as e:
#         print(f"❌ Error scraping {url}: {e}")
#         return {source_label: {}}

#     soup = BeautifulSoup(html, "lxml")
#     for s in soup(["script", "style"]):
#         s.extract()
#     text = soup.get_text("\n")
#     parsed = parse_metars_tafs(text)
#     return {source_label: parsed}


# # ------------------------------------------------------------
# # STEP 3: Core comparison logic (7 test cases)
# # ------------------------------------------------------------
# def classify_status(base_value, comparisons):
#     """
#     Implements 7-case comparison matrix.

#     Returns: match / partial_match / no_match / no_data
#     (Only valid statuses per IndiGo use)
#     """
#     present = [list(c.keys())[0] for c in comparisons if list(c.values())[0] is not None]
#     missing = [list(c.keys())[0] for c in comparisons if list(c.values())[0] is None]
#     values = [list(c.values())[0] for c in comparisons if list(c.values())[0] is not None]

#     # --- Case 5,6,7: Base missing ---
#     if base_value is None:
#         if not values:  # all missing
#             return "no_data"  # Case 7
#         elif len(values) < len(comparisons):
#             return "no_data"  # Case 6
#         else:
#             return "no_data"  # Case 5

#     # --- Base valid ---
#     if not values:
#         return "partial_match"  # Case 4

#     all_same = all(v == base_value for v in values)
#     any_same = any(v == base_value for v in values)

#     if all_same and len(values) == len(comparisons):
#         return "match"  # Case 1
#     elif any_same and not all_same:
#         return "partial_match"  # Case 2
#     elif all(v != base_value for v in values):
#         return "no_match"  # Case 3

#     return "partial_match"  # fallback safety


# # ------------------------------------------------------------
# # STEP 4: Aggregate and compare across sources
# # ------------------------------------------------------------
# async def scrape_all(sources_file=r"C:\Users\krishna.jaiswal\Downloads\web\metar.txt"):
#     with open(sources_file) as f:
#         lines = [line.strip() for line in f if line.strip()]

#     base_metar_url, base_taf_url = None, None
#     urls = []
#     for line in lines:
#         if line.startswith("BASE_METAR="):
#             base_metar_url = line.split("=", 1)[1].strip()
#         elif line.startswith("BASE_TAF="):
#             base_taf_url = line.split("=", 1)[1].strip()
#         else:
#             urls.append(line)

#     print(f"\n🌐 Base METAR URL: {base_metar_url}")
#     print(f"🌐 Base TAF URL:   {base_taf_url}\n")

#     base_metar_data = await scrape_url(base_metar_url)
#     base_taf_data = await scrape_url(base_taf_url)

#     base_metar_stations = list(base_metar_data.values())[0]
#     base_taf_stations = list(base_taf_data.values())[0]

#     combined = {}

#     # Compare each additional source against base
#     for url in urls:
#         print(f"🔍 Comparing -> {url}")
#         parsed = await scrape_url(url)
#         for source, stations in parsed.items():
#             for station, obs in stations.items():
#                 base_m = base_metar_stations.get(station, {}).get("METAR")
#                 m_value = obs.get("METAR")
#                 combined.setdefault(station, {}).setdefault("METAR", {"base": base_m, "comparisons": []})
#                 combined[station]["METAR"]["comparisons"].append({source: m_value})

#                 base_t = base_taf_stations.get(station, {}).get("TAF")
#                 t_value = obs.get("TAF")
#                 combined.setdefault(station, {}).setdefault("TAF", {"base": base_t, "comparisons": []})
#                 combined[station]["TAF"]["comparisons"].append({source: t_value})

#     # Classify per station
#     for station, types in combined.items():
#         for dtype, info in types.items():
#             status = classify_status(info["base"], info["comparisons"])
#             info["status"] = status

#     return combined


# # ------------------------------------------------------------
# # STEP 5: Save results per status
# # ------------------------------------------------------------
# def save_results(data: dict):
#     timestamp = datetime.utcnow().isoformat() + "Z"
#     os.makedirs("metar", exist_ok=True)
#     os.makedirs("taf", exist_ok=True)

#     categorized = {
#         "METAR": {"match": {}, "partial_match": {}, "no_match": {}, "no_data": {}},
#         "TAF": {"match": {}, "partial_match": {}, "no_match": {}, "no_data": {}}
#     }

#     for station, types in data.items():
#         for dtype, info in types.items():
#             categorized[dtype][info["status"]][station] = {dtype: info}

#     for dtype in ["METAR", "TAF"]:
#         for status, stations in categorized[dtype].items():
#             if stations:
#                 folder = dtype.lower()
#                 filename = os.path.join(folder, f"results_{dtype.lower()}_{status}.json")
#                 with open(filename, "w") as f:
#                     json.dump({timestamp: stations}, f, indent=2)
#                 print(f"✅ Saved {len(stations)} {dtype} stations in {filename}")


# # ------------------------------------------------------------
# # STEP 6: Run end-to-end
# # ------------------------------------------------------------
# async def main():
#     results = await scrape_all()
#     save_results(results)


# if __name__ == "__main__":
#     asyncio.run(main())





#-------------updated code ----------------------








# import asyncio
# from playwright.async_api import async_playwright
# from bs4 import BeautifulSoup
# import re, json, os
# from urllib.parse import urlparse
# from datetime import datetime

# # ============================================================
# # STEP 1: Parse METAR & TAF Data from HTML Text
# # ============================================================
# def parse_metars_tafs(text: str) -> dict:
#     """
#     Parses METAR and TAF strings from the page text into:
#     { 'VABB': {'METAR': '080500Z ... =', 'TAF': '080530Z ... ='} }
#     """
#     text = text.replace('\xa0', ' ').replace('\u200b', '')
#     data = {}

#     # ----- METAR -----
#     for m in re.finditer(r"METAR\s+(V[A-Z]{3,4})\s+([^=]*=)", text, re.I):
#         data.setdefault(m.group(1).upper(), {})["METAR"] = re.sub(r"\s+", " ", m.group(2).strip())
#     for m in re.finditer(r"(^|\n)\s*(V[A-Z]{3,4})\s+(\d{6}Z\b[^=]*=)", text, re.I | re.M):
#         st = m.group(2).upper()
#         if "METAR" not in data.get(st, {}):
#             data.setdefault(st, {})["METAR"] = re.sub(r"\s+", " ", m.group(3).strip())

#     # ----- TAF -----
#     for m in re.finditer(r"TAF\s+(V[A-Z]{3,4})\s+([^=]*=)", text, re.I):
#         data.setdefault(m.group(1).upper(), {})["TAF"] = re.sub(r"\s+", " ", m.group(2).strip())
#     for m in re.finditer(r"(^|\n)\s*(V[A-Z]{3,4})\s+(\d{6}Z\s+\d{4}/\d{4}[^=]*=)", text, re.I | re.M):
#         st = m.group(2).upper()
#         if "TAF" not in data.get(st, {}):
#             data.setdefault(st, {})["TAF"] = re.sub(r"\s+", " ", m.group(3).strip())

#     # ----- "No data for" -----
#     for m in re.finditer(r"No data for\s+(V[A-Z]{3,4})", text, re.I):
#         st = m.group(1).upper()
#         data.setdefault(st, {}).setdefault("METAR", None)
#         data.setdefault(st, {}).setdefault("TAF", None)

#     return data


# # ============================================================
# # STEP 2: Scrape URL using Playwright
# # ============================================================
# async def scrape_url(url: str) -> dict:
#     """
#     Launches a headless browser to scrape data from URL.
#     Returns a parsed {source_label: {...}} dict.
#     """
#     print(f"🌍 Scraping: {url}")
#     source_label = urlparse(url).netloc.replace('.', '_')
#     try:
#         async with async_playwright() as p:
#             browser = await p.chromium.launch(headless=True)
#             page = await browser.new_page()
#             await page.goto(url, wait_until="domcontentloaded", timeout=60000)
#             await page.wait_for_timeout(3000)
#             html = await page.content()
#             await browser.close()
#     except Exception as e:
#         print(f"❌ Error scraping {url}: {e}")
#         return {source_label: {}}

#     soup = BeautifulSoup(html, "lxml")
#     [s.extract() for s in soup(["script", "style"])]
#     text = soup.get_text("\n")
#     return {source_label: parse_metars_tafs(text)}


# # ============================================================
# # STEP 3: Classify Status — Implements 7 IndiGo Cases
# # ============================================================
# def classify_status(base_value, comparisons):
#     """
#     Implements exact 7 IndiGo test cases.
#     Returns: 'match', 'partial_match', 'no_match', or 'no_data'
#     """
#     values = [list(c.values())[0] for c in comparisons]
#     total_sources = len(comparisons)
#     non_null = [v for v in values if v is not None]
#     null_count = total_sources - len(non_null)

#     # --- CASE 5,6,7: Base missing ---
#     if base_value is None:
#         if not non_null:
#             return "no_data"  # Case 7: base & all missing
#         elif null_count > 0:
#             return "no_data"  # Case 6: base missing + some missing
#         else:
#             return "no_data"  # Case 5: base missing, others valid

#     # --- CASE 4: Base valid, some missing ---
#     if null_count > 0:
#         return "partial_match"

#     # --- CASE 1,2,3 ---
#     if all(v == base_value for v in non_null):
#         return "match"  # Case 1: All identical
#     elif any(v == base_value for v in non_null):
#         return "partial_match"  # Case 2: Some same, some differ
#     else:
#         return "no_match"  # Case 3: All differ


# # ============================================================
# # STEP 4: Compare Across All Sources
# # ============================================================
# async def scrape_all(sources_file=r"C:\Users\krishna.jaiswal\Downloads\web\metar.txt"):
#     with open(sources_file) as f:
#         lines = [line.strip() for line in f if line.strip()]

#     base_metar_url = next((l.split("=", 1)[1] for l in lines if l.startswith("BASE_METAR=")), None)
#     base_taf_url = next((l.split("=", 1)[1] for l in lines if l.startswith("BASE_TAF=")), None)
#     urls = [l for l in lines if not l.startswith("BASE_")]

#     print(f"\n🌐 Base METAR URL: {base_metar_url}")
#     print(f"🌐 Base TAF URL:   {base_taf_url}\n")

#     base_metar_data = await scrape_url(base_metar_url)
#     base_taf_data = await scrape_url(base_taf_url)

#     base_metar_stations = list(base_metar_data.values())[0]
#     base_taf_stations = list(base_taf_data.values())[0]

#     combined = {}

#     for url in urls:
#         print(f"🔍 Comparing -> {url}")
#         parsed = await scrape_url(url)
#         for source, stations in parsed.items():
#             for st, obs in stations.items():
#                 # METAR comparison
#                 base_m = base_metar_stations.get(st, {}).get("METAR")
#                 m_value = obs.get("METAR")
#                 combined.setdefault(st, {}).setdefault("METAR", {"base": base_m, "comparisons": []})
#                 combined[st]["METAR"]["comparisons"].append({source: m_value})

#                 # TAF comparison
#                 base_t = base_taf_stations.get(st, {}).get("TAF")
#                 t_value = obs.get("TAF")
#                 combined.setdefault(st, {}).setdefault("TAF", {"base": base_t, "comparisons": []})
#                 combined[st]["TAF"]["comparisons"].append({source: t_value})

#     # Classification per station
#     for st, types in combined.items():
#         for dtype, info in types.items():
#             info["status"] = classify_status(info["base"], info["comparisons"])

#     return combined


# # ============================================================
# # STEP 5: Save Unified JSON Results (METAR + TAF)
# # ============================================================
# def save_results(data: dict):
#     timestamp = datetime.utcnow().isoformat() + "Z"

#     metar_results = {"match": {}, "partial_match": {}, "no_match": {}, "no_data": {}}
#     taf_results = {"match": {}, "partial_match": {}, "no_match": {}, "no_data": {}}

#     for st, types in data.items():
#         for dtype, info in types.items():
#             if dtype == "METAR":
#                 metar_results[info["status"]][st] = info
#             elif dtype == "TAF":
#                 taf_results[info["status"]][st] = info

#     # Save two JSONs
#     with open("results_metar.json", "w") as f:
#         json.dump({timestamp: metar_results}, f, indent=2)
#     with open("results_taf.json", "w") as f:
#         json.dump({timestamp: taf_results}, f, indent=2)

#     print("\n✅ Results saved -> results_metar.json, results_taf.json\n")


# # ============================================================
# # STEP 6: Run End-to-End Pipeline
# # ============================================================
# async def main():
#     results = await scrape_all()
#     save_results(results)


# if __name__ == "__main__":
#     asyncio.run(main())



#------------------upgrad the code again -----------------------------






import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import re, json
from urllib.parse import urlparse
from datetime import datetime


# ------------------------------------------------------------
# STEP 1: Parse METAR and TAF
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 2: Scrape URL with Playwright
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 3: Classify status (7 IndiGo test cases)
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# STEP 4: Scrape all and compare
# ------------------------------------------------------------
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

    return combined


# ------------------------------------------------------------
# STEP 5: Save as two files — metar.json & taf.json
# ------------------------------------------------------------
def save_results(data: dict):
    timestamp = datetime.utcnow().isoformat() + "Z"
    result_metar = {"timestamp": timestamp, "data": {"match": {}, "partial_match": {}, "no_match": {}, "no_data": {}}}
    result_taf = {"timestamp": timestamp, "data": {"match": {}, "partial_match": {}, "no_match": {}, "no_data": {}}}

    for stn, types in data.items():
        for dtype, info in types.items():
            entry = {
                "base": info["base"],
                "comparisons": info["comparisons"],
                "status": info["status"]
            }
            if dtype == "METAR":
                result_metar["data"][info["status"]][stn] = entry
            elif dtype == "TAF":
                result_taf["data"][info["status"]][stn] = entry

    with open("metar.json", "w") as f:
        json.dump(result_metar, f, indent=2)
    with open("taf.json", "w") as f:
        json.dump(result_taf, f, indent=2)

    print(f"✅ Saved METAR → metar.json")
    print(f"✅ Saved TAF   → taf.json")


# ------------------------------------------------------------
# STEP 6: Run end-to-end
# ------------------------------------------------------------
async def main():
    results = await scrape_all()
    save_results(results)


if __name__ == "__main__":
    asyncio.run(main())
