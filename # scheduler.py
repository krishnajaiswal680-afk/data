

# scheduler.py
import asyncio
import time
import schedule
from metar_taf import run_scraper

# ============================================================== 
# 🕒 Scheduled Run
# ============================================================== 
def job():
    asyncio.run(run_scraper())


# ============================================================== 
# 🚀 Main Loop
# ============================================================== 
if __name__ == "__main__":
    job()  # Run immediately once
    schedule.every(15).minutes.do(job)
    print("🗓️ Scheduler started — scraping every 15 minutes...\n")
    while True:
        schedule.run_pending()
        time.sleep(1)
