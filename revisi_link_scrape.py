import pandas as pd
import time
import random
import os
import winsound
from urllib.parse import unquote
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# --- KONFIGURASI GLOBAL ---
START_DATE_GLOBAL = "01/06/2025" # Format: MM/DD/YYYY
END_DATE_GLOBAL   = "11/30/2025" # Format: MM/DD/YYYY
OUTPUT_FILE       = "pestle_link.csv"
JUMLAH_HALAMAN_PER_BULAN = 10

# Keyword PESTLE
PESTLE_KEYWORDS = {
    'Political': [
        "Badan Gizi Nasional", "Makan Bergizi Gratis", "Prabowo Makan Gratis", "SPPG Makan Bergizi Gratis",
        "Janji Makan Siang Gratis", "Kebijakan Makan Bergizi", "Dukungan Partai Makan Bergizi"
    ],
    'Economic': [
        "Anggaran Makan Bergizi", "Kebocoran Dana Makan Gratis", "Dampak Ekonomi Makan Bergizi Gratis", 
        "Subsidi dan Efisiensi Makan Bergizi Gratis", "UMKM Makan Bergizi", "Inflasi Pangan Makan Gratis"
    ],
    'Social': [
        "Peran Media Sosial Makan Bergizi Gratis", "Gizi Anak Sekolah", "Menu Makan Bergizi Sekolah", 
        "Stunting Makan Bergizi", "Isu Keracunan Makan Siang Gratis", "Keluhan Makan Siang Gratis", "Dapur Makan Bergizi Gratis",
    ],
    'Technological': [
        "Aplikasi Makan Bergizi", "Data Siswa Makan Gratis", "Sistem Distribusi Makanan", 
        "Digitalisasi Badan Gizi", "Dashboard Pantau Makan Gratis", "Teknologi Pangan Makan Bergizi"
    ],
    'Legal': [
        "BPOM Pengawasan Makanan Sekolah", "Aturan Makan Bergizi Gratis", "Sertifikasi Halal Makan Bergizi Gratis", 
        "Audit Program Makan Bergizi Gratis", "Sanksi Pelanggaran Makan Siang Gratis", "Korupsi Dana Makan Gratis",
        "penyidikan KPK dana makan gratis", "UU Pangan No 18 2012 penerapan", "pengaduan masyarakat ke Ombudsman program makan",
        "putusan pengadilan program makan bergizi"
    ],
    'Environmental': [
        "Limbah dan Sampah Sisa Makan Siang Gratis", "Penggunaan Bahan Makanan MBG", "Food Waste Makan Bergizi",
        "Dampak Lingkungan Makan Bergizi Gratis", "Daur Ulang Makan Siang Gratis", "Kotak Makan Ramah Lingkungan MBG"
    ]
}

def setup_driver():
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def alert_user_captcha():
    try:
        for _ in range(3):
            winsound.Beep(1000, 500)
            time.sleep(0.2)
    except:
        pass

def clean_google_link(url):
    if "google.com/url" in url or "google.co.id/url" in url:
        try:
            start = url.find("q=") + 2
            end = url.find("&", start)
            clean_url = url[start:] if end == -1 else url[start:end]
            return unquote(clean_url)
        except:
            return url
    return url

def generate_monthly_ranges(start_str, end_str):
    start_date = datetime.strptime(start_str, "%m/%d/%Y")
    end_date = datetime.strptime(end_str, "%m/%d/%Y")
    ranges = []
    current = start_date
    while current <= end_date:
        next_month = current.replace(day=1) + relativedelta(months=1)
        month_end = next_month - timedelta(days=1)
        if month_end > end_date: month_end = end_date
        ranges.append({
            'start': current.strftime("%m/%d/%Y"),
            'end': month_end.strftime("%m/%d/%Y"),
            'label': current.strftime("%B %Y")
        })
        current = next_month
    return ranges

def check_and_wait_captcha(driver):
    try:
        page_source = driver.page_source.lower()
        if "unusual traffic" in page_source or "recaptcha" in page_source or "bukan robot" in page_source:
            print("\n🛑 🛑 CAPTCHA TERDETEKSI! SCRIPT DI-PAUSE 🛑 🛑")
            alert_user_captcha()
            input("👉 Jika sudah centang hijau / aman, TEKAN ENTER DI SINI untuk lanjut...")
            print("✅ Melanjutkan scraping...")
            return True
    except:
        pass
    return False

def main():
    print("=============================================")
    print("   🤖 GOOGLE NEWS SCRAPER (PERIOD PRIORITY)   ")
    print("   Logika: Selesaikan 1 Bulan utk Semua KW   ")
    print("=============================================")
    
    driver = setup_driver()
    collected_data = []
    unique_links = set()
    completed_tasks = set()

    # --- LOAD DATA LAMA ---
    if os.path.exists(OUTPUT_FILE):
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            unique_links = set(existing['Link'].tolist())
            collected_data = existing.to_dict('records')
            
            # Logic Resume: Mencatat pasangan Keyword|Periode yang sudah ada
            if 'Keyword' in existing.columns and 'Periode_Scrape' in existing.columns:
                task_history = list(zip(existing['Keyword'], existing['Periode_Scrape']))
                seen = set()
                ordered_tasks = []
                for x in task_history:
                    if x not in seen:
                        ordered_tasks.append(x)
                        seen.add(x)
                
                if ordered_tasks:
                    last_task = ordered_tasks[-1]
                    print(f"⚠️ Tugas terakhir dianggap belum tuntas: {last_task[0]} | {last_task[1]}")
                    # Masukkan semua KECUALI yang terakhir ke daftar selesai
                    for t in ordered_tasks[:-1]: 
                        completed_tasks.add(f"{t[0]}|{t[1]}")
                
            print(f"📂 Resume: {len(collected_data)} data link sudah ada.")
        except Exception as e:
            print(f"⚠️ Warning Load File: {e}")

    date_ranges = generate_monthly_ranges(START_DATE_GLOBAL, END_DATE_GLOBAL)
    print(f"📅 Target Waktu: {len(date_ranges)} Periode Bulanan.")

    try:
        # --- PERUBAHAN LOGIKA LOOPING DI SINI ---
        # Loop 1: Periode (Bulan) dulu
        for periode in date_ranges:
            print(f"\n📆 MASUK PERIODE: {periode['label']} ({periode['start']} - {periode['end']})")
            print("------------------------------------------------------")

            # Loop 2: Kategori PESTLE
            for category, keywords in PESTLE_KEYWORDS.items():
                
                # Loop 3: Keyword
                for keyword in keywords:
                    current_task_id = f"{keyword}|{periode['label']}"
                    
                    # Cek Resume (Apakah Keyword X di Bulan Y sudah selesai?)
                    if current_task_id in completed_tasks:
                        # Kita print short log saja biar tidak spamming
                        # print(f"   ⏩ SKIP: {keyword} (Sudah ada)")
                        continue 

                    print(f"🔎 Scrape: [{category}] '{keyword}'")
                    
                    query = keyword.replace(' ', '+')
                    url = (f"https://www.google.com/search?q={query}&tbm=nws&hl=id&gl=ID"
                           f"&tbs=cdr:1,cd_min:{periode['start']},cd_max:{periode['end']}")
                    
                    driver.get(url)
                    check_and_wait_captcha(driver)
                    
                    found_in_month = 0
                    
                    for page in range(1, JUMLAH_HALAMAN_PER_BULAN + 1):
                        try:
                            cards = driver.find_elements(By.CSS_SELECTOR, "div.SoaBEf")
                            
                            if not cards:
                                is_captcha = check_and_wait_captcha(driver)
                                if is_captcha:
                                    cards = driver.find_elements(By.CSS_SELECTOR, "div.SoaBEf")
                                if not cards:
                                    break

                            page_found = len(cards)
                            page_saved = 0
                            page_skipped = 0
                            
                            for card in cards:
                                try:
                                    link_elem = card.find_element(By.TAG_NAME, "a")
                                    real_link = clean_google_link(link_elem.get_attribute("href"))
                                    
                                    # Cek Duplikat
                                    if real_link in unique_links: 
                                        page_skipped += 1
                                        continue
                                    
                                    title = card.find_element(By.CSS_SELECTOR, "div.n0jPhd").text
                                    source = card.find_element(By.CSS_SELECTOR, "div.MgUUmf").text
                                    date_txt = card.find_element(By.CSS_SELECTOR, "div.OSrXXb").text
                                    
                                    unique_links.add(real_link)
                                    collected_data.append({
                                        "Kategori": category,
                                        "Keyword": keyword,
                                        "Periode_Scrape": periode['label'],
                                        "Sumber": source,
                                        "Tanggal_Tayang": date_txt,
                                        "Judul": title,
                                        "Link": real_link
                                    })
                                    page_saved += 1
                                    found_in_month += 1
                                except: 
                                    continue
                            
                            print(f"      -> Hal {page}: Ketemu {page_found} | Baru +{page_saved} | Skip {page_skipped}")

                            # Save otomatis tiap halaman dapat data
                            if page_saved > 0:
                                pd.DataFrame(collected_data).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

                            try:
                                next_btn = driver.find_element(By.ID, "pnnext")
                                next_btn.click()
                                time.sleep(random.uniform(2, 4))
                                check_and_wait_captcha(driver)
                            except:
                                break 
                        except:
                            break
                    
                    # Tandai task selesai setelah semua halaman keyword tsb beres
                    completed_tasks.add(current_task_id)
                    time.sleep(random.uniform(1.5, 3))
                
                # Jeda antar kategori dalam bulan yang sama (opsional)
                # time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stop Manual.")
    finally:
        driver.quit()
        if collected_data:
            pd.DataFrame(collected_data).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print("🎉 Selesai Total.")

if __name__ == "__main__":
    main()