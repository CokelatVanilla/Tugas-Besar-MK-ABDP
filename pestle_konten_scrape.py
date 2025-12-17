import pandas as pd
from newspaper import Article
from tqdm import tqdm
import os
import time
import random

# --- KONFIGURASI ---
INPUT_FILE = "pestle_link.csv"  # File CSV hasil tahap sebelumnya
OUTPUT_FILE = "pestle_konten.csv" # File output yang ada isi beritanya

def download_article_content(url):
    """
    Fungsi 'Magic' untuk menyedot isi berita tanpa pusing mikirin HTML
    """
    try:
        article = Article(url, language='id')
        article.download()
        article.parse()
        
        # Kita ambil teksnya, dan ganti baris baru dengan spasi biar rapi di CSV
        text = article.text.replace('\n', ' ').replace('\r', ' ')
        
        # Validasi: Kalau teks terlalu pendek (mungkin gagal/cuma loading), anggap gagal
        if len(text) < 50: 
            return "ERROR: Konten terlalu pendek/Gagal Parsing"
            
        return text
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    print("=============================================")
    print("   📰 NEWS CONTENT DOWNLOADER (NEWSPAPER3K)  ")
    print("=============================================")

    # 1. Cek apakah kita melanjutkan pekerjaan (Resume) atau mulai baru
    if os.path.exists(OUTPUT_FILE):
        print(f"📂 File '{OUTPUT_FILE}' ditemukan. Melanjutkan scraping konten...")
        df = pd.read_csv(OUTPUT_FILE)
    else:
        if os.path.exists(INPUT_FILE):
            print(f"📂 Membaca file sumber: '{INPUT_FILE}'...")
            df = pd.read_csv(INPUT_FILE)
            # Bikin kolom baru kosong
            df['Isi_Berita'] = ""
            df['Status_Scrape'] = "" 
        else:
            print(f"❌ File input '{INPUT_FILE}' tidak ditemukan!")
            return

    # 2. Filter mana yang belum di-download (Isi_Berita masih kosong/NaN)
    #    Kita anggap yang belum selesai adalah yang 'Isi_Berita' nya kosong atau NaN
    total_data = len(df)
    
    # Pastikan kolom Isi_Berita formatnya string dulu biar gak error
    df['Isi_Berita'] = df['Isi_Berita'].fillna("").astype(str)
    
    # Cari index mana saja yang masih kosong isinya
    indices_to_scrape = df[df['Isi_Berita'] == ""].index.tolist()
    
    print(f"📊 Total Link: {total_data}")
    print(f"⏳ Sisa Link yang perlu diambil kontennya: {len(indices_to_scrape)}")
    print("🚀 Memulai proses download konten...\n")

    # 3. Loop dengan Progress Bar (tqdm)
    counter = 0
    
    for index in tqdm(indices_to_scrape, desc="Downloading"):
        link = df.at[index, 'Link']
        
        # Ambil konten
        content = download_article_content(link)
        
        # Simpan ke DataFrame di memori
        df.at[index, 'Isi_Berita'] = content
        df.at[index, 'Status_Scrape'] = "Sukses" if "ERROR" not in content else "Gagal"
        
        counter += 1
        
        # --- AUTO SAVE setiap 10 artikel ---
        if counter % 10 == 0:
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
            # Jeda dikit biar server berita gak ngira kita serangan DDOS
            time.sleep(random.uniform(0.5, 1.5))

    # 4. Final Save
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ SELESAI! Data lengkap tersimpan di: {OUTPUT_FILE}")
    
    # Statistik Singkat
    sukses = len(df[df['Status_Scrape'] == 'Sukses'])
    gagal = len(df[df['Status_Scrape'] == 'Gagal'])
    print(f"📈 Statistik: Sukses: {sukses} | Gagal: {gagal}")

if __name__ == "__main__":
    main()