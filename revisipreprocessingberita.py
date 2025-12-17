import pandas as pd
import re
import nltk
import string
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import swifter  # Untuk mempercepat pandas apply
from datetime import datetime, timedelta  # [BARU] Untuk hitung tanggal relatif

# Download resource NLTK jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- KONFIGURASI ---
INPUT_FILE = "pestle_konten.csv"
OUTPUT_FILE = "preprocessing_berita_revisi.csv" 
VISUALIZATION_DIR = "preprocessing_berita_visualisasi_revisi"

class NewsPreprocessor:
    def __init__(self):
        print("[-] Menginisialisasi Cleaner (Mode: Light untuk BERTopic)...")
        
        # --- DAFTAR NOISE WORDS (STOPWORDS) ---
        # Sesuai permintaan: TIDAK DITAMBAH & TIDAK DIKURANG DARI VERSI AWAL
        self.noise_words = {
            'baca', 'juga', 'halaman', 'editor', 'penulis', 'sumber', 'foto', 'wartawan',
            'reporter', 'jakarta', 'liputan6', 'kompas', 'tribun', 'detik', 'antara', 
            'copyright', 'all', 'rights', 'reserved', 'news', 'com', 'co', 'id', 'tempo',
            'bbc', 'kontan', 'website', 'video', 'selanjutnya', 'berikut', 'redaksi', 
            'jurnalis', 'dok', 'istimewa', 'liputan', 'laman', 'klik', 'tautan', 'ini',
            'dilansir', 'dikutip', 'simak', 'selengkapnya', 'bergabung', 'whatsapp',
            'channel', 'saluran', 'cnn', 'cnbc', 'republika', 'link', 'langsung',
            'senin', 'selasa', 'rabu', 'kamis', 'jumat', 'sabtu', 'minggu'
        }

    def clean_text_basic(self, text):
        text = str(text)
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub(r'[^\x00-\x7f]', r'', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def normalize_elongation(self, text):
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def process_row(self, text):
        text = self.clean_text_basic(text)
        text = self.normalize_elongation(text)
        if len(text) == 0: return ""
        tokens = word_tokenize(text)
        # Filter tokens berdasarkan noise_words
        clean_tokens = [t for t in tokens if t not in self.noise_words]
        return " ".join(clean_tokens)

# --- FUNGSI PARSING TANGGAL INDONESIA & RELATIF ---
def parse_indonesian_date(date_series):
    """
    Mengubah string tanggal Indonesia ke datetime.
    Mendukung format:
    1. Absolut: "Senin, 20 Februari 2024"
    2. Relatif: "2 hari lalu", "Kemarin", "Baru saja"
    """
    # Mapping bulan Indonesia ke Angka
    month_map = {
        'januari': '01', 'februari': '02', 'maret': '03', 'april': '04',
        'mei': '05', 'juni': '06', 'juli': '07', 'agustus': '08',
        'september': '09', 'oktober': '10', 'november': '11', 'desember': '12',
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'jun': '06',
        'jul': '07', 'agu': '08', 'sep': '09', 'okt': '10', 'nov': '11', 'des': '12',
        'agust': '08'
    }
    
    # --- [UPDATE PENTING] SET TANGGAL SCRAPING ---
    # Jika scraping dilakukan 30 Nov 2025, set ini agar "Kemarin" terbaca sbg 29 Nov 2025
    # Jangan gunakan datetime.now() jika waktu preprocessing berbeda dengan waktu scraping
    SCRAPE_DATE = datetime(2025, 11, 30) # <-- Ubah tanggal ini sesuai waktu scraping Anda
    current_time = SCRAPE_DATE

    def clean_date_str(x):
        x_str = str(x).lower().strip()
        
        # --- [BARU] LOGIKA TANGGAL RELATIF ---
        
        # 1. "Baru saja", "x jam lalu" -> Anggap Hari Ini (Sesuai SCRAPE_DATE)
        if any(k in x_str for k in ['baru saja', 'menit lalu', 'jam lalu', 'detik lalu', 'beberapa saat']):
            return current_time.strftime("%d/%m/%Y")
            
        # 2. "Kemarin" -> SCRAPE_DATE - 1 Hari
        if 'kemarin' in x_str:
            return (current_time - timedelta(days=1)).strftime("%d/%m/%Y")
            
        # 3. "x hari lalu"
        days_match = re.search(r'(\d+)\s+hari\s+lalu', x_str)
        if days_match:
            d = int(days_match.group(1))
            return (current_time - timedelta(days=d)).strftime("%d/%m/%Y")
            
        # 4. "x minggu lalu"
        weeks_match = re.search(r'(\d+)\s+minggu\s+lalu', x_str)
        if weeks_match:
            w = int(weeks_match.group(1))
            return (current_time - timedelta(weeks=w)).strftime("%d/%m/%Y")
            
        # 5. "x bulan lalu" (Estimasi kasar 30 hari)
        months_match = re.search(r'(\d+)\s+bulan\s+lalu', x_str)
        if months_match:
            m = int(months_match.group(1))
            return (current_time - timedelta(days=m*30)).strftime("%d/%m/%Y")

        # --- LOGIKA TANGGAL STANDAR ---
        x_str = re.sub(r'(senin|selasa|rabu|kamis|jumat|sabtu|minggu)\s*,?', '', x_str).strip()
        for ind, eng in month_map.items():
            # Pakai spasi agar tidak mereplace bagian kata lain (misal: 'jan' di 'janjian')
            x_str = x_str.replace(f" {ind} ", f"/{eng}/")
            x_str = x_str.replace(f" {ind}", f"/{eng}/") 
        
        x_str = re.sub(r'\s*\d{2}:\d{2}.*', '', x_str) # Hapus jam
        x_str = x_str.replace(" ", "") 
        return x_str

    cleaned_series = date_series.apply(clean_date_str)
    return pd.to_datetime(cleaned_series, dayfirst=True, errors='coerce')

# --- FUNGSI VISUALISASI TREN (DEBUGGING MODE) ---
def plot_trend_distribution(df, output_dir, title_prefix="Raw Data"):
    """Visualisasi Tren Berita dengan Debugging Tanggal"""
    print(f"[-] Membuat Grafik Tren untuk: {title_prefix}...")
    
    date_col = None
    possible_cols = ['Tanggal_Tayang', 'tanggal', 'date', 'waktu', 'Date', 'Time']
    for col in possible_cols:
        if col in df.columns:
            date_col = col
            break
    
    if not date_col:
        print("[!] Kolom tanggal tidak ditemukan.")
        return

    # Parsing Tanggal (Sekarang support Relative Date)
    df['parsed_date'] = parse_indonesian_date(df[date_col])
    
    # Debugging Info
    total_rows = len(df)
    valid_dates = df.dropna(subset=['parsed_date'])
    failed_dates = df[df['parsed_date'].isna()][date_col].unique()[:5]
    
    print(f"    > Total Baris: {total_rows}")
    print(f"    > Sukses Parsing: {len(valid_dates)}")
    print(f"    > Gagal Parsing: {total_rows - len(valid_dates)} (Data ini tidak masuk grafik)")
    if len(failed_dates) > 0:
        print(f"    > Contoh format tanggal yang gagal: {failed_dates}")

    if len(valid_dates) == 0:
        print("[!] Tidak ada tanggal valid untuk di-plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    # Hitung per bulan & Sort
    monthly_counts = valid_dates.groupby(valid_dates['parsed_date'].dt.to_period('M')).size()

    if not monthly_counts.empty:
        monthly_counts = monthly_counts.sort_index()
        
        x_dates = monthly_counts.index.astype(str)
        y_values = monthly_counts.values

        plt.figure(figsize=(12, 6))
        # Menggunakan warna merah untuk Raw Data
        color = '#e74c3c' if "Raw" in title_prefix else '#2980b9'
        sns.lineplot(x=x_dates, y=y_values, marker='o', linewidth=2.5, color=color)
        
        for x, y in zip(x_dates, y_values):
            plt.text(x, y + (max(y_values)*0.02), f'{y}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.title(f"Tren Volume Berita ({title_prefix})", fontsize=16, pad=15)
        plt.xlabel("Bulan", fontsize=12)
        plt.ylabel("Total Berita", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        filename = f"0_tren_berita_{title_prefix.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"[-] Grafik disimpan: {filename}")

# --- FUNGSI ANALISIS STATISTIK ---
def analyze_corpus_stats(text_series, label="Data"):
    print(f"[-] Menghitung statistik untuk: {label}...")
    doc_lengths = text_series.apply(lambda x: len(str(x).split()))
    sample_text = text_series if len(text_series) < 5000 else text_series.sample(5000)
    all_words = []
    for text in tqdm(sample_text, desc=f"Counting Words ({label})"):
        all_words.extend(str(text).split())
    word_counts = Counter(all_words)
    return {
        "Total Dokumen": len(text_series),
        "Total Kata (Est)": sum(doc_lengths),
        "Rata-rata Kata/Dokumen": doc_lengths.mean(),
        "Vocabulary": len(word_counts),
        "Doc Lengths": doc_lengths, 
        "Word Counts": word_counts
    }

def plot_comparison(stats_before, stats_after, data_flow_stats, output_dir):
    print("[-] Membuat Visualisasi Perbandingan Lengkap...")
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    # 1. PIPELINE
    plt.figure(figsize=(10, 6))
    stages = ['Raw Data', 'Final/Clean Data']
    values = [data_flow_stats['raw'], stats_after['Total Dokumen']]
    colors = ['#95a5a6', '#2ecc71']
    ax = sns.barplot(x=stages, y=values, palette=colors)
    plt.title("Data Reduction Pipeline", fontsize=16, pad=20)
    plt.ylabel("Number of Documents", fontsize=12)
    for i, v in enumerate(values):
        ax.text(i, v + (max(values)*0.02), f"{v:,}", ha='center', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_pipeline_data_reduction.png"), dpi=300)
    plt.close()

    # 2. HISTOGRAM
    plt.figure(figsize=(10, 6))
    sns.histplot(stats_before["Doc Lengths"], color="skyblue", label="Sebelum NLP", kde=True, element="step", alpha=0.5)
    sns.histplot(stats_after["Doc Lengths"], color="orange", label="Sesudah NLP", kde=True, element="step", alpha=0.5)
    plt.title("Perubahan Distribusi Panjang Artikel")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "2_distribusi_panjang_kata.png"), dpi=300)
    plt.close()
    
    # 3. TOP WORDS
    top_n = 15
    top_before = stats_before["Word Counts"].most_common(top_n)
    top_after = stats_after["Word Counts"].most_common(top_n)
    if top_before and top_after:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        wb, cb = zip(*top_before)
        sns.barplot(x=list(cb), y=list(wb), ax=axes[0], palette="Reds_d")
        axes[0].set_title(f"Top {top_n} Kata SEBELUM Cleaning", fontsize=12)
        wa, ca = zip(*top_after)
        sns.barplot(x=list(ca), y=list(wa), ax=axes[1], palette="Greens_d")
        axes[1].set_title(f"Top {top_n} Kata SESUDAH Cleaning", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "3_top_words_comparison.png"), dpi=300)
        plt.close()

def save_preprocessing_report(data_flow_stats, stats_before, stats_after, sample_comparisons, output_dir):
    report_file = os.path.join(output_dir, "RANGKUMAN_METODOLOGI_PREPROCESSING.txt")
    with open(report_file, "w", encoding='utf-8') as f:
        # Header
        f.write("="*70 + "\n")
        f.write("       LAPORAN & METODOLOGI PREPROCESSING DATA BERITA\n")
        f.write("="*70 + "\n\n")

        # [1] ALUR PROSES
        f.write("[1] ALUR PROSES (PIPELINE)\n")
        f.write("    Proses preprocessing dilakukan dengan tahapan berikut:\n")
        f.write("    1. Filtering Status  : Menghapus data yang gagal di-scrape ('Status_Scrape' != 'Sukses').\n")
        f.write("    2. Deduplikasi       : Menghapus data ganda berdasarkan isi konten yang persis sama.\n")
        f.write("    3. Lowercasing       : Mengubah semua huruf menjadi kecil.\n")
        f.write("    4. Regex Cleaning    : Menghapus URL, Email, HTML Tags, dan Angka.\n")
        f.write("    5. Punctuation Removal: Menghapus tanda baca secara total (diganti spasi).\n")
        f.write("                           Tujuan: Mencegah error pada perhitungan metrik NPMI.\n")
        f.write("    6. Noise Filtering   : Menghapus kata navigasi web spesifik (contoh: 'baca juga', 'halaman').\n")
        f.write("    7. Elongation Fix    : Normalisasi kata berulang (contoh: 'hallooo' -> 'halo').\n\n")

        # [2] JUSTIFIKASI
        f.write("[2] JUSTIFIKASI METODOLOGI (MENGAPA LIGHT CLEANING?)\n")
        f.write("    - Mengapa Stopwords (yang, dan, di, tidak) DIPERTAHANKAN?\n")
        f.write("      Model Topic Modeling yang digunakan (BERTopic) berbasis Transformer/BERT.\n")
        f.write("      Model BERT membutuhkan struktur kalimat yang utuh (termasuk kata sambung)\n")
        f.write("      untuk memahami konteks semantik (seperti negasi atau hubungan waktu).\n")
        f.write("      Penghapusan stopwords dilakukan terpisah di tahap Vectorizer (c-TF-IDF).\n\n")

        # [3] PERBANDINGAN STATISTIK DATA
        f.write("[3] PERBANDINGAN STATISTIK DATA\n")
        f.write(f"    {'METRIK':<25} | {'SEBELUM (RAW)':<15} | {'SESUDAH (CLEAN)':<15}\n")
        f.write("-" * 65 + "\n")
        f.write(f"    {'Total Dokumen':<25} | {stats_before['Total Dokumen']:<15} | {stats_after['Total Dokumen']:<15}\n")
        f.write(f"    {'Total Kata (Est)':<25} | {stats_before['Total Kata (Est)']:<15} | {stats_after['Total Kata (Est)']:<15}\n")
        f.write(f"    {'Rata-rata Kata/Dok':<25} | {stats_before['Rata-rata Kata/Dokumen']:<15.2f} | {stats_after['Rata-rata Kata/Dokumen']:<15.2f}\n")
        f.write(f"    {'Ukuran Vocabulary':<25} | {stats_before['Vocabulary']:<15} | {stats_after['Vocabulary']:<15}\n\n")

        # [4] SAMPEL DATA
        f.write("[4] SAMPEL DATA (BEFORE vs AFTER)\n")
        f.write("    Berikut adalah contoh perubahan teks pada 5 dokumen acak:\n")
        f.write("-" * 70 + "\n")
        for i, (raw, clean) in enumerate(sample_comparisons, 1):
            f.write(f"    Contoh #{i}:\n")
            # Cleaning newline characters for cleaner display in txt
            raw_clean = str(raw).replace('\n', ' ')
            clean_clean = str(clean).replace('\n', ' ')
            # Truncate to avoid extremely long lines
            f.write(f"    [RAW]   : {raw_clean[:200]}...\n")
            f.write(f"    [CLEAN] : {clean_clean[:200]}...\n")
            f.write("-" * 70 + "\n")
        f.write("\n")

        # [5] INSIGHT & VALIDASI
        f.write("[5] INSIGHT & VALIDASI\n")
        f.write("    - Deduplikasi dilakukan untuk menghindari bias topik pada berita viral.\n")
        f.write("    - Teks bersih siap digunakan untuk embedding IndoBERT/IndoGovBERT.\n")

def main():
    print("==========================================")
    print(" 🧹 PREPROCESSING BERITA (FULL SCRIPT)    ")
    print("==========================================")
    
    if not os.path.exists(INPUT_FILE):
        print(f"[!] File {INPUT_FILE} tidak ditemukan.")
        return
    
    # 1. LOAD RAW DATA
    print("[-] Membaca file CSV...")
    df_raw = pd.read_csv(INPUT_FILE)
    count_raw = len(df_raw)
    
    # --- VISUALISASI RAW DATA SEBELUM FILTER ---
    plot_trend_distribution(df_raw, VISUALIZATION_DIR, title_prefix="Raw Data Sebelum Filter")

    # 2. Filter Status Scrape
    df = df_raw.copy()
    if 'Status_Scrape' in df_raw.columns:
        df = df[df['Status_Scrape'] == 'Sukses']
    count_scraped = len(df)
    
    # 3. Validasi Konten
    target_col = 'Isi_Berita' if 'Isi_Berita' in df.columns else 'content'
    df = df.dropna(subset=[target_col])
    
    # 4. Deduplikasi
    df.drop_duplicates(subset=[target_col], inplace=True)
    count_deduped = len(df)
    
    data_flow_stats = {
        'raw': count_raw, 
        'scraped': count_scraped, 
        'deduped': count_deduped
    }
    print(f"[-] Data Unik Valid: {count_deduped}")

    # 5. Analisis Awal (Before)
    stats_before = analyze_corpus_stats(df[target_col], label="BEFORE")

    # 6. Cleaning NLP
    processor = NewsPreprocessor()
    tqdm.pandas(desc="Processing NLP")
    df['processed_text'] = df[target_col].swifter.apply(processor.process_row)
    df = df[df['processed_text'].str.strip().astype(bool)]
    
    # 7. Analisis Akhir (After)
    stats_after = analyze_corpus_stats(df['processed_text'], label="AFTER")
    
    # 8. Visualisasi Pipeline & Words
    plot_comparison(stats_before, stats_after, data_flow_stats, VISUALIZATION_DIR)
    
    # 9. Laporan
    sample_indices = df.sample(min(5, len(df))).index
    sample_comparisons = list(zip(df.loc[sample_indices, target_col], df.loc[sample_indices, 'processed_text']))
    save_preprocessing_report(data_flow_stats, stats_before, stats_after, sample_comparisons, VISUALIZATION_DIR)
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[+] Selesai. Output: {OUTPUT_FILE}")
    print(f"[+] Cek folder '{VISUALIZATION_DIR}' untuk grafik Raw Data & hasil laporan .txt.")

if __name__ == "__main__":
    main()