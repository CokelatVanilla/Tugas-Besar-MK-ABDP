import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import ast
from tqdm import tqdm

# Library NLTK untuk Stopwords
import nltk
from nltk.corpus import stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Library LDA Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaMulticore
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Setup Logging
import logging
logging.getLogger("gensim").setLevel(logging.ERROR)

# --- KONFIGURASI ---
INPUT_FILE = "preprocessing_berita_revisi.csv"
OUTPUT_DIR = "Result_LDA_Analysis_revisi"
SEED = 42

# [PILIHAN BIGRAM]
USE_BIGRAMS = False # Set True untuk mengaktifkan Frasa, False untuk kata tunggal

# Range jumlah topik yang akan dites (Hyperparameter Tuning)
TOPIC_RANGE = [5, 10, 15, 20, 25, 30]

# Konfigurasi Training LDA
LDA_PARAMS = {
    'chunksize': 2000,
    'passes': 20,
    'per_word_topics': True,
    'alpha': 'symmetric',
    'eta': None
}

def prepare_data(filepath):
    """
    Memuat data dan melakukan Heavy Cleaning (Stopword Removal) khusus untuk LDA.
    """
    print(f"[-] Memuat data: {filepath}")
    if not os.path.exists(filepath):
        print(f"[!] File {filepath} tidak ditemukan.")
        return None

    df = pd.read_csv(filepath)
    data_tokens = []
    
    # 1. Setup Stopwords Bahasa Indonesia
    print("[-] Menginisialisasi Stopwords (NLTK + Custom)...")
    stop_words = set(stopwords.words('indonesian'))
    
    # Tambahan stopwords spesifik berita/jurnalistik yang sering jadi noise di LDA
    custom_noise = {
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "dengan", 
        "adalah", "yaitu", "tersebut", "juga", "sudah", "telah", "akan", "sedang", 
        "tapi", "tetapi", "melalui", "karena", "oleh", "sebagai", "bisa", "dapat",
        "seperti", "dalam", "antara", "bagi", "kepada", "agar", "supaya", "atau",
        "saya", "kita", "kami", "anda", "mereka", "dia", "ia", "beliau", "rp", "ndak",
        "mengatakan", "kata", "ujar", "tutur", "jelas", "ungkap", "sebut", "menurut",
        "menjadi", "melakukan", "memberikan", "mengambil", "memiliki", "ada", "tidak",
        "banyak", "sedikit", "besar", "kecil", "baru", "lama", "tinggi", "rendah", 
        "sangat", "lebih", "paling", "kurang", "cukup", "sendiri", "lain", "tulis"
    }
    stop_words.update(custom_noise)

    print("[-] Memproses token & Menghapus Stopwords...")
    
    # Deteksi kolom target
    if 'processed_text' in df.columns:
        col_target = 'processed_text'
    elif 'clean_tokens' in df.columns:
        col_target = 'clean_tokens'
    else:
        col_target = df.columns[0] # Fallback

    for item in df[col_target]:
        try:
            # Parsing jika formatnya list string "['a', 'b']"
            tokens = ast.literal_eval(item) if (isinstance(item, str) and "[" in item) else item
            
            # Normalisasi jadi list
            if not isinstance(tokens, list):
                # Tokenize sederhana jika masih string kalimat
                tokens = str(item).split()
            
            # --- FILTER STOPWORDS (CRITICAL FOR LDA) ---
            # Hanya ambil kata yang:
            # 1. Tidak ada di list stopwords
            # 2. Panjang karakter > 2 (membuang "di", "ke", "yg")
            filtered_tokens = [w for w in tokens if w.lower() not in stop_words and len(w) > 2]
            
            data_tokens.append(filtered_tokens)
        except:
            continue

    # Hapus dokumen kosong/pendek setelah stopword removal
    valid_tokens = [d for d in data_tokens if len(d) > 3]
    print(f"[-] Total Dokumen Valid untuk LDA: {len(valid_tokens)} (dari {len(df)})")
    
    return valid_tokens

def create_dictionary_corpus(data_tokens):
    # --- LOGIKA BIGRAM TOGGLE ---
    if USE_BIGRAMS:
        print("[-] Mode Bigram: AKTIF. Membangun Frasa...")
        bigram = gensim.models.Phrases(data_tokens, min_count=5, threshold=50) 
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        # Update data_tokens dengan Bigram
        data_tokens = [bigram_mod[doc] for doc in data_tokens]
    else:
        print("[-] Mode Bigram: NON-AKTIF. Menggunakan Unigram...")
    # ----------------------------------------

    # 1. Buat Dictionary
    id2word = corpora.Dictionary(data_tokens)
    
    # 2. Filter kata ekstrem
    id2word.filter_extremes(no_below=5, no_above=0.5)
    
    # 3. Buat Corpus
    corpus = [id2word.doc2bow(text) for text in data_tokens]
    
    return id2word, corpus, data_tokens

def calculate_metrics(lda_model, corpus, id2word, data_tokens):
    """
    Menghitung 3 Metrik Utama: NPMI, Cv, dan Diversity
    """
    # 1. Coherence NPMI (Evaluasi Semantik Ketat)
    cm_npmi = CoherenceModel(model=lda_model, texts=data_tokens, dictionary=id2word, coherence='c_npmi')
    npmi = cm_npmi.get_coherence()
    
    # 2. Coherence Cv (Evaluasi Standar)
    cm_cv = CoherenceModel(model=lda_model, texts=data_tokens, dictionary=id2word, coherence='c_v')
    cv = cm_cv.get_coherence()
    
    # 3. Topic Diversity (Uniqueness)
    topics = lda_model.show_topics(num_topics=-1, num_words=20, formatted=False)
    unique_words = set()
    total_words = 0
    
    for _, word_list in topics:
        top_n = [w[0] for w in word_list[:10]] 
        unique_words.update(top_n)
        total_words += 10
        
    diversity = len(unique_words) / total_words if total_words > 0 else 0
    
    return npmi, cv, diversity

def train_and_evaluate(data_tokens, id2word, corpus):
    results = []
    models_store = {}
    
    print(f"\n{'='*40}")
    print(f"  MULAI TUNING JUMLAH TOPIK (K)")
    print(f"{'='*40}")
    
    for k in tqdm(TOPIC_RANGE, desc="Training LDA Models"):
        # Train LDA
        lda_model = LdaMulticore(corpus=corpus,
                                 id2word=id2word,
                                 num_topics=k, 
                                 random_state=SEED,
                                 chunksize=LDA_PARAMS['chunksize'],
                                 passes=LDA_PARAMS['passes'],
                                 per_word_topics=LDA_PARAMS['per_word_topics'],
                                 workers=3)
        
        # Hitung Metrik
        npmi, cv, div = calculate_metrics(lda_model, corpus, id2word, data_tokens)
        
        results.append({
            'Num_Topics': k,
            'NPMI': npmi,
            'Cv': cv,
            'Diversity': div
        })
        models_store[k] = lda_model
        
    return pd.DataFrame(results), models_store

def plot_tuning_result(df_results, output_dir):
    """
    [MODIFIKASI] Membuat 4 Grafik: NPMI, Cv, Diversity, dan Combined.
    """
    print("[-] Membuat Grafik Evaluasi Tuning (Individu & Gabungan)...")
    
    # Gunakan style seaborn
    sns.set_style("whitegrid")
    
    # 1. Grafik Coherence NPMI
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['Num_Topics'], df_results['NPMI'], marker='o', color='#27ae60', linewidth=2.5)
    plt.title("Evaluasi Metrik 1: Coherence NPMI (Semantik)", fontsize=14, pad=15)
    plt.xlabel("Jumlah Topik (K)", fontsize=12)
    plt.ylabel("NPMI Score (Higher is Better)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grafik_evaluasi_1_NPMI.png"), dpi=300)
    plt.close()

    # 2. Grafik Coherence Cv
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['Num_Topics'], df_results['Cv'], marker='s', color='#2980b9', linewidth=2.5)
    plt.title("Evaluasi Metrik 2: Coherence Cv (Standar)", fontsize=14, pad=15)
    plt.xlabel("Jumlah Topik (K)", fontsize=12)
    plt.ylabel("Cv Score (Higher is Better)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grafik_evaluasi_2_Cv.png"), dpi=300)
    plt.close()

    # 3. Grafik Topic Diversity
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['Num_Topics'], df_results['Diversity'], marker='^', color='#e74c3c', linewidth=2.5)
    plt.title("Evaluasi Metrik 3: Topic Diversity (Keunikan)", fontsize=14, pad=15)
    plt.xlabel("Jumlah Topik (K)", fontsize=12)
    plt.ylabel("Diversity Score (0-1)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grafik_evaluasi_3_Diversity.png"), dpi=300)
    plt.close()

    # 4. Grafik Gabungan (Combined)
    plt.figure(figsize=(12, 7))
    plt.plot(df_results['Num_Topics'], df_results['NPMI'], marker='o', label='NPMI (Semantik)', color='#27ae60', linewidth=2)
    plt.plot(df_results['Num_Topics'], df_results['Cv'], marker='s', label='Cv (Standar)', color='#2980b9', linewidth=2)
    plt.plot(df_results['Num_Topics'], df_results['Diversity'], marker='^', label='Diversity (Keunikan)', color='#e74c3c', linewidth=2)
    
    plt.title("Komparasi Gabungan Semua Metrik Evaluasi LDA", fontsize=16, pad=20)
    plt.xlabel("Jumlah Topik (K)", fontsize=12)
    plt.ylabel("Normalized Score Value", fontsize=12)
    plt.legend(loc='best', fontsize=11, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grafik_evaluasi_4_GABUNGAN.png"), dpi=300)
    plt.close()

def generate_visualizations(lda_model, corpus, id2word, best_k, output_dir):
    img_dir = os.path.join(output_dir, "plots_visualization")
    os.makedirs(img_dir, exist_ok=True)
    
    # A. Bar Chart Bobot Kata per Topik
    # [PERBAIKAN] Menggunakan best_k agar menampilkan SEMUA topik yang terbentuk
    topics = lda_model.show_topics(num_topics=best_k, num_words=10, formatted=False)
    
    cols = 2
    rows = int(np.ceil(len(topics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), sharex=False)
    axes = axes.flatten()

    for i, (topic_id, word_list) in enumerate(topics):
        words = [w[0] for w in word_list]
        scores = [w[1] for w in word_list]
        
        axes[i].barh(words, scores, color='#3498db')
        axes[i].set_title(f'Topik #{topic_id}', fontsize=12)
        axes[i].invert_yaxis()
        
    for j in range(i+1, len(axes)): axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "1_barchart_topic_keywords.png"), dpi=300)
    plt.close()

    # B. WordClouds
    print("    [-] Membuat WordCloud LDA...")
    # [PERBAIKAN] Menggunakan best_k agar menampilkan SEMUA topik
    topics_wc = lda_model.show_topics(num_topics=best_k, num_words=30, formatted=False)
    
    # [PERBAIKAN] Hitung baris dinamis berdasarkan jumlah topik
    cols = 3
    rows = int(np.ceil(len(topics_wc) / cols))
    
    # Sesuaikan tinggi figure dengan jumlah baris
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()
    
    for i, (topic_id, word_list) in enumerate(topics_wc):
        word_freq = {w[0]: w[1] for w in word_list}
        wc = WordCloud(width=600, height=400, background_color='white', colormap='magma').generate_from_frequencies(word_freq)
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].axis("off")
        axes[i].set_title(f"LDA Topik {topic_id}", fontsize=14)
        
    # Matikan axis untuk plot kosong sisa grid
    for j in range(i+1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "2_wordclouds_lda.png"), dpi=300)
    plt.close()

    # C. Interactive pyLDAvis
    try:
        print("    [-] Membuat HTML Interaktif (pyLDAvis)...")
        vis_data = gensimvis.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(vis_data, os.path.join(img_dir, "lda_interactive_map.html"))
    except Exception as e:
        print(f"    [!] Gagal membuat pyLDAvis: {e}")

def save_detailed_report(best_model, best_row, df_results, output_dir):
    report_file = os.path.join(output_dir, "ANALISIS_PESTLE_MANUAL_LDA.txt")
    best_k = int(best_row['Num_Topics'])
    
    with open(report_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("      LAPORAN EKSPERIMEN TOPIC MODELING (LDA)\n")
        f.write("="*60 + "\n\n")
        
        f.write("[1] SPESIFIKASI MODEL & METODOLOGI\n")
        f.write(f"    - Metode              : Latent Dirichlet Allocation (LDA)\n")
        f.write(f"    - Library             : Gensim\n")
        f.write(f"    - Input Preprocessing : Stopword Removal (Heavy Cleaning) + Filter Extremes\n")
        f.write(f"    - Random Seed         : {SEED}\n\n")
        
        f.write("[2] HASIL TUNING (KOMPARASI JUMLAH TOPIK)\n")
        f.write(df_results.to_string(index=False))
        f.write("\n\n")
        
        f.write("[3] PERFORMA MODEL TERBAIK (SELECTED MODEL)\n")
        f.write(f"    - Jumlah Topik (K)  : {best_k}\n")
        f.write(f"    - Coherence NPMI    : {best_row['NPMI']:.4f}\n")
        f.write(f"    - Coherence Cv      : {best_row['Cv']:.4f}\n")
        f.write(f"    - Topic Diversity   : {best_row['Diversity']:.4f}\n")
        f.write("="*60 + "\n\n")
        
        # --- DAFTAR TOPIK UNTUK MAPPING PESTLE ---
        topics = best_model.show_topics(num_topics=-1, num_words=15, formatted=False)
        for topic_id, word_list in topics:
            keywords = ", ".join([w[0] for w in word_list])
            f.write(f"[TOPIK #{topic_id}]\n")
            f.write(f"Keywords: {keywords}\n")
            f.write("Analisis PESTLE (Silakan Centang):\n")
            f.write("[ ] Political (Kebijakan, Partai, Tokoh)\n")
            f.write("[ ] Economic (Anggaran, Harga, Pasar)\n")
            f.write("[ ] Social (Masyarakat, Gizi, Kesehatan)\n")
            f.write("[ ] Technological (Sistem, Data, Aplikasi)\n")
            f.write("[ ] Legal (Hukum, Aturan, Sanksi)\n")
            f.write("[ ] Environmental (Limbah, Lingkungan)\n")
            f.write("-" * 40 + "\n")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load & Prepare
    data_tokens = prepare_data(INPUT_FILE)
    if not data_tokens: return
    
    # Tangkap data_tokens yang baru (yang sudah Bigram) dari return function
    id2word, corpus, data_tokens = create_dictionary_corpus(data_tokens)
    
    # 2. Tuning Loop
    df_results, models_store = train_and_evaluate(data_tokens, id2word, corpus)
    
    # 3. Pilih Model Terbaik (Prioritas NPMI)
    best_row = df_results.loc[df_results['NPMI'].idxmax()]
    best_k = int(best_row['Num_Topics'])
    best_model = models_store[best_k]
    
    print(f"\n{'='*50}")
    print(f"  PEMENANG TUNING: {best_k} TOPIK")
    print(f"{'='*50}")
    print(f"Skor NPMI Tertinggi: {best_row['NPMI']:.4f}")
    
    # Simpan Hasil
    df_results.to_csv(os.path.join(OUTPUT_DIR, "tabel_tuning_lda.csv"), index=False)
    
    # --- VISUALISASI EVALUASI (UPDATE) ---
    plot_tuning_result(df_results, OUTPUT_DIR)
    
    # Visualisasi & Laporan
    generate_visualizations(best_model, corpus, id2word, best_k, OUTPUT_DIR)
    save_detailed_report(best_model, best_row, df_results, OUTPUT_DIR)
            
    print(f"\n[+] Selesai. Laporan detail ada di: {os.path.join(OUTPUT_DIR, 'ANALISIS_PESTLE_MANUAL_LDA.txt')}")

if __name__ == "__main__":
    main()