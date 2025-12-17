import pandas as pd
import os
import sys
import torch
import re  
import math 
import numpy as np
import warnings 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm

import bertopic
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer, models
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# --- KONFIGURASI GLOBAL WARNING ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- KONFIGURASI ---
INPUT_FILE = "preprocessing_berita_revisi.csv"
OUTPUT_DIR = "Result_BERTopic_Analysis_revisi"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# DAFTAR STOPWORDS (Untuk membuang topik sampah "yang", "dan", dll)
STOPWORDS_ID = [
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "dengan", 
    "adalah", "yaitu", "tersebut", "juga", "sudah", "telah", "akan", "sedang", 
    "tapi", "tetapi", "melalui", "karena", "oleh", "sebagai", "bisa", "dapat",
    "seperti", "dalam", "antara", "bagi", "kepada", "agar", "supaya", "atau",
    "saya", "kita", "kami", "anda", "mereka", "dia", "ia", "beliau", "rp", "ndak",
    "mengatakan", "kata", "ujar", "tutur", "jelas", "ungkap", "sebut", "menurut",
    "menjadi", "melakukan", "memberikan", "mengambil", "memiliki", "ada", "tidak",
    "banyak", "sedikit", "besar", "kecil", "baru", "lama", "tinggi", "rendah",
    "sangat", "lebih", "paling", "kurang", "cukup", "sendiri", "lain"
]

# 1. PARAMETER ARSITEKTUR
BERTOPIC_PARAMS = {
    'umap_neighbors': 15,
    'umap_components': 5,
    'umap_metric': 'cosine',
    'hdbscan_min_cluster': 15,
    'hdbscan_metric': 'euclidean',
    'vectorizer_ngram': (1, 1) 
}

# 2. MODEL YANG AKAN DIUJI
MODELS_TO_COMPARE = {
    "IndoGovBERT": "just-108/IndoGovBERT-Wirawans-FT-C3",
    "IndoBERT": "indolem/indobert-base-uncased",
    "Multilingual-MiniLM": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

def build_embedding_model(model_name):
    print(f"    [-] Menginisialisasi Model: {model_name} (Device: {DEVICE.upper()})")
    try:
        if 'sentence-transformers' in model_name or 'paraphrase' in model_name:
             model = SentenceTransformer(model_name, device=DEVICE)
        else:
            print(f"        [INFO] Terdeteksi Raw BERT. Menambahkan layer Pooling otomatis...")
            word_embedding_model = models.Transformer(model_name, max_seq_length=512)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True)
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=DEVICE)
        return model, model_name 
    except Exception as e:
        print(f"        [ERROR] Gagal memuat {model_name}: {e}")
        fallback = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        print(f"        [FALLBACK] Mengalihkan ke model aman: {fallback}")
        return SentenceTransformer(fallback, device=DEVICE), fallback

def calculate_metrics(topic_model, docs_tokens):
    """Menghitung NPMI, Cv, dan Diversity dengan NaN Handler"""
    topics = topic_model.get_topics()
    cleaned_topics = []
    
    # Filter outlier (-1)
    for topic_id, words in topics.items():
        if topic_id != -1: 
            cleaned_topics.append([word for word, _ in words])

    if not cleaned_topics: return 0.0, 0.0, 0.0

    # Dictionary Gensim (Dibangun dari token yang bersih)
    dictionary = Dictionary(docs_tokens)

    # 1. Coherence NPMI
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cm_npmi = CoherenceModel(topics=cleaned_topics, texts=docs_tokens, dictionary=dictionary, coherence='c_npmi')
            npmi = cm_npmi.get_coherence()
            
        if math.isnan(npmi): npmi = 0.0 
    except:
        npmi = 0.0

    # 2. Coherence Cv
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cm_cv = CoherenceModel(topics=cleaned_topics, texts=docs_tokens, dictionary=dictionary, coherence='c_v')
            cv = cm_cv.get_coherence()
            
        if math.isnan(cv): cv = 0.0 
    except:
        cv = 0.0

    # 3. Topic Diversity
    unique_words = set()
    total_words = 0
    for topic_id, words in topics.items():
        if topic_id != -1:
            top_n = [word for word, _ in words[:10]]
            unique_words.update(top_n)
            total_words += 10
    
    diversity = len(unique_words) / total_words if total_words > 0 else 0
    
    return npmi, cv, diversity

def generate_visualizations(topic_model, output_dir, model_label):
    freq = topic_model.get_topic_info()
    freq = freq[freq['Topic'] != -1]
    
    if freq.empty: return

    # A. Bar Chart Top Topics
    top_15 = freq.head(15).sort_values('Count', ascending=True)
    plt.figure(figsize=(12, 8))
    plt.barh(top_15['Name'].apply(lambda x: " ".join(x.split("_")[1:5])), top_15['Count'], color='#3498db')
    plt.title(f'Topik Dominan - {model_label}', fontsize=15)
    plt.xlabel('Jumlah Berita')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_barchart_top_topics.png"), dpi=300)
    plt.close()

    # B. WordClouds
    print(f"    [-] Membuat WordCloud ({model_label})...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    for i, (index, row) in enumerate(freq.head(6).iterrows()):
        topic_id = row['Topic']
        words = topic_model.get_topic(topic_id)
        if not words: continue
        word_freq = {w[0]: w[1] for w in words}
        wc = WordCloud(width=600, height=400, background_color='white', colormap='Dark2').generate_from_frequencies(word_freq)
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].axis("off")
        axes[i].set_title(f"Topik {topic_id}\n(n={row['Count']})", fontsize=14)
    
    for j in range(i+1, 6): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_wordclouds.png"), dpi=300)
    plt.close()

    # C. Interactive Maps
    try:
        topic_model.visualize_topics().write_html(os.path.join(output_dir, "interactive_map.html"))
        topic_model.visualize_hierarchy().write_html(os.path.join(output_dir, "interactive_hierarchy.html"))
    except: pass

def save_model_specific_report(topic_model, model_name, metrics, output_dir):
    report_file = os.path.join(output_dir, f"LAPORAN_{model_name}.txt")
    
    npmi, cv, diversity = metrics
    freq = topic_model.get_topic_info()
    n_topics = len(freq) - 1

    with open(report_file, "w") as f:
        f.write("="*60 + "\n")
        f.write(f"      LAPORAN DETAIL MODEL: {model_name}\n")
        f.write("="*60 + "\n\n")
        
        f.write("[1] SPESIFIKASI ARSITEKTUR\n")
        f.write(f"    - Embedding Model   : {model_name}\n")
        f.write(f"    - UMAP Config       : Neigh={BERTOPIC_PARAMS['umap_neighbors']}, Comp={BERTOPIC_PARAMS['umap_components']}\n")
        f.write(f"    - HDBSCAN Config    : MinCluster={BERTOPIC_PARAMS['hdbscan_min_cluster']}\n")
        f.write(f"    - Vectorizer        : Ngram={BERTOPIC_PARAMS['vectorizer_ngram']}\n")
        f.write(f"    - Stopwords Applied : YA ({len(STOPWORDS_ID)} kata)\n")
        f.write(f"    - Device Run        : {DEVICE.upper()}\n\n")

        f.write("[2] HASIL EVALUASI\n")
        f.write(f"    - Jumlah Topik      : {n_topics}\n")
        f.write(f"    - Coherence NPMI    : {npmi:.4f}\n")
        f.write(f"    - Coherence Cv      : {cv:.4f}\n")
        f.write(f"    - Topic Diversity   : {diversity:.4f}\n\n")
        
        f.write("[3] DAFTAR TOPIK (Untuk Analisis Manual PESTLE)\n")
        f.write("-" * 60 + "\n")
        
        for idx, row in freq.head(20).iterrows():
            if row['Topic'] == -1: continue
            tid = row['Topic']
            count = row['Count']
            words = topic_model.get_topic(tid)
            keywords = ", ".join([w[0] for w in words[:10]])
            
            f.write(f"[TOPIK #{tid}] - Jumlah Berita: {count}\n")
            f.write(f"Keywords: {keywords}\n")
            f.write("Analisis PESTLE:\n")
            f.write("[ ] Political\n[ ] Economic\n[ ] Social\n[ ] Technological\n[ ] Legal\n[ ] Environmental\n")
            f.write("-" * 40 + "\n")

def save_comparison_report(df_results, output_dir):
    """Membuat file rangkuman gabungan (.txt)"""
    report_file = os.path.join(output_dir, "RANGKUMAN_PERBANDINGAN_MODEL.txt")
    
    best_row = df_results.loc[df_results['Coherence_NPMI'].idxmax()]
    
    with open(report_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("      LAPORAN PERBANDINGAN MODEL BERTOPIC (SUMMARY)\n")
        f.write("="*60 + "\n\n")
        
        f.write("[1] SKEMA EKSPERIMEN\n")
        f.write(f"    - Tujuan            : Membandingkan kualitas topik dari 3 model embedding berbeda.\n")
        f.write(f"    - Model Diuji       : {', '.join(df_results['Model'].tolist())}\n")
        f.write(f"    - Metrik Utama      : NPMI (Normalized Pointwise Mutual Information)\n")
        f.write(f"    - Parameter Fixed   : UMAP(n=15), HDBSCAN(min=15), Vectorizer(ngram=1, stopwords=YES)\n\n")
        
        f.write("[2] TABEL HASIL KOMPARASI\n")
        f.write(df_results.to_string(index=False, float_format="%.4f"))
        f.write("\n\n")
        
        f.write("[3] KESIMPULAN & PEMENANG\n")
        f.write(f"    - Model Terbaik     : {best_row['Model']}\n")
        f.write(f"    - Skor NPMI         : {best_row['Coherence_NPMI']:.4f}\n")
        f.write(f"    - Skor Cv           : {best_row['Coherence_Cv']:.4f}\n")
        f.write(f"    - Diversity         : {best_row['Diversity']:.4f}\n\n")
        
        f.write("[4] ANALISIS SINGKAT\n")
        f.write(f"    Model '{best_row['Model']}' terpilih karena memiliki skor NPMI tertinggi,\n")
        f.write("    yang mengindikasikan bahwa topik-topik yang dihasilkannya memiliki koherensi\n")
        f.write("    semantik yang paling kuat dan mudah diinterpretasikan manusia dibandingkan model lainnya.\n")

def plot_comparison(results_df, output_dir):
    plt.figure(figsize=(12, 7))
    bar_width = 0.25
    x = np.arange(len(results_df['Model']))
    
    plt.bar(x - bar_width, results_df['Coherence_Cv'], width=bar_width, label='Coherence (Cv)', color='#3498db', alpha=0.9)
    plt.bar(x, results_df['Coherence_NPMI'], width=bar_width, label='Coherence (NPMI)', color='#2ecc71', alpha=0.9)
    plt.bar(x + bar_width, results_df['Diversity'], width=bar_width, label='Topic Diversity', color='#e74c3c', alpha=0.9)
    
    plt.xlabel('Model Embedding', fontsize=12)
    plt.ylabel('Skor Evaluasi', fontsize=12)
    plt.title('Evaluasi Tuning: Perbandingan Performa Model Embedding', fontsize=14)
    plt.xticks(x, results_df['Model'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    for i in x:
        val_npmi = results_df['Coherence_NPMI'].iloc[i]
        plt.text(i, max(0, val_npmi) + 0.01, f"{val_npmi:.3f}", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "GRAFIK_PERBANDINGAN_SKOR.png"), dpi=300)
    plt.close()

def run_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- STARTUP INFO ---
    print("="*60)
    print("       PIPELINE ANALISIS TOPIK BERITA (BERTopic)")
    print("="*60)
    print(f"[-] Device: {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(INPUT_FILE):
        print(f"\n[!] ERROR: Input {INPUT_FILE} tidak ditemukan.")
        return

    print("[-] Memuat dataset...")
    df = pd.read_csv(INPUT_FILE)
    docs = df['processed_text'].dropna().astype(str).tolist()
    
    # --- TOKENISASI ROBUST UNTUK GENSIM ---
    print("[-] Menyiapkan tokens untuk Evaluasi Gensim (Regex Cleaning)...")
    def clean_tokenize(text):
        return re.findall(r'\w+', text.lower()) 
    
    docs_tokens = [clean_tokenize(doc) for doc in tqdm(docs, desc="Tokenizing")]
    
    print(f"    > Total Dokumen: {len(docs)} berita")
    print("-" * 60)

    comparison_results = []

    # LOOP EXPERIMENT
    for label, model_path in MODELS_TO_COMPARE.items():
        print(f"\n{'='*50}")
        print(f"  EKSPERIMEN: {label}")
        print(f"{'='*50}")

        # 1. Buat folder khusus model ini
        model_output_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(model_output_dir, exist_ok=True)

        # 2. Embedding & Training
        embed_model, _ = build_embedding_model(model_path)
        print("    [-] Encoding Embeddings...")
        embeddings = embed_model.encode(docs, show_progress_bar=True, batch_size=32)

        umap_model = UMAP(n_neighbors=BERTOPIC_PARAMS['umap_neighbors'], 
                          n_components=BERTOPIC_PARAMS['umap_components'], 
                          min_dist=0.0, metric='cosine', random_state=SEED)
        
        hdbscan_model = HDBSCAN(min_cluster_size=BERTOPIC_PARAMS['hdbscan_min_cluster'], 
                                min_samples=10, metric='euclidean', 
                                cluster_selection_method='eom', prediction_data=True)

        # --- STOPWORDS DITERAPKAN DI SINI ---
        vectorizer_model = CountVectorizer(ngram_range=BERTOPIC_PARAMS['vectorizer_ngram'],
                                         stop_words=STOPWORDS_ID)

        topic_model = BERTopic(
            embedding_model=embed_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            verbose=True
        )

        print(f"    [-] Training BERTopic...")
        topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)

        # 3. Evaluasi
        metrics = calculate_metrics(topic_model, docs_tokens)
        npmi, cv, diversity = metrics
        freq = topic_model.get_topic_info()
        n_topics = len(freq) - 1 
        
        print(f"    > HASIL: Topik={n_topics} | NPMI={npmi:.4f} | Cv={cv:.4f}")
        
        comparison_results.append({
            "Model": label,
            "Jml_Topik": n_topics,
            "Coherence_Cv": cv,
            "Coherence_NPMI": npmi,
            "Diversity": diversity
        })
        
        # 4. SIMPAN LAPORAN & VISUALISASI DI FOLDER MASING-MASING
        print(f"    [-] Menyimpan Laporan & Grafik ke folder: {label}/")
        save_model_specific_report(topic_model, label, metrics, model_output_dir)
        generate_visualizations(topic_model, model_output_dir, label)

    # FINALISASI
    print(f"\n{'='*50}")
    print(f"  FINALISASI KOMPARASI")
    print(f"{'='*50}")
    
    df_results = pd.DataFrame(comparison_results)
    
    # Simpan Tabel & Grafik Gabungan di Root Folder Output
    df_results.to_csv(os.path.join(OUTPUT_DIR, "tabel_komparasi_semua.csv"), index=False)
    plot_comparison(df_results, OUTPUT_DIR)
    
    # --- TAMBAHAN: SIMPAN RANGKUMAN GABUNGAN .TXT ---
    save_comparison_report(df_results, OUTPUT_DIR)
    
    best_model = df_results.loc[df_results['Coherence_NPMI'].idxmax()]['Model']
    print(f"[+] Pemenang (NPMI Tertinggi): {best_model}")
    print(f"[+] Selesai. Cek folder '{OUTPUT_DIR}' untuk hasil lengkap.")

if __name__ == "__main__":
    run_analysis()