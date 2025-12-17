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
from tqdm import tqdm

# Libraries
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# TAMBAHAN: Library Visualisasi LDA
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# --- KONFIGURASI GLOBAL WARNING ---
warnings.filterwarnings("ignore")
import logging
logging.getLogger("gensim").setLevel(logging.ERROR) 

# --- KONFIGURASI ---
INPUT_FILE = "preprocessing_berita_revisi.csv"
OUTPUT_DIR = "bigram_Result_Hybrid_Analysis"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# PARAMETER HYBRID
NUM_TOPICS_LDA = 25  # Menggunakan Hasil Tuning Terbaik
EMBEDDING_MODEL = "just-108/IndoGovBERT-Wirawans-FT-C3" 

# [TAMBAHAN: PILIHAN BIGRAM]
USE_BIGRAMS = True  # Set True untuk mengaktifkan Frasa, False untuk kata tunggal

# STOPWORDS (Untuk Representasi Akhir BERTopic)
STOPWORDS_ID = [
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "dengan", 
    "adalah", "yaitu", "tersebut", "juga", "sudah", "telah", "akan", "sedang", 
    "tapi", "tetapi", "melalui", "karena", "oleh", "sebagai", "bisa", "dapat",
    "seperti", "dalam", "antara", "bagi", "kepada", "agar", "supaya", "atau",
    "saya", "kita", "kami", "anda", "mereka", "dia", "ia", "beliau", "rp", "ndak",
    "mengatakan", "kata", "ujar", "tutur", "jelas", "ungkap", "sebut", "menurut",
    "menjadi", "melakukan", "memberikan", "mengambil", "memiliki", "ada", "tidak",
    "banyak", "sedikit", "besar", "kecil", "baru", "lama", "tinggi", "rendah", 
    "sangat", "lebih", "paling", "kurang", "cukup", "sendiri", "lain", "tulis"
]

def clean_tokenize_lda(text):
    """Tokenisasi khusus LDA (tanpa stopwords)"""
    tokens = re.findall(r'\w+', text.lower())
    return [t for t in tokens if len(t) > 2]

def run_lda_process(valid_docs_tokens):
    """
    Melatih LDA pada dokumen yang SUDAH divalidasi
    Output: Model LDA, Corpus, Label per Dokumen, Dictionary
    """
    # --- MODIFIKASI: LOGIKA BIGRAM TOGGLE ---
    if USE_BIGRAMS:
        print("[-] Mode Bigram: AKTIF. Membangun Frasa...")
        bigram = gensim.models.Phrases(valid_docs_tokens, min_count=5, threshold=50) 
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        data_for_lda = [bigram_mod[doc] for doc in valid_docs_tokens]
    else:
        print("[-] Mode Bigram: NON-AKTIF. Menggunakan Unigram...")
        data_for_lda = valid_docs_tokens
    # ----------------------------------------

    id2word = corpora.Dictionary(data_for_lda)
    id2word.filter_extremes(no_below=10, no_above=0.4)
    
    corpus = [id2word.doc2bow(text) for text in data_for_lda]
    
    print(f"[-] Melatih LDA Guide ({NUM_TOPICS_LDA} Topik)...")
    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=NUM_TOPICS_LDA, 
                         random_state=SEED,
                         chunksize=2000,
                         passes=20,
                         alpha='auto',
                         eta='auto')
    
    print("[-] Mengekstrak Label Topik dari LDA...")
    lda_labels = []
    for bow in corpus:
        topics = lda_model.get_document_topics(bow)
        dominant_topic = max(topics, key=lambda x: x[1])[0]
        lda_labels.append(dominant_topic)
        
    # Return corpus & id2word juga untuk visualisasi pyLDAvis
    return lda_model, lda_labels, corpus, id2word

def calculate_metrics_hybrid(topic_model, docs_tokens):
    topics = topic_model.get_topics()
    cleaned_topics = []
    for topic_id, words in topics.items():
        if topic_id != -1: 
            cleaned_topics.append([word for word, _ in words])

    if not cleaned_topics: return 0.0, 0.0, 0.0

    dictionary = Dictionary(docs_tokens)

    # NPMI
    try:
        cm_npmi = CoherenceModel(topics=cleaned_topics, texts=docs_tokens, dictionary=dictionary, coherence='c_npmi')
        npmi = cm_npmi.get_coherence()
        if math.isnan(npmi): npmi = 0.0
    except: npmi = 0.0

    # Cv
    try:
        cm_cv = CoherenceModel(topics=cleaned_topics, texts=docs_tokens, dictionary=dictionary, coherence='c_v')
        cv = cm_cv.get_coherence()
        if math.isnan(cv): cv = 0.0
    except: cv = 0.0

    # Diversity
    try:
        unique_words = set()
        total_words = 0
        for topic_id, words in topics.items():
            if topic_id != -1:
                top_n = [word for word, _ in words[:10]]
                unique_words.update(top_n)
                total_words += 10
        diversity = len(unique_words) / total_words if total_words > 0 else 0
    except: diversity = 0.0
    
    return npmi, cv, diversity

def save_report_and_viz(topic_model, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "plots")
    os.makedirs(img_dir, exist_ok=True)
    
    npmi, cv, diversity = metrics
    freq = topic_model.get_topic_info()
    
    # Info Hardware
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"

    # 1. Laporan TXT
    report_file = os.path.join(output_dir, "LAPORAN_HYBRID_LDA_BERT.txt")
    with open(report_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("      LAPORAN HYBRID TOPIC MODELING (LDA-Guided BERTopic)\n")
        f.write("="*60 + "\n\n")
        f.write(f"[1] ARSITEKTUR & METODOLOGI\n")
        f.write(f"    - Device Run        : {DEVICE.upper()} ({gpu_name})\n")
        f.write(f"    - Clustering Engine : LDA (Gensim) - {NUM_TOPICS_LDA} Topik\n")
        f.write(f"    - Representation    : BERTopic (c-TF-IDF) with IndoGovBERT\n")
        f.write(f"    - Embedding Model   : {EMBEDDING_MODEL}\n")
        f.write(f"    - Bigram Mode       : {'AKTIF' if USE_BIGRAMS else 'NON-AKTIF'}\n")
        f.write(f"    - Alignment Strategy: Pre-filtered Sync (Docs & Tokens matched)\n\n")
        f.write(f"[2] HASIL EVALUASI\n")
        f.write(f"    - Coherence NPMI    : {npmi:.4f}\n")
        f.write(f"    - Coherence Cv      : {cv:.4f}\n")
        f.write(f"    - Diversity         : {diversity:.4f}\n\n")
        f.write("[3] DAFTAR TOPIK HYBRID (Analisis PESTLE)\n")
        f.write("-" * 60 + "\n")
        
        for idx, row in freq.iterrows():
            if row['Topic'] == -1: continue
            tid = row['Topic']
            count = row['Count']
            words = topic_model.get_topic(tid)
            keywords = ", ".join([w[0] for w in words[:10]])
            f.write(f"[HYBRID TOPIK #{tid}] - Jumlah Berita: {count}\n")
            f.write(f"Keywords: {keywords}\n")
            f.write("Analisis PESTLE:\n[ ] Political\n[ ] Economic\n[ ] Social\n[ ] Technological\n[ ] Legal\n[ ] Environmental\n")
            f.write("-" * 40 + "\n")

    # 2. Visualisasi BERTopic
    print("[-] Membuat Visualisasi BERTopic...")
    top_15 = freq.head(15).sort_values('Count', ascending=True)
    plt.figure(figsize=(12, 8))
    plt.barh(top_15['Name'].apply(lambda x: " ".join(x.split("_")[1:5])), top_15['Count'], color='#9b59b6')
    plt.title(f'Topik Dominan - Hybrid Model', fontsize=15)
    plt.xlabel('Jumlah Berita')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "1_barchart_hybrid.png"), dpi=300)
    plt.close()
    
    try:
        topic_model.visualize_topics().write_html(os.path.join(output_dir, "interactive_hybrid_bertopic.html"))
    except: pass

def save_lda_visualization(lda_model, corpus, id2word, output_dir):
    """
    Membuat visualisasi pyLDAvis (Persis seperti gambar referensi user)
    """
    print("[-] Membuat Visualisasi pyLDAvis (LDA Map)...")
    try:
        img_dir = os.path.join(output_dir, "plots")
        os.makedirs(img_dir, exist_ok=True)
        
        # Persiapan data pyLDAvis
        vis_data = gensimvis.prepare(lda_model, corpus, id2word)
        
        # Simpan ke HTML
        output_path = os.path.join(output_dir, "interactive_hybrid_pyldavis.html")
        pyLDAvis.save_html(vis_data, output_path)
        print(f"    > Sukses! File tersimpan di: {output_path}")
    except Exception as e:
        print(f"    [!] Gagal membuat pyLDAvis: {e}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("="*60)
    print("       PIPELINE HYBRID: LDA-GUIDED BERTOPIC")
    print("="*60)
    print(f"[-] Status Device: {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"    GPU Name: {torch.cuda.get_device_name(0)}")
    
    # 1. Load Data
    if not os.path.exists(INPUT_FILE): return
    df = pd.read_csv(INPUT_FILE)
    
    # 2. FILTERING & ALIGNMENT
    print("[-] Melakukan Alignment Data (Filter Dokumen Pendek)...")
    
    valid_docs_text = []   
    valid_docs_tokens = [] 
    
    raw_docs = df['processed_text'].dropna().astype(str).tolist()
    
    for doc in tqdm(raw_docs, desc="Filtering"):
        tokens = clean_tokenize_lda(doc)
        if len(tokens) > 3: 
            valid_docs_text.append(doc)      
            valid_docs_tokens.append(tokens) 
            
    print(f"    > Total Awal: {len(raw_docs)} -> Valid: {len(valid_docs_text)}")
    
    # 3. JALANKAN LDA (OTAK KIRI)
    # Sekarang kita tangkap corpus dan id2word juga untuk visualisasi
    lda_model, lda_labels, corpus, id2word = run_lda_process(valid_docs_tokens)
    
    # 4. JALANKAN BERTOPIC (OTAK KANAN)
    print(f"[-] Menjalankan BERTopic dengan Panduan LDA...")
    
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    embeddings = embedding_model.encode(valid_docs_text, show_progress_bar=True, batch_size=32)
    
    vectorizer_model = CountVectorizer(ngram_range=(1, 1), stop_words=STOPWORDS_ID)
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        verbose=True
    )
    
    topic_model.fit_transform(valid_docs_text, embeddings=embeddings, y=lda_labels)
    
    # 5. Evaluasi
    print("[-] Menghitung Metrik Hybrid...")
    npmi, cv, diversity = calculate_metrics_hybrid(topic_model, valid_docs_tokens)
    print(f"    > HASIL HYBRID: Topik={len(topic_model.get_topic_info())-1} | NPMI={npmi:.4f} | Cv={cv:.4f}")
    
    # 6. Simpan Hasil
    save_report_and_viz(topic_model, (npmi, cv, diversity), OUTPUT_DIR)
    
    # 7. Simpan Visualisasi pyLDAvis (SESUAI REQUEST GAMBAR)
    save_lda_visualization(lda_model, corpus, id2word, OUTPUT_DIR)
    
    print(f"\n[+] Selesai. Hasil Hybrid ada di: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()