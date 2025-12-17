import pandas as pd
import re
import os
from datetime import datetime, timedelta

# --- KONFIGURASI ---
# Tanggal Patokan (Saat kamu melakukan scraping)
# Ganti sesuai tanggal terakhir kamu run script scraping
CURRENT_DATE = datetime(2025, 11, 30) 

def perbaiki_tanggal(teks_tanggal):
    """Fungsi sakti pengubah tanggal 'alay' jadi standar ISO"""
    if pd.isna(teks_tanggal) or str(teks_tanggal).strip() == "":
        return ""
    
    teks = str(teks_tanggal).lower().strip()
    
    # Kalau sudah format ISO (2025-11-19), skip
    if re.match(r'^\d{4}-\d{2}-\d{2}$', teks):
        return teks

    try:
        # 1. Handle Relative Time (detik/menit/jam/hari/minggu/bulan/tahun lalu)
        match = re.search(r'(\d+)\s+(detik|menit|jam|hari|minggu|bulan|tahun)', teks)
        if match:
            angka = int(match.group(1))
            satuan = match.group(2)
            
            delta = timedelta(days=0)
            if satuan in ['detik', 'menit', 'jam']:
                delta = timedelta(days=0) # Hari ini
            elif satuan == 'hari':
                delta = timedelta(days=angka)
            elif satuan == 'minggu':
                delta = timedelta(weeks=angka)
            elif satuan == 'bulan':
                delta = timedelta(days=angka * 30)
            elif satuan == 'tahun':
                delta = timedelta(days=angka * 365)
            
            return (CURRENT_DATE - delta).strftime("%Y-%m-%d")

        # 2. Handle "Kemarin"
        if 'kemarin' in teks:
            return (CURRENT_DATE - timedelta(days=1)).strftime("%Y-%m-%d")

        # 3. Handle Format Indo (17 November 2025)
        teks_bersih = re.sub(r'wib|wita|wit|pukul.*', '', teks).strip()
        teks_bersih = re.sub(r'[,|\|].*', '', teks_bersih).strip() # Hapus jam setelah koma

        bulan_indo = {
            'januari': '01', 'februari': '02', 'maret': '03', 'april': '04',
            'mei': '05', 'juni': '06', 'juli': '07', 'agustus': '08',
            'september': '09', 'oktober': '10', 'november': '11', 'desember': '12',
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'mei': '05', 'jun': '06',
            'jul': '07', 'agu': '08', 'sep': '09', 'okt': '10', 'nov': '11', 'des': '12'
        }

        # Regex: (Angka) (Huruf) (Angka)
        match_indo = re.search(r'(\d{1,2})\s+([a-z]+)\s+(\d{4})', teks_bersih)
        if match_indo:
            tgl = match_indo.group(1).zfill(2)
            nm_bln = match_indo.group(2)
            thn = match_indo.group(3)
            
            if nm_bln in bulan_indo:
                return f"{thn}-{bulan_indo[nm_bln]}-{tgl}"
        
        # Regex: (Angka)/(Angka)/(Angka)
        match_slash = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', teks_bersih)
        if match_slash:
             # Asumsi format Indonesia: DD/MM/YYYY
             return f"{match_slash.group(3)}-{match_slash.group(2).zfill(2)}-{match_slash.group(1).zfill(2)}"

    except:
        pass

    return teks_tanggal # Kalau gagal, kembalikan aslinya

def main():
    # Cari semua file CSV di folder ini
    files = [f for f in os.listdir('.') if f.endswith('.csv') and 'dataset' in f or 'lampiran' in f]
    
    print(f"🔍 Ditemukan {len(files)} file CSV yang akan diperbaiki tanggalnya:")
    for f in files: print(f"   - {f}")
    print("-" * 30)

    for filename in files:
        print(f"📂 Memproses: {filename}...")
        try:
            df = pd.read_csv(filename)
            
            # Cari nama kolom tanggal yang valid
            col_target = None
            possible_names = ['Tanggal', 'Tanggal_Tayang', 'Date', 'Waktu', 'date', 'tanggal']
            
            for col in possible_names:
                if col in df.columns:
                    col_target = col
                    break
            
            if col_target:
                print(f"   🛠️  Memperbaiki kolom '{col_target}'...")
                # Terapkan perbaikan
                df[col_target] = df[col_target].apply(perbaiki_tanggal)
                
                # Simpan (Overwrite file lama biar gak menuh-menuhin folder)
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"   ✅ Selesai disimpan.")
            else:
                print(f"   ⚠️  Tidak ada kolom tanggal di file ini. Skip.")
                
        except Exception as e:
            print(f"   ❌ Gagal memproses file ini: {e}")

    print("\n" + "="*50)
    print("🎉 SEMUA TANGGAL SUDAH DIPERBAIKI KE FORMAT YYYY-MM-DD")
    print("="*50)

if __name__ == "__main__":
    main()