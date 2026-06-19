# Deteksi Dini Micro-Crash di Pasar Kripto dengan SAX dan KMP

Penerapan Algoritma KMP pada Representasi Simbolik Limit Order Book untuk Deteksi Dini Micro-Crash di Pasar Kripto.

Tugas Makalah IF2211 Strategi Algoritma, Semester II 2025/2026, Institut Teknologi Bandung.

## Deskripsi

Program ini mengimplementasikan pipeline deteksi dini micro-crash yang mengintegrasikan algoritma Symbolic Aggregate approXimation (SAX) dengan algoritma pencocokan string Knuth-Morris-Pratt (KMP). Deret waktu Order Book Imbalance (OBI) dikonversi menjadi representasi simbolik menggunakan SAX, kemudian dipindai oleh KMP untuk mendeteksi pola-pola karakteristik vakum likuiditas sebelum terjadinya micro-crash.

## Cara Menjalankan

### Prasyarat

```
pip install -r requirements.txt
```

### Eksekusi

```
python src/main_pipeline.py
```

Hasil eksperimen akan tersimpan di folder `results/`.

## Struktur Folder

```
.
├── README.md
├── requirements.txt
├── src/
│   └── main_pipeline.py
├── results/
│   ├── experiment_results.json
│   ├── btc_usdt_data.csv
│   ├── table_parameter_comparison.csv
│   ├── table_baseline_comparison.csv
│   ├── table_speed_benchmark.csv
│   ├── fig1_price_obi_crashes.png
│   ├── fig2_sax_encoding_detail.png
│   ├── fig3_parameter_comparison.png
│   ├── fig4_kmp_vs_bruteforce.png
│   └── fig5_sax_kmp_vs_baseline.png
└── doc/
    ├── makalah.tex
    └── makalah.pdf
```

## Penulis

Rava Khoman Tuah Saragih (13524107)