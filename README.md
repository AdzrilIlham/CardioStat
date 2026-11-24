# CardioStat Lab

**Sistem Analisis Empiris Faktor Risiko Serangan Jantung Berbasis Statistika Terapan (Non-Machine Learning)**

CardioStat Lab adalah aplikasi web interaktif berbasis **Python + Streamlit** untuk mengolah dan menguji dataset klinis faktor risiko serangan jantung secara deskriptif, uji asumsi klasik, hingga statistika inferensial—lengkap dengan visualisasi interaktif (`Plotly`), narasi interpretasi klinis otomatis dalam bahasa Indonesia, dan ekspor laporan Excel (`.xlsx`) multi-sheet.

---

## Alur Kerja Antarmuka (5-Stage Progressive Workflow)

1. **Tahap 1 — Panel Unggah Data**: Mendukung unggah file `.csv`, `.xlsx`, `.xls`, atau memuat langsung dataset bawaan (`heart_attack_prediction_dataset.csv`). Kolom `Blood Pressure` berformat `"Sistolik/Diastolik"` (mis. `"158/88"`) otomatis dipecah menjadi dua kolom numerik `Systolic_BP` dan `Diastolic_BP`.
2. **Tahap 2 — Pratinjau Tabel Dataset**: Kartu ringkasan jumlah pasien, jumlah variabel, prevalensi risiko, status *auto-parse* tekanan darah, serta tabel interaktif dengan fitur pencarian dan paginasi.
3. **Tahap 3 — Bilah Navigasi Analisis Horizontal**: 6 tombol modul uji statistik tepat di bawah tabel.
4. **Tahap 4 — Konfigurasi Variabel Dinamis**: Panduan uji (*Kegunaan*, *Kriteria Keputusan*, *Referensi*) serta dropdown kolom yang otomatis memfilter variabel numerik dan kategorik yang valid.
5. **Tahap 5 — Hasil Uji, Visualisasi & Ekspor**: Kotak narasi klinis otomatis, tabel statistik lengkap, grafik interaktif `Plotly`, dan tombol unduh laporan `.xlsx` multi-sheet.

---

## 6 Modul Analisis Statistik

| No | Modul | Metode & Output Utama |
|---|---|---|
| 1 | **Deskriptif & Risiko (PAN/PAK)** | Mean, Median, Mode, SD, Variance, SE, 95% CI, CV%, Min, P10/Q1/Q3/P90, Max, Range, IQR, Skewness, Excess Kurtosis; Tabel Distribusi Frekuensi; Kategorisasi 2–5 tingkat (**PAK Acuan Klinis Medis**, **PAK Skor Ideal Teoretis $M_i$ & $SD_i$**, dan **PAN Mean ± SD Empiris**); Histogram, Bar Chart Kategori, Boxplot, Violin Plot. |
| 2 | **Uji Asumsi (Shapiro-Wilk & Levene)** | Uji normalitas *Shapiro-Wilk* (maks. 5.000 sampel acak bila $N > 5.000$) + *Normal Q-Q Plot*, serta uji homogenitas varians *Levene's Test* + Boxplot per kelompok. |
| 3 | **Uji Beda 2 Kelompok** | *Independent t-test* (Student/Welch otomatis berdasar Levene, $t$, $df$, $p$, 95% CI beda mean, *Cohen's d*) & *Mann-Whitney U* ($U$, $p$, *Rank-Biserial r*) + Boxplot perbandingan. |
| 4 | **Uji Beda >2 Kelompok** | *One-Way ANOVA* ($F$, $df_1$, $df_2$, $p$, $\eta^2$), *Kruskal-Wallis* ($H$, $df$, $p$, $\varepsilon^2$), dan uji lanjut *Post-Hoc Tukey HSD* + Boxplot perbandingan. |
| 5 | **Uji Asosiasi (Chi-Square & Cramér's V)** | *Chi-Square Test of Independence* ($\chi^2$, $df$, $p$), *Cramér's V*, *Minimum Expected Count*, Tabel Kontingensi, dan Grafik Batang Berkelompok. |
| 6 | **Korelasi & Regresi Berganda** | Korelasi *Pearson*, *Spearman*, dan *Kendall* (matriks heatmap & $p$-value berpasangan) serta Regresi Linear Berganda OLS ($R^2$, Adjusted $R^2$, uji simultan $F$, RMSE, koefisien $B$, *Std. Beta*, 95% CI, VIF, statistik residual, dan plot residual). |

---

## Struktur Proyek

```text
CardioStat/
├── app.py                                  # Antarmuka Streamlit 5 tahap & visualisasi Plotly
├── stats_engine.py                         # Mesin perhitungan statistik, narasi klinis & ekspor Excel
├── requirements.txt                        # Daftar dependensi Python
├── data/
│   └── heart_attack_prediction_dataset.csv # Dataset bawaan (8.763 pasien, 26 variabel asli)
└── .gitignore
```

---

## Cara Menjalankan di Lokal

1. **Clone repository:**
   ```powershell
   git clone https://github.com/joessar/CardioStat.git
   cd CardioStat
   ```

2. **(Opsional) Buat dan aktifkan virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Pasang dependensi:**
   ```powershell
   python -m pip install -r requirements.txt
   ```

4. **Jalankan aplikasi Streamlit:**
   ```powershell
   python -m streamlit run app.py
   ```
   Buka URL lokal yang ditampilkan di terminal (umumnya `http://localhost:8501`).
