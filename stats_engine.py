"""Statistical computation engine and Indonesian clinical narrative generator for CardioStat Lab.

Aligned with the statistical specifications and output structure of Research & Statistics Lab (app.R).
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import openpyxl  # noqa: F401 - ensures Excel engine availability
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor


CLINICAL_CUTOFFS: dict[str, dict[str, Any]] = {
    "Cholesterol": {
        "bins": [-np.inf, 200, 239, np.inf],
        "labels": ["Rendah / Normal (<200)", "Sedang / Borderline (200-239)", "Tinggi (>=240)"],
        "note": "Acuan klinis AHA/NCEP ATP III (mg/dL).",
    },
    "Systolic_BP": {
        "bins": [-np.inf, 119, 139, np.inf],
        "labels": ["Rendah / Normal (<120)", "Sedang / Pra-hipertensi (120-139)", "Tinggi / Hipertensi (>=140)"],
        "note": "Acuan tekanan darah sistolik klinis (mmHg).",
    },
    "Diastolic_BP": {
        "bins": [-np.inf, 79, 89, np.inf],
        "labels": ["Rendah / Normal (<80)", "Sedang / Pra-hipertensi (80-89)", "Tinggi / Hipertensi (>=90)"],
        "note": "Acuan tekanan darah diastolik klinis (mmHg).",
    },
    "BMI": {
        "bins": [-np.inf, 24.99, 29.99, np.inf],
        "labels": ["Rendah / Normal (<25)", "Sedang / Overweight (25-29.9)", "Tinggi / Obesitas (>=30)"],
        "note": "Acuan klasifikasi BMI WHO (kg/m^2).",
    },
    "Triglycerides": {
        "bins": [-np.inf, 149, 199, np.inf],
        "labels": ["Rendah / Normal (<150)", "Sedang / Borderline (150-199)", "Tinggi (>=200)"],
        "note": "Acuan kadar trigliserida klinis (mg/dL).",
    },
    "Heart Rate": {
        "bins": [-np.inf, 59, 100, np.inf],
        "labels": ["Rendah / Bradikardia (<60)", "Sedang / Normal (60-100)", "Tinggi / Takikardia (>100)"],
        "note": "Acuan detak jantung istirahat (bpm).",
    },
}


def format_p_value(p: float) -> str:
    """Format p-value consistently for clinical reporting."""
    if pd.isna(p):
        return "N/A"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


def preprocess_blood_pressure(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Split 'Blood Pressure' ('sys/dia') into numeric Systolic_BP and Diastolic_BP columns."""
    out = df.copy()
    if "Blood Pressure" not in out.columns:
        return out, False

    if "Systolic_BP" in out.columns and "Diastolic_BP" in out.columns:
        return out, True

    bp_str = out["Blood Pressure"].astype(str).str.strip()
    split_df = bp_str.str.extract(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")
    if split_df.dropna().empty:
        return out, False

    systolic = pd.to_numeric(split_df[0], errors="coerce")
    diastolic = pd.to_numeric(split_df[1], errors="coerce")

    bp_idx = out.columns.get_loc("Blood Pressure")
    out.insert(bp_idx + 1, "Systolic_BP", systolic)
    out.insert(bp_idx + 2, "Diastolic_BP", diastolic)
    return out, True


def load_dataset(file_obj_or_path: Any, filename: str | None = None) -> tuple[pd.DataFrame, bool]:
    """Load CSV or Excel dataset and automatically parse Blood Pressure."""
    name = (filename or str(file_obj_or_path)).lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file_obj_or_path)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_obj_or_path)
    else:
        raise ValueError("Format file tidak didukung. Gunakan file .csv, .xlsx, atau .xls.")
    return preprocess_blood_pressure(df)


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify dataset columns into numeric, binary/categorical, 2-group, and >=3-group lists."""
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    two_group_cols: list[str] = []
    multi_group_cols: list[str] = []

    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue

        n_unique = series.nunique()
        is_num = pd.api.types.is_numeric_dtype(series)

        if not is_num and n_unique > max(30, int(0.5 * len(series))):
            continue
        if col.lower() in {"patient id", "patient_id", "id"}:
            continue

        if is_num:
            unique_vals = set(series.unique())
            if unique_vals.issubset({0, 1}) or unique_vals.issubset({0.0, 1.0}):
                categorical_cols.append(col)
                if n_unique == 2:
                    two_group_cols.append(col)
            else:
                numeric_cols.append(col)
                if n_unique == 2:
                    categorical_cols.append(col)
                    two_group_cols.append(col)
                elif 3 <= n_unique <= 15:
                    categorical_cols.append(col)
                    multi_group_cols.append(col)
        else:
            categorical_cols.append(col)
            if n_unique == 2:
                two_group_cols.append(col)
            elif n_unique >= 3:
                multi_group_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "two_group": two_group_cols,
        "multi_group": multi_group_cols,
    }


def _build_frequency_table(series: pd.Series) -> pd.DataFrame:
    """Build non-overlapping frequency distribution table matching make_freq_table in app.R."""
    vals = series.values
    n = len(vals)
    k = max(1, int(np.ceil(np.log2(n) + 1)))  # Sturges' rule
    counts, edges = np.histogram(vals, bins=k)
    labels = [
        f"{edges[i]:.2f} - <{edges[i + 1]:.2f}" if i < len(counts) - 1 else f"{edges[i]:.2f} - {edges[i + 1]:.2f}"
        for i in range(len(counts))
    ]
    freq_df = pd.DataFrame(
        {
            "Kelas Interval": labels,
            "Batas Bawah": np.round(edges[:-1], 3),
            "Batas Atas": np.round(edges[1:], 3),
            "Frekuensi": counts,
            "Persen (%)": np.round(counts / n * 100, 2),
            "Frekuensi Kumulatif": np.cumsum(counts),
        }
    )
    return freq_df


def _pan_scheme(k: int, center: float, spread: float) -> tuple[list[float], list[str], list[str]]:
    """Generate 2/3/4/5-tier categorization scheme matching category_scheme() in app.R."""
    if k == 2:
        cuts = [-np.inf, center, np.inf]
        labels = ["Rendah", "Tinggi"]
        criteria = [f"X < {center:.3f}", f"X >= {center:.3f}"]
    elif k == 3:
        cuts = [-np.inf, center - spread, center + spread, np.inf]
        labels = ["Rendah", "Sedang", "Tinggi"]
        criteria = [
            f"X < {center - spread:.3f}",
            f"{center - spread:.3f} <= X < {center + spread:.3f}",
            f"X >= {center + spread:.3f}",
        ]
    elif k == 4:
        cuts = [-np.inf, center - spread, center, center + spread, np.inf]
        labels = ["Sangat Rendah", "Rendah", "Tinggi", "Sangat Tinggi"]
        criteria = [
            f"X < {center - spread:.3f}",
            f"{center - spread:.3f} <= X < {center:.3f}",
            f"{center:.3f} <= X < {center + spread:.3f}",
            f"X >= {center + spread:.3f}",
        ]
    else:
        cuts = [
            -np.inf,
            center - 1.5 * spread,
            center - 0.5 * spread,
            center + 0.5 * spread,
            center + 1.5 * spread,
            np.inf,
        ]
        labels = ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
        criteria = [
            f"X < {center - 1.5 * spread:.3f}",
            f"{center - 1.5 * spread:.3f} <= X < {center - 0.5 * spread:.3f}",
            f"{center - 0.5 * spread:.3f} <= X < {center + 0.5 * spread:.3f}",
            f"{center + 0.5 * spread:.3f} <= X < {center + 1.5 * spread:.3f}",
            f"X >= {center + 1.5 * spread:.3f}",
        ]
    return cuts, labels, criteria


def run_profile_and_risk(
    df: pd.DataFrame,
    column: str,
    method: str = "PAK (Acuan Klinis Medis)",
    k_categories: int = 3,
    ideal_min: float | None = None,
    ideal_max: float | None = None,
) -> dict[str, Any]:
    """Compute full descriptive stats (app.R make_desc_stats), frequency table, and PAN/PAK categorization."""
    raw_col = pd.to_numeric(df[column], errors="coerce")
    n_missing = int(raw_col.isna().sum())
    series = raw_col.dropna()
    if series.empty:
        raise ValueError(f"Kolom '{column}' tidak memiliki data numerik yang valid.")

    n_valid = len(series)
    mean_val = float(series.mean())
    median_val = float(series.median())
    mode_series = series.mode()
    mode_val = float(mode_series.iloc[0]) if not mode_series.empty else np.nan
    sd_val = float(series.std(ddof=1))
    var_val = float(series.var(ddof=1))
    min_val = float(series.min())
    max_val = float(series.max())
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    p10 = float(series.quantile(0.10))
    p90 = float(series.quantile(0.90))
    iqr_val = q3 - q1
    range_val = max_val - min_val
    se_val = sd_val / np.sqrt(n_valid) if n_valid > 1 else np.nan
    t_crit = float(stats.t.ppf(0.975, df=n_valid - 1)) if n_valid > 1 else np.nan
    ci_low = mean_val - t_crit * se_val if n_valid > 1 else np.nan
    ci_high = mean_val + t_crit * se_val if n_valid > 1 else np.nan
    cv_pct = (sd_val / mean_val * 100) if abs(mean_val) > 1e-12 else np.nan
    skew_val = float(stats.skew(series, bias=False))
    kurt_val = float(stats.kurtosis(series, fisher=True, bias=False))

    stats_df = pd.DataFrame(
        [
            {"Metrik": "N Valid", "Nilai": f"{n_valid:,}"},
            {"Metrik": "Missing", "Nilai": f"{n_missing:,}"},
            {"Metrik": "Mean (Rata-rata)", "Nilai": f"{mean_val:.3f}"},
            {"Metrik": "Median (Q2)", "Nilai": f"{median_val:.3f}"},
            {"Metrik": "Mode (Modus)", "Nilai": f"{mode_val:.3f}" if not pd.isna(mode_val) else "N/A"},
            {"Metrik": "Standar Deviasi (SD)", "Nilai": f"{sd_val:.3f}"},
            {"Metrik": "Variance", "Nilai": f"{var_val:.3f}"},
            {"Metrik": "Standard Error (SE)", "Nilai": f"{se_val:.4f}"},
            {"Metrik": "95% CI Mean", "Nilai": f"[{ci_low:.3f}, {ci_high:.3f}]"},
            {"Metrik": "CV (%)", "Nilai": f"{cv_pct:.2f}%" if not pd.isna(cv_pct) else "N/A"},
            {"Metrik": "Minimum", "Nilai": f"{min_val:.3f}"},
            {"Metrik": "P10 / Q1 / Q3 / P90", "Nilai": f"{p10:.2f} / {q1:.2f} / {q3:.2f} / {p90:.2f}"},
            {"Metrik": "Maksimum", "Nilai": f"{max_val:.3f}"},
            {"Metrik": "Range (Max - Min)", "Nilai": f"{range_val:.3f}"},
            {"Metrik": "IQR (Q3 - Q1)", "Nilai": f"{iqr_val:.3f}"},
            {"Metrik": "Skewness", "Nilai": f"{skew_val:.3f}"},
            {"Metrik": "Excess Kurtosis", "Nilai": f"{kurt_val:.3f}"},
        ]
    )

    freq_df = _build_frequency_table(series)

    used_method = method
    cutoff_note = ""
    if "Skor Ideal Teoretis" in method:
        if ideal_min is None or ideal_max is None or not (np.isfinite(ideal_min) and np.isfinite(ideal_max)):
            raise ValueError("Masukkan skor minimum dan maksimum teoretis untuk metode PAK.")
        if ideal_max <= ideal_min:
            raise ValueError("Skor maksimum teoretis harus lebih besar dari skor minimum teoretis.")
        mi = (ideal_min + ideal_max) / 2.0
        sdi = (ideal_max - ideal_min) / 6.0
        used_method = f"PAK ({k_categories} Kategori — Mi = {mi:.3f}, SDi = {sdi:.3f})"
        cutoff_note = (
            f"PAK Acuan Kriteria (app.R / Azwar): Skor Min Teoretis = {ideal_min:.2f}, Max Teoretis = {ideal_max:.2f} "
            f"-> Mean ideal (Mi) = {mi:.3f}, SD ideal (SDi) = {sdi:.3f}."
        )
        cuts, labels, criteria_list = _pan_scheme(k_categories, mi, sdi)
        categories = pd.cut(series, bins=cuts, labels=labels, right=False, include_lowest=True)
    elif "Acuan Klinis" in method and column in CLINICAL_CUTOFFS and k_categories == 3:
        cfg = CLINICAL_CUTOFFS[column]
        categories = pd.cut(series, bins=cfg["bins"], labels=cfg["labels"], right=True)
        criteria_list = cfg["labels"]
        cutoff_note = cfg["note"]
    else:
        if "Acuan Klinis" in method and (column not in CLINICAL_CUTOFFS or k_categories != 3):
            used_method = f"PAN ({k_categories} Kategori — Mean ± SD)"
            cutoff_note = (
                f"Kolom '{column}' ({k_categories} kategori) menggunakan fallback PAN "
                f"(Mean empiris = {mean_val:.2f}, SD = {sd_val:.2f})."
            )
        else:
            used_method = f"PAN ({k_categories} Kategori — Mean ± SD)"
            cutoff_note = (
                f"Pengelompokan PAN {k_categories} kategori berbasis Azwar/app.R "
                f"(Mean = {mean_val:.2f}, SD = {sd_val:.2f})."
            )

        if sd_val == 0:
            categories = pd.Series(["Sedang"] * n_valid, index=series.index)
            criteria_list = ["Semua nilai sama"]
        else:
            cuts, labels, criteria_list = _pan_scheme(k_categories, mean_val, sd_val)
            categories = pd.cut(series, bins=cuts, labels=labels, right=False, include_lowest=True)

    cat_counts = categories.value_counts(sort=False)
    cat_df = pd.DataFrame(
        {
            "Kategori Risiko": cat_counts.index.astype(str),
            "Kriteria": criteria_list[: len(cat_counts)] if len(criteria_list) == len(cat_counts) else cat_counts.index.astype(str),
            "Frekuensi": cat_counts.values,
            "Persentase (%)": np.round(cat_counts.values / n_valid * 100, 2),
        }
    )

    detail_cols: dict[str, Any] = {}
    if "Patient ID" in df.columns:
        detail_cols["Patient ID"] = df.loc[series.index, "Patient ID"].values
    detail_cols["Skor"] = series.values
    detail_cols["Kategori"] = categories.astype(str).values
    detail_df = pd.DataFrame(detail_cols)

    skew_desc = (
        "cenderung simetris"
        if abs(skew_val) < 0.5
        else ("menceng kanan (positif)" if skew_val > 0 else "menceng kiri (negatif)")
    )
    top_cat = cat_df.sort_values("Frekuensi", ascending=False).iloc[0]
    narrative = (
        f"Parameter **{column}** pada **{n_valid:,}** pasien memiliki rata-rata **{mean_val:.2f}** "
        f"(95% CI [{ci_low:.2f}, {ci_high:.2f}], SD = **{sd_val:.2f}**), median **{median_val:.2f}** "
        f"(IQR = **{iqr_val:.2f}**), dengan rentang **{min_val:.2f}** hingga **{max_val:.2f}**. "
        f"Sebaran data {skew_desc} (skewness = {skew_val:.3f}, excess kurtosis = {kurt_val:.3f}). "
        f"Berdasarkan metode **{used_method}**, mayoritas pasien berada pada kategori "
        f"**{top_cat['Kategori Risiko']}** sebanyak **{int(top_cat['Frekuensi']):,} pasien ({top_cat['Persentase (%)']:.2f}%)**."
    )

    return {
        "series": series,
        "categories": categories.astype(str),
        "stats_df": stats_df,
        "freq_df": freq_df,
        "risk_df": cat_df,
        "detail_df": detail_df,
        "cutoff_note": cutoff_note,
        "narrative": narrative,
    }


def run_assumption_tests(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str | None = None,
) -> dict[str, Any]:
    """Run Shapiro-Wilk normality test (capped at 5,000 random samples) and Levene's test."""
    cols = [numeric_col] + ([group_col] if group_col else [])
    clean_df = df[cols].dropna().copy()
    clean_df[numeric_col] = pd.to_numeric(clean_df[numeric_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[numeric_col])

    if len(clean_df) < 3:
        raise ValueError("Minimal diperlukan 3 observasi valid untuk uji normalitas.")

    total_n = len(clean_df)
    sampled = total_n > 5000
    sw_sample = clean_df[numeric_col].sample(n=5000, random_state=42) if sampled else clean_df[numeric_col]

    sw_stat, sw_p = stats.shapiro(sw_sample)
    is_normal = sw_p >= 0.05

    osm, osr = stats.probplot(sw_sample, dist="norm", fit=False)
    qq_df = pd.DataFrame({"Theoretical Quantiles": osm, "Sample Quantiles": osr})

    rows = [
        {
            "Uji Asumsi": "Shapiro-Wilk (Normalitas)",
            "Statistik": f"W = {sw_stat:.4f}",
            "df": f"n = {len(sw_sample):,}",
            "p-value": format_p_value(sw_p),
            "Kesimpulan (alpha = 0.05)": "Normal (p >= 0.05)" if is_normal else "Tidak Normal (p < 0.05)",
            "Catatan": f"Sampel acak 5.000 dari {total_n:,} baris" if sampled else f"Seluruh data (n = {total_n:,})",
        }
    ]

    levene_p = None
    is_homogenous = None
    if group_col:
        clean_df[group_col] = clean_df[group_col].astype(str)
        groups = [g[numeric_col].values for _, g in clean_df.groupby(group_col) if len(g) >= 2]
        if len(groups) < 2:
            raise ValueError(f"Variabel kelompok '{group_col}' harus memiliki minimal 2 kategori dengan >= 2 data.")
        lev_stat, lev_p = stats.levene(*groups, center="median")
        levene_p = float(lev_p)
        is_homogenous = lev_p >= 0.05
        df1 = len(groups) - 1
        df2 = sum(len(g) for g in groups) - len(groups)
        rows.append(
            {
                "Uji Asumsi": f"Levene's Test ({group_col})",
                "Statistik": f"F = {lev_stat:.4f}",
                "df": f"df1 = {df1}, df2 = {df2:,}",
                "p-value": format_p_value(lev_p),
                "Kesimpulan (alpha = 0.05)": "Homogen (p >= 0.05)" if is_homogenous else "Tidak Homogen (p < 0.05)",
                "Catatan": f"{len(groups)} kelompok, total n = {total_n:,}",
            }
        )

    summary_df = pd.DataFrame(rows)

    sample_note = (
        f"Karena jumlah baris ({total_n:,}) melebihi batas 5.000 pada uji Shapiro-Wilk, pengujian dilakukan pada **5.000 sampel acak**. "
        if sampled
        else ""
    )
    norm_text = (
        f"Data **{numeric_col}** berdistribusi **normal** (W = {sw_stat:.4f}, p = {format_p_value(sw_p)})."
        if is_normal
        else f"Data **{numeric_col}** **tidak berdistribusi normal** (W = {sw_stat:.4f}, p = {format_p_value(sw_p)})."
    )
    if levene_p is not None:
        hom_text = (
            f" Varians antar kelompok **{group_col}** bersifat **homogen** (p = {format_p_value(levene_p)})."
            if is_homogenous
            else f" Varians antar kelompok **{group_col}** **tidak homogen** (p = {format_p_value(levene_p)})."
        )
    else:
        hom_text = ""

    if is_normal and (is_homogenous is None or is_homogenous):
        rec = "Rekomendasi: Gunakan **uji parametrik** (*Independent t-test* / *One-Way ANOVA* / *Pearson*)."
    else:
        rec = "Rekomendasi: Pertimbangkan **uji nonparametrik** (*Mann-Whitney U* / *Kruskal-Wallis* / *Spearman*) atau Welch's test bila varians tidak homogen."

    return {
        "clean_df": clean_df,
        "summary_df": summary_df,
        "qq_df": qq_df,
        "narrative": f"{sample_note}{norm_text}{hom_text} {rec}",
    }


def _interpret_cohens_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "sangat kecil"
    if ad < 0.5:
        return "kecil"
    if ad < 0.8:
        return "sedang"
    return "besar"


def run_two_group_comparison(df: pd.DataFrame, numeric_col: str, group_col: str) -> dict[str, Any]:
    """Run Independent t-test (df, 95% CI, Cohen's d) and Mann-Whitney U (U, p, Rank-Biserial r)."""
    clean_df = df[[numeric_col, group_col]].dropna().copy()
    clean_df[numeric_col] = pd.to_numeric(clean_df[numeric_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[numeric_col])
    clean_df[group_col] = clean_df[group_col].astype(str)

    unique_groups = sorted(clean_df[group_col].unique())
    if len(unique_groups) != 2:
        raise ValueError(
            f"Uji komparasi 2 kelompok membutuhkan tepat 2 kategori pada '{group_col}', "
            f"ditemukan {len(unique_groups)} kategori: {unique_groups}."
        )

    g1_name, g2_name = unique_groups[0], unique_groups[1]
    g1 = clean_df.loc[clean_df[group_col] == g1_name, numeric_col].values
    g2 = clean_df.loc[clean_df[group_col] == g2_name, numeric_col].values

    if len(g1) < 2 or len(g2) < 2:
        raise ValueError("Masing-masing kelompok harus memiliki minimal 2 observasi.")

    _, lev_p = stats.levene(g1, g2, center="median")
    equal_var = lev_p >= 0.05

    res_t = stats.ttest_ind(g1, g2, equal_var=equal_var)
    t_stat, t_p = float(res_t.statistic), float(res_t.pvalue)
    t_df = float(res_t.df)
    ci = res_t.confidence_interval(confidence_level=0.95)
    ci_low, ci_high = float(ci.low), float(ci.high)

    u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    n1, n2 = len(g1), len(g2)
    rank_biserial = 1.0 - (2.0 * float(u_stat)) / (n1 * n2)

    s1, s2 = float(np.std(g1, ddof=1)), float(np.std(g2, ddof=1))
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohens_d = float((np.mean(g1) - np.mean(g2)) / s_pooled) if s_pooled > 0 else 0.0

    group_df = pd.DataFrame(
        [
            {
                "Kelompok": g1_name,
                "N": n1,
                "Mean": round(float(np.mean(g1)), 3),
                "SD": round(s1, 3),
                "Median": round(float(np.median(g1)), 3),
                "IQR": round(float(np.percentile(g1, 75) - np.percentile(g1, 25)), 3),
            },
            {
                "Kelompok": g2_name,
                "N": n2,
                "Mean": round(float(np.mean(g2)), 3),
                "SD": round(s2, 3),
                "Median": round(float(np.median(g2)), 3),
                "IQR": round(float(np.percentile(g2, 75) - np.percentile(g2, 25)), 3),
            },
        ]
    )

    test_df = pd.DataFrame(
        [
            {
                "Metode Uji": f"Independent t-test ({'Student' if equal_var else 'Welch'})",
                "Statistik": f"t = {t_stat:.4f}",
                "df": f"{t_df:.2f}",
                "p-value": format_p_value(t_p),
                "95% CI Beda Mean": f"[{ci_low:.3f}, {ci_high:.3f}]",
                "Effect Size": f"Cohen's d = {cohens_d:.4f} ({_interpret_cohens_d(cohens_d)})",
                "Signifikansi (alpha = 0.05)": "Signifikan" if t_p < 0.05 else "Tidak Signifikan",
            },
            {
                "Metode Uji": "Mann-Whitney U (Nonparametrik)",
                "Statistik": f"U = {u_stat:,.1f}",
                "df": "-",
                "p-value": format_p_value(u_p),
                "95% CI Beda Mean": "-",
                "Effect Size": f"Rank-Biserial r = {rank_biserial:.4f}",
                "Signifikansi (alpha = 0.05)": "Signifikan" if u_p < 0.05 else "Tidak Signifikan",
            },
        ]
    )

    sig_t = "berbeda secara signifikan" if t_p < 0.05 else "tidak berbeda secara signifikan"
    narrative = (
        f"Perbandingan **{numeric_col}** antara kelompok **{g1_name}** (Mean = {np.mean(g1):.2f}, SD = {s1:.2f}) "
        f"dan **{g2_name}** (Mean = {np.mean(g2):.2f}, SD = {s2:.2f}) menunjukkan bahwa kedua kelompok **{sig_t}** "
        f"berdasarkan uji *Independent t-test* (t({t_df:.1f}) = {t_stat:.3f}, p = {format_p_value(t_p)}, "
        f"95% CI [{ci_low:.2f}, {ci_high:.2f}]) maupun *Mann-Whitney U* (U = {u_stat:,.0f}, p = {format_p_value(u_p)}, "
        f"Rank-Biserial r = {rank_biserial:.3f}). Besaran efek *Cohen's d* = **{cohens_d:.3f}** "
        f"(kategori **{_interpret_cohens_d(cohens_d)}**)."
    )

    return {
        "clean_df": clean_df,
        "group_df": group_df,
        "test_df": test_df,
        "narrative": narrative,
    }


def run_multi_group_comparison(df: pd.DataFrame, numeric_col: str, group_col: str) -> dict[str, Any]:
    """Run One-Way ANOVA (df1, df2, Eta2), Kruskal-Wallis (Epsilon2), and Tukey HSD."""
    clean_df = df[[numeric_col, group_col]].dropna().copy()
    clean_df[numeric_col] = pd.to_numeric(clean_df[numeric_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[numeric_col])
    clean_df[group_col] = clean_df[group_col].astype(str)

    unique_groups = sorted(clean_df[group_col].unique())
    k = len(unique_groups)
    if k < 3:
        raise ValueError(
            f"Uji komparasi >2 kelompok membutuhkan minimal 3 kategori pada '{group_col}', "
            f"ditemukan {k} kategori: {unique_groups}."
        )

    grouped_arrays = [clean_df.loc[clean_df[group_col] == g, numeric_col].values for g in unique_groups]
    if any(len(arr) < 2 for arr in grouped_arrays):
        raise ValueError("Setiap kategori kelompok harus memiliki minimal 2 observasi.")

    n_total = len(clean_df)
    f_stat, f_p = stats.f_oneway(*grouped_arrays)
    h_stat, h_p = stats.kruskal(*grouped_arrays)

    df1 = k - 1
    df2 = n_total - k
    grand_mean = clean_df[numeric_col].mean()
    ss_between = sum(len(arr) * (np.mean(arr) - grand_mean) ** 2 for arr in grouped_arrays)
    ss_total = float(((clean_df[numeric_col] - grand_mean) ** 2).sum())
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
    epsilon_sq = max(0.0, float((h_stat - k + 1) / (n_total - k))) if n_total > k else 0.0

    eta_label = "kecil" if eta_sq < 0.06 else ("sedang" if eta_sq < 0.14 else "besar")

    group_rows = []
    for g, arr in zip(unique_groups, grouped_arrays):
        group_rows.append(
            {
                "Kelompok": g,
                "N": len(arr),
                "Mean": round(float(np.mean(arr)), 3),
                "SD": round(float(np.std(arr, ddof=1)), 3),
                "Median": round(float(np.median(arr)), 3),
                "IQR": round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 3),
            }
        )
    group_df = pd.DataFrame(group_rows)

    test_df = pd.DataFrame(
        [
            {
                "Metode Uji": "One-Way ANOVA",
                "Statistik": f"F = {f_stat:.4f}",
                "df": f"df1 = {df1}, df2 = {df2:,}",
                "p-value": format_p_value(f_p),
                "Effect Size": f"Eta-squared = {eta_sq:.4f} ({eta_label})",
                "Signifikansi (alpha = 0.05)": "Signifikan" if f_p < 0.05 else "Tidak Signifikan",
            },
            {
                "Metode Uji": "Kruskal-Wallis (Nonparametrik)",
                "Statistik": f"H = {h_stat:.4f}",
                "df": f"df = {df1}",
                "p-value": format_p_value(h_p),
                "Effect Size": f"Epsilon-squared = {epsilon_sq:.4f}",
                "Signifikansi (alpha = 0.05)": "Signifikan" if h_p < 0.05 else "Tidak Signifikan",
            },
        ]
    )

    tukey = pairwise_tukeyhsd(endog=clean_df[numeric_col], groups=clean_df[group_col], alpha=0.05)
    tukey_data = tukey._results_table.data
    tukey_df = pd.DataFrame(tukey_data[1:], columns=tukey_data[0])

    sig_pairs = tukey_df[tukey_df["reject"] == True]  # noqa: E712
    sig_text = "terdapat perbedaan rata-rata yang signifikan" if f_p < 0.05 else "tidak terdapat perbedaan yang signifikan"
    pair_note = (
        f" Uji lanjut *Post-Hoc Tukey HSD* menemukan **{len(sig_pairs)} pasangan kelompok** yang berbeda signifikan."
        if len(sig_pairs) > 0
        else " Uji lanjut *Post-Hoc Tukey HSD* tidak menemukan pasangan kelompok yang berbeda signifikan."
    )

    narrative = (
        f"Hasil *One-Way ANOVA* pada **{numeric_col}** antar **{k} kelompok {group_col}** "
        f"menunjukkan bahwa {sig_text} (F({df1}, {df2:,}) = {f_stat:.3f}, p = {format_p_value(f_p)}; "
        f"Kruskal-Wallis H({df1}) = {h_stat:.3f}, p = {format_p_value(h_p)}, $\\varepsilon^2$ = {epsilon_sq:.4f}). "
        f"Besaran efek $\\eta^2$ = **{eta_sq:.4f}** (kategori **{eta_label}**).{pair_note}"
    )

    return {
        "clean_df": clean_df,
        "group_df": group_df,
        "test_df": test_df,
        "tukey_df": tukey_df,
        "narrative": narrative,
    }


def run_chi_square_association(df: pd.DataFrame, row_col: str, col_col: str) -> dict[str, Any]:
    """Run Chi-Square test of independence, Minimum Expected, and Cramer's V."""
    if row_col == col_col:
        raise ValueError("Pilih dua variabel kategorik yang berbeda untuk uji asosiasi.")

    clean_df = df[[row_col, col_col]].dropna().copy()
    clean_df[row_col] = clean_df[row_col].astype(str)
    clean_df[col_col] = clean_df[col_col].astype(str)

    ct = pd.crosstab(clean_df[row_col], clean_df[col_col])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise ValueError("Masing-masing variabel kategorik harus memiliki minimal 2 kategori.")

    chi2_stat, p_val, dof, expected = stats.chi2_contingency(ct)
    n_total = int(ct.values.sum())
    min_dim = min(ct.shape[0] - 1, ct.shape[1] - 1)
    cramers_v = float(np.sqrt(chi2_stat / (n_total * min_dim))) if (n_total * min_dim) > 0 else 0.0
    min_expected = float(expected.min())

    v_label = "lemah" if cramers_v < 0.1 else ("sedang" if cramers_v < 0.3 else "kuat")
    low_expected = int((expected < 5).sum())

    test_df = pd.DataFrame(
        [
            {"Metrik": "Chi-Square (Pearson)", "Nilai": f"{chi2_stat:.4f} (df = {dof})"},
            {"Metrik": "p-value", "Nilai": format_p_value(p_val)},
            {"Metrik": "Cramér's V", "Nilai": f"{cramers_v:.4f} (Asosiasi {v_label})"},
            {"Metrik": "Minimum Expected Count", "Nilai": f"{min_expected:.2f}"},
            {"Metrik": "Sel dengan Expected < 5", "Nilai": f"{low_expected} dari {expected.size} sel"},
            {"Metrik": "Total Observasi (N)", "Nilai": f"{n_total:,}"},
        ]
    )

    ct_display = ct.copy()
    ct_display.index.name = row_col
    ct_reset = ct_display.reset_index()

    assoc_text = "terdapat asosiasi yang signifikan secara statistik" if p_val < 0.05 else "tidak terdapat asosiasi yang signifikan secara statistik"
    narrative = (
        f"Hasil uji *Chi-Square* antara **{row_col}** dan **{col_col}** pada **{n_total:,}** pasien menunjukkan bahwa "
        f"{assoc_text} ($\\chi^2$({dof}) = {chi2_stat:.3f}, p = {format_p_value(p_val)}). "
        f"Kekuatan asosiasi *Cramér's V* sebesar **{cramers_v:.4f}** (kategori **{v_label}**), "
        f"dengan *minimum expected count* = {min_expected:.2f}."
    )

    return {
        "clean_df": clean_df,
        "crosstab": ct,
        "crosstab_df": ct_reset,
        "test_df": test_df,
        "narrative": narrative,
    }


def run_correlation(df: pd.DataFrame, columns: list[str], method: str = "Pearson") -> dict[str, Any]:
    """Compute Pearson, Spearman, or Kendall correlation matrix and pairwise p-values."""
    if len(columns) < 2:
        raise ValueError("Pilih minimal 2 variabel numerik untuk analisis korelasi.")

    num_df = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(num_df) < 3:
        raise ValueError("Minimal diperlukan 3 baris data lengkap untuk menghitung korelasi.")

    m_low = method.lower()
    corr_matrix = num_df.corr(method=m_low)
    coef_label = "r" if m_low == "pearson" else ("rho" if m_low == "spearman" else "tau")

    pair_rows = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            c1, c2 = columns[i], columns[j]
            if m_low == "pearson":
                r_val, p_val = stats.pearsonr(num_df[c1], num_df[c2])
            elif m_low == "spearman":
                r_val, p_val = stats.spearmanr(num_df[c1], num_df[c2])
            else:
                r_val, p_val = stats.kendalltau(num_df[c1], num_df[c2])

            ar = abs(r_val)
            strength = (
                "sangat lemah"
                if ar < 0.1
                else ("lemah" if ar < 0.3 else ("sedang" if ar < 0.5 else ("kuat" if ar < 0.7 else "sangat kuat")))
            )
            direction = "positif" if r_val >= 0 else "negatif"
            pair_rows.append(
                {
                    "Variabel 1": c1,
                    "Variabel 2": c2,
                    f"Koefisien ({coef_label})": round(float(r_val), 4),
                    "p-value": format_p_value(p_val),
                    "Arah & Kekuatan": f"{direction.capitalize()}, {strength}",
                    "Signifikansi (alpha = 0.05)": "Signifikan" if p_val < 0.05 else "Tidak Signifikan",
                }
            )

    pairs_df = pd.DataFrame(pair_rows)
    sig_count = int((pairs_df["Signifikansi (alpha = 0.05)"] == "Signifikan").sum())
    strongest = pairs_df.iloc[pairs_df.iloc[:, 2].abs().argmax()]
    narrative = (
        f"Analisis korelasi **{method}** pada **{len(num_df):,}** pasien menghasilkan **{sig_count} dari {len(pairs_df)}** "
        f"pasangan variabel yang berkorelasi signifikan (alpha = 0.05). Hubungan terkuat ditemukan antara "
        f"**{strongest['Variabel 1']}** dan **{strongest['Variabel 2']}** dengan koefisien **{strongest.iloc[2]:.4f}** "
        f"({strongest['Arah & Kekuatan']}, p = {strongest['p-value']})."
    )

    return {
        "corr_matrix": corr_matrix,
        "pairs_df": pairs_df,
        "narrative": narrative,
    }


def run_multiple_regression(df: pd.DataFrame, target_col: str, predictor_cols: list[str]) -> dict[str, Any]:
    """Run OLS multiple linear regression with Std_Beta, RMSE, Residual_SD, residual stats, and VIF."""
    predictors = [c for c in predictor_cols if c != target_col]
    if not predictors:
        raise ValueError("Pilih minimal 1 variabel prediktor yang berbeda dari variabel target.")

    reg_df = df[[target_col] + predictors].apply(pd.to_numeric, errors="coerce").dropna()
    if len(reg_df) <= len(predictors) + 2:
        raise ValueError("Jumlah observasi tidak mencukupi untuk pemodelan regresi.")

    y = reg_df[target_col]
    X = sm.add_constant(reg_df[predictors], has_constant="add")
    model = sm.OLS(y, X).fit()

    # Standardized Beta (matching zmod in app.R)
    z_df = (reg_df - reg_df.mean()) / reg_df.std(ddof=1)
    z_X = sm.add_constant(z_df[predictors], has_constant="add")
    z_model = sm.OLS(z_df[target_col], z_X).fit()

    conf = model.conf_int()
    vifs = []
    if len(predictors) >= 2:
        X_vals = X.values
        for idx in range(1, X_vals.shape[1]):
            vifs.append(round(float(variance_inflation_factor(X_vals, idx)), 3))
    else:
        vifs = [1.0]

    coef_rows = []
    for idx, name in enumerate(model.params.index):
        vif_val = "-" if name == "const" else vifs[idx - 1]
        std_beta = "-" if name == "const" else round(float(z_model.params[name]), 4)
        coef_rows.append(
            {
                "Variabel": "Intercept (Konstanta)" if name == "const" else name,
                "Koefisien (B)": round(float(model.params[name]), 4),
                "Std. Error": round(float(model.bse[name]), 4),
                "Std. Beta": std_beta,
                "t-stat": round(float(model.tvalues[name]), 4),
                "p-value": format_p_value(float(model.pvalues[name])),
                "CI 95% Bawah": round(float(conf.iloc[idx, 0]), 4),
                "CI 95% Atas": round(float(conf.iloc[idx, 1]), 4),
                "VIF": vif_val,
            }
        )
    coef_df = pd.DataFrame(coef_rows)

    rmse_val = float(np.sqrt(np.mean(model.resid**2)))
    resid_sd = float(model.resid.std(ddof=1))

    summary_df = pd.DataFrame(
        [
            {"Metrik Model": "Jumlah Observasi (N)", "Nilai": f"{int(model.nobs):,}"},
            {"Metrik Model": "R-squared (R²)", "Nilai": f"{model.rsquared:.4f} ({model.rsquared * 100:.2f}%)"},
            {"Metrik Model": "Adjusted R-squared", "Nilai": f"{model.rsquared_adj:.4f}"},
            {"Metrik Model": "F-statistic", "Nilai": f"{model.fvalue:.4f} (df1 = {int(model.df_model)}, df2 = {int(model.df_resid):,})"},
            {"Metrik Model": "p-value (Uji Simultan F)", "Nilai": format_p_value(float(model.f_pvalue))},
            {"Metrik Model": "RMSE", "Nilai": f"{rmse_val:.4f}"},
            {"Metrik Model": "Residual Std. Deviation", "Nilai": f"{resid_sd:.4f}"},
        ]
    )

    resid_stats_df = pd.DataFrame(
        [
            {
                "Mean": round(float(model.resid.mean()), 4),
                "SD": round(resid_sd, 4),
                "Min": round(float(model.resid.min()), 4),
                "Max": round(float(model.resid.max()), 4),
                "Skewness": round(float(stats.skew(model.resid, bias=False)), 4),
                "Excess Kurtosis": round(float(stats.kurtosis(model.resid, fisher=True, bias=False)), 4),
            }
        ]
    )

    sig_preds = coef_df[(coef_df["Variabel"] != "Intercept (Konstanta)") & (model.pvalues.values < 0.05)]["Variabel"].tolist()
    sig_text = (
        f"Prediktor yang berpengaruh signifikan secara parsial adalah: **{', '.join(sig_preds)}**."
        if sig_preds
        else "Tidak ada prediktor yang berpengaruh signifikan secara parsial pada alpha = 0.05."
    )
    sim_text = "signifikan secara simultan" if model.f_pvalue < 0.05 else "tidak signifikan secara simultan"

    narrative = (
        f"Model regresi linear berganda untuk memprediksi **{target_col}** menggunakan {len(predictors)} prediktor "
        f"**{sim_text}** (F({int(model.df_model)}, {int(model.df_resid):,}) = {model.fvalue:.3f}, p = {format_p_value(float(model.f_pvalue))}). "
        f"Nilai **R² = {model.rsquared:.4f}** (Adjusted R² = {model.rsquared_adj:.4f}, RMSE = {rmse_val:.3f}) menunjukkan bahwa "
        f"model mampu menjelaskan **{model.rsquared * 100:.2f}%** variasi pada {target_col}. {sig_text}"
    )

    return {
        "summary_df": summary_df,
        "coef_df": coef_df,
        "resid_stats_df": resid_stats_df,
        "fitted": model.fittedvalues,
        "residuals": model.resid,
        "narrative": narrative,
    }


def export_results_to_excel(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Export multiple result DataFrames into an in-memory multi-sheet .xlsx workbook."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheets.items():
            safe_name = sheet_name[:31].replace("/", "_").replace("\\", "_")
            sheet_df.to_excel(writer, sheet_name=safe_name, index=False)
    buffer.seek(0)
    return buffer.getvalue()
