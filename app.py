"""
ANALISIS CLUSTERING CACAT PRODUK INDUSTRI MANUFAKTUR
Aplikasi Streamlit - Deployment Project (Pertemuan ke-12)

Fitur:
1. Eksplorasi data cacat produk (defects_data.csv)
2. Rekayasa fitur per produk & clustering K-Means
3. Diagnostik model (elbow method & silhouette score)
4. Interpretasi hasil clustering & insight bisnis otomatis
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------------
# IDENTITAS PENYUSUN
# --------------------------------------------------------------------------------
NAMA = "Muhammad Ulul Azmi Nugroho"
NIM = "E12.2024.01937"

# --------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Clustering Cacat Produk Manufaktur",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------
# THEME / CUSTOM CSS  -- industrial QC control-panel look
# --------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root{
    --bg:        #161B22;
    --panel:     #1F2733;
    --panel-2:   #232C3B;
    --line:      #313C4E;
    --text:      #E8ECF1;
    --muted:     #8A94A6;
    --amber:     #F2A93B;
    --alert:     #E4572E;
    --teal:      #4FA69A;
    --slate-blue:#5E7CE2;
}

html, body, [class*="css"]  { font-family: 'Space Grotesk', 'Segoe UI', sans-serif; }
.stApp { background-color: var(--bg); color: var(--text); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #12161D;
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Headings */
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; }
h1 { color: var(--text); font-weight: 700; }
h2, h3 { color: var(--amber); font-weight: 700; }

/* Rivet divider - industrial signature element */
.rivet-divider {
    display: flex; align-items: center; gap: 6px;
    margin: 0.4rem 0 1.2rem 0;
}
.rivet-divider .line {
    flex: 1; height: 2px;
    background: repeating-linear-gradient(90deg, var(--line) 0 10px, transparent 10px 16px);
}
.rivet-divider .bolt {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--amber); box-shadow: 0 0 0 3px rgba(242,169,59,0.15);
}

/* Metric / gauge card */
.gauge-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-left: 4px solid var(--amber);
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.gauge-card .label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
.gauge-card .value { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; color: var(--text); }
.gauge-card .sub   { color: var(--teal); font-size: 0.8rem; }

/* Cluster tag badges */
.tag-critical { background: rgba(228,87,46,0.18); color: var(--alert); border:1px solid var(--alert); padding:2px 10px; border-radius: 20px; font-size:0.78rem; font-weight:600;}
.tag-moderate { background: rgba(242,169,59,0.18); color: var(--amber); border:1px solid var(--amber); padding:2px 10px; border-radius: 20px; font-size:0.78rem; font-weight:600;}
.tag-good     { background: rgba(79,166,154,0.18); color: var(--teal); border:1px solid var(--teal); padding:2px 10px; border-radius: 20px; font-size:0.78rem; font-weight:600;}

/* Insight panel */
.insight-box {
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 14px;
}
.insight-box h4 { margin-top:0; color: var(--text); font-family:'IBM Plex Mono', monospace; font-size:0.95rem;}

/* Dataframe tweak */
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px; }

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background: var(--amber); color: #161B22; border: none; font-weight: 600; border-radius: 4px;
}

/* Radio nav in sidebar look like panel switches */
div[role="radiogroup"] label {
    background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
    padding: 8px 10px; margin-bottom: 6px; width: 100%;
}
</style>
""", unsafe_allow_html=True)


def rivet_divider():
    st.markdown(
        '<div class="rivet-divider"><div class="bolt"></div><div class="line"></div>'
        '<div class="bolt"></div></div>', unsafe_allow_html=True
    )


# --------------------------------------------------------------------------------
# DATA LOADING & FEATURE ENGINEERING
# --------------------------------------------------------------------------------
@st.cache_data
def load_raw_data(path="defects_data.csv"):
    df = pd.read_csv(path)
    df["defect_date"] = pd.to_datetime(df["defect_date"])
    return df


@st.cache_data
def build_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate defect records into one row per product (unit of clustering)."""
    agg = df.groupby("product_id").agg(
        total_defects=("defect_id", "count"),
        avg_repair_cost=("repair_cost", "mean"),
        total_repair_cost=("repair_cost", "sum"),
        max_repair_cost=("repair_cost", "max"),
    ).reset_index()

    sev = pd.crosstab(df["product_id"], df["severity"], normalize="index").add_prefix("pct_sev_")
    typ = pd.crosstab(df["product_id"], df["defect_type"], normalize="index").add_prefix("pct_type_")
    loc = pd.crosstab(df["product_id"], df["defect_location"], normalize="index").add_prefix("pct_loc_")
    insp = pd.crosstab(df["product_id"], df["inspection_method"], normalize="index").add_prefix("pct_insp_")

    features = (
        agg.merge(sev, on="product_id")
           .merge(typ, on="product_id")
           .merge(loc, on="product_id")
           .merge(insp, on="product_id")
    )
    return features


FEATURE_COLS = [
    "total_defects", "avg_repair_cost", "total_repair_cost", "max_repair_cost",
    "pct_sev_Critical", "pct_sev_Moderate", "pct_sev_Minor",
    "pct_type_Structural", "pct_type_Functional", "pct_type_Cosmetic",
    "pct_loc_Component", "pct_loc_Internal", "pct_loc_Surface",
    "pct_insp_Automated Testing", "pct_insp_Manual Testing", "pct_insp_Visual Inspection",
]


@st.cache_data
def scan_k(features: pd.DataFrame, k_min=2, k_max=8):
    X = StandardScaler().fit_transform(features[FEATURE_COLS].values)
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        sil = silhouette_score(X, km.labels_)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


@st.cache_data
def run_kmeans(features: pd.DataFrame, k: int):
    X = StandardScaler().fit_transform(features[FEATURE_COLS].values)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    labels = km.labels_
    sil = silhouette_score(X, labels)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    out = features.copy()
    out["cluster"] = labels
    out["pca_1"] = coords[:, 0]
    out["pca_2"] = coords[:, 1]
    return out, sil, pca.explained_variance_ratio_


def zlabel(z):
    if z >= 0.35:
        return "Tinggi"
    if z <= -0.35:
        return "Rendah"
    return "Sedang"


def generate_cluster_narrative(clustered: pd.DataFrame):
    """Auto-generate a plain-language business narrative for each cluster,
    based on standardized deviation of cluster mean vs overall mean."""
    overall_mean = clustered[FEATURE_COLS].mean()
    overall_std = clustered[FEATURE_COLS].std().replace(0, 1)

    narratives = {}
    for c in sorted(clustered["cluster"].unique()):
        sub = clustered[clustered["cluster"] == c]
        n = len(sub)
        cmean = sub[FEATURE_COLS].mean()
        z = (cmean - overall_mean) / overall_std

        freq_lvl = zlabel(z["total_defects"])
        cost_lvl = zlabel(z["total_repair_cost"])

        dom_sev = max(["Critical", "Moderate", "Minor"], key=lambda s: cmean[f"pct_sev_{s}"])
        dom_type = max(["Structural", "Functional", "Cosmetic"], key=lambda s: cmean[f"pct_type_{s}"])
        dom_loc = max(["Component", "Internal", "Surface"], key=lambda s: cmean[f"pct_loc_{s}"])
        dom_insp = max(["Automated Testing", "Manual Testing", "Visual Inspection"],
                        key=lambda s: cmean[f"pct_insp_{s}"])

        risk = "Tinggi" if (freq_lvl == "Tinggi" or cost_lvl == "Tinggi" or dom_sev == "Critical") else \
               ("Rendah" if (freq_lvl == "Rendah" and cost_lvl == "Rendah") else "Sedang")

        title = f"Frekuensi {freq_lvl} & Biaya {cost_lvl} — dominan cacat {dom_type}"

        rekomendasi = []
        if risk == "Tinggi":
            rekomendasi.append("Prioritaskan audit kualitas dan root-cause analysis untuk produk di klaster ini.")
        if dom_insp == "Manual Testing" and dom_sev == "Critical":
            rekomendasi.append("Pertimbangkan menambah cakupan Automated Testing agar cacat kritis terdeteksi lebih awal.")
        if dom_type == "Structural":
            rekomendasi.append("Tinjau ulang desain/proses produksi dan pemilihan material terkait cacat struktural.")
        if dom_type == "Functional":
            rekomendasi.append("Perkuat pengujian fungsional sebelum produk keluar dari lini produksi.")
        if dom_type == "Cosmetic":
            rekomendasi.append("Perbaikan finishing/handling untuk menekan cacat kosmetik yang berulang.")
        if cost_lvl == "Tinggi":
            rekomendasi.append("Evaluasi biaya perbaikan — pertimbangkan penggantian komponen vs perbaikan berulang.")
        if not rekomendasi:
            rekomendasi.append("Pertahankan praktik QC saat ini; jadikan klaster ini sebagai acuan (benchmark).")

        narratives[c] = {
            "n": n,
            "title": title,
            "risk": risk,
            "freq_lvl": freq_lvl,
            "cost_lvl": cost_lvl,
            "dom_sev": dom_sev,
            "dom_type": dom_type,
            "dom_loc": dom_loc,
            "dom_insp": dom_insp,
            "avg_defects": cmean["total_defects"],
            "avg_cost": cmean["avg_repair_cost"],
            "total_cost": cmean["total_repair_cost"],
            "rekomendasi": rekomendasi,
        }
    return narratives


def risk_tag(risk):
    cls = {"Tinggi": "tag-critical", "Sedang": "tag-moderate", "Rendah": "tag-good"}[risk]
    return f'<span class="{cls}">Risiko {risk}</span>'


# --------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ QC ANALYTICS")
st.sidebar.caption("Analisis Clustering Cacat Produk — Industri Manufaktur")
st.sidebar.markdown(
    f'<div style="background:var(--panel); border:1px solid var(--line); '
    f'border-left:3px solid var(--amber); border-radius:6px; padding:8px 12px; '
    f'margin-bottom:10px; font-size:0.82rem;">'
    f'<b style="color:var(--text);">{NAMA}</b><br>'
    f'<span style="color:var(--muted);">NIM {NIM}</span></div>',
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Beranda", "📊 Data & Eksplorasi", "🔍 Analisis Clustering", "💡 Interpretasi & Insight Bisnis", "ℹ️ Tentang Aplikasi"],
    label_visibility="collapsed",
)

df = load_raw_data()
features = build_product_features(df)

st.sidebar.divider()
k_choice = st.sidebar.slider("Jumlah Klaster (K)", min_value=2, max_value=8, value=3,
                              help="Jumlah klaster K-Means yang diterapkan pada level produk.")
st.sidebar.caption(f"Data: {len(df):,} catatan cacat • {features.shape[0]} produk unik")

clustered, sil_score, evr = run_kmeans(features, k_choice)
narratives = generate_cluster_narrative(clustered)

# --------------------------------------------------------------------------------
# IDENTITY STRIP — tampil di setiap halaman
# --------------------------------------------------------------------------------
st.markdown(
    f'<div style="text-align:right; color:var(--muted); font-size:0.8rem; '
    f'margin-bottom:6px;"><b style="color:var(--text);">{NAMA}</b> &nbsp;·&nbsp; NIM {NIM}</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# PAGE 1 — BERANDA
# --------------------------------------------------------------------------------
if page == "🏠 Beranda":
    st.markdown("# ⚙️ Analisis Clustering Cacat Produk Industri Manufaktur")
    st.caption("Real-world use case · Deployment Streamlit · Pertemuan ke-12")
    rivet_divider()

    st.write(
        "Aplikasi ini mengelompokkan **produk** berdasarkan karakteristik cacat yang tercatat selama "
        "proses produksi & inspeksi kualitas — meliputi frekuensi cacat, biaya perbaikan, tingkat "
        "keparahan (severity), jenis cacat, lokasi cacat, dan metode inspeksi. Tujuannya membantu tim "
        "Quality Control (QC) memprioritaskan produk mana yang paling butuh perhatian."
    )

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Total Catatan Cacat", f"{len(df):,}", "baris data mentah"),
        ("Produk Unik", f"{features.shape[0]}", "unit analisis clustering"),
        ("Total Biaya Perbaikan", f"${df['repair_cost'].sum():,.0f}", "akumulasi seluruh cacat"),
        ("Rata-rata Biaya / Cacat", f"${df['repair_cost'].mean():,.2f}", "per kejadian"),
    ]
    for col, (label, value, sub) in zip([c1, c2, c3, c4], cards):
        col.markdown(
            f'<div class="gauge-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )

    rivet_divider()
    st.markdown("### Cuplikan Data Mentah")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.markdown("### Alur Analisis")
    st.markdown(
        "1. **Rekayasa fitur** — agregasi 1.000 catatan cacat menjadi 100 baris (satu per produk).\n"
        "2. **Standardisasi** fitur numerik & proporsi kategori (severity, tipe, lokasi, metode inspeksi).\n"
        "3. **K-Means Clustering** dengan validasi *elbow method* & *silhouette score*.\n"
        "4. **Interpretasi** — penamaan klaster otomatis & rekomendasi bisnis untuk tim QC."
    )

# --------------------------------------------------------------------------------
# PAGE 2 — DATA & EKSPLORASI
# --------------------------------------------------------------------------------
elif page == "📊 Data & Eksplorasi":
    st.markdown("# 📊 Data & Eksplorasi")
    rivet_divider()

    tab1, tab2 = st.tabs(["Data Mentah (per kejadian cacat)", "Data Teragregasi (per produk)"])

    with tab1:
        colf1, colf2, colf3 = st.columns(3)
        sev_f = colf1.multiselect("Severity", sorted(df["severity"].unique()), default=list(df["severity"].unique()))
        typ_f = colf2.multiselect("Jenis Cacat", sorted(df["defect_type"].unique()), default=list(df["defect_type"].unique()))
        insp_f = colf3.multiselect("Metode Inspeksi", sorted(df["inspection_method"].unique()), default=list(df["inspection_method"].unique()))
        filt = df[df["severity"].isin(sev_f) & df["defect_type"].isin(typ_f) & df["inspection_method"].isin(insp_f)]
        st.dataframe(filt, use_container_width=True, hide_index=True)
        st.caption(f"Menampilkan {len(filt):,} dari {len(df):,} catatan.")

        cA, cB = st.columns(2)
        with cA:
            fig = px.histogram(filt, x="repair_cost", nbins=30, color="severity",
                                title="Distribusi Biaya Perbaikan per Severity",
                                color_discrete_map={"Critical": "#E4572E", "Moderate": "#F2A93B", "Minor": "#4FA69A"})
            fig.update_layout(template="plotly_dark", plot_bgcolor="#1F2733", paper_bgcolor="#1F2733")
            st.plotly_chart(fig, use_container_width=True)
        with cB:
            ct = filt.groupby(["defect_type", "defect_location"]).size().reset_index(name="jumlah")
            fig2 = px.bar(ct, x="defect_type", y="jumlah", color="defect_location", barmode="group",
                          title="Jumlah Cacat: Jenis x Lokasi",
                          color_discrete_sequence=["#5E7CE2", "#F2A93B", "#4FA69A"])
            fig2.update_layout(template="plotly_dark", plot_bgcolor="#1F2733", paper_bgcolor="#1F2733")
            st.plotly_chart(fig2, use_container_width=True)

        df_month = filt.copy()
        df_month["bulan"] = df_month["defect_date"].dt.to_period("M").astype(str)
        trend = df_month.groupby("bulan").agg(jumlah=("defect_id", "count"), biaya=("repair_cost", "sum")).reset_index()
        fig3 = go.Figure()
        fig3.add_bar(x=trend["bulan"], y=trend["jumlah"], name="Jumlah Cacat", marker_color="#5E7CE2")
        fig3.add_scatter(x=trend["bulan"], y=trend["biaya"] / trend["biaya"].max() * trend["jumlah"].max(),
                          name="Biaya (skala relatif)", mode="lines+markers", line=dict(color="#F2A93B"))
        fig3.update_layout(title="Tren Bulanan: Jumlah Cacat & Biaya Perbaikan", template="plotly_dark",
                            plot_bgcolor="#1F2733", paper_bgcolor="#1F2733")
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.write(
            "Setiap baris merepresentasikan **satu produk**, dengan fitur hasil agregasi dari seluruh "
            "catatan cacat produk tersebut — inilah tabel yang menjadi input clustering."
        )
        st.dataframe(features.round(3), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Unduh Data Teragregasi (CSV)", features.to_csv(index=False),
                            file_name="product_features_aggregated.csv", mime="text/csv")

# --------------------------------------------------------------------------------
# PAGE 3 — ANALISIS CLUSTERING
# --------------------------------------------------------------------------------
elif page == "🔍 Analisis Clustering":
    st.markdown("# 🔍 Analisis Clustering (K-Means)")
    rivet_divider()

    st.markdown(
        "**Metodologi:** 16 fitur numerik/proporsi (frekuensi cacat, biaya, serta komposisi severity, "
        "jenis cacat, lokasi cacat, dan metode inspeksi) distandardisasi (Z-score) lalu dikelompokkan "
        "dengan **K-Means**. Nilai K divalidasi menggunakan *elbow method* (inertia) dan *silhouette score*."
    )

    scan = scan_k(features)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(scan, x="k", y="inertia", markers=True, title="Elbow Method — Inertia vs K")
        fig.update_traces(line_color="#5E7CE2", marker=dict(size=9, color="#F2A93B"))
        fig.update_layout(template="plotly_dark", plot_bgcolor="#1F2733", paper_bgcolor="#1F2733")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(scan, x="k", y="silhouette", markers=True, title="Silhouette Score vs K")
        fig.update_traces(line_color="#4FA69A", marker=dict(size=9, color="#F2A93B"))
        fig.update_layout(template="plotly_dark", plot_bgcolor="#1F2733", paper_bgcolor="#1F2733")
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"K yang dipilih saat ini: **K = {k_choice}** · Silhouette score: **{sil_score:.3f}** "
        "(gunakan slider di sidebar untuk mencoba nilai K lain)."
    )

    rivet_divider()
    cL, cR = st.columns([1.3, 1])
    with cL:
        fig = px.scatter(
            clustered, x="pca_1", y="pca_2", color=clustered["cluster"].astype(str),
            hover_data=["product_id", "total_defects", "avg_repair_cost"],
            title=f"Visualisasi Klaster (PCA 2D — variansi terjelaskan {sum(evr)*100:.1f}%)",
            color_discrete_sequence=["#5E7CE2", "#F2A93B", "#4FA69A", "#E4572E", "#9B7EDE", "#3AAFA9", "#D6A2E8", "#E88D67"],
        )
        fig.update_traces(marker=dict(size=11, line=dict(width=1, color="#161B22")))
        fig.update_layout(template="plotly_dark", plot_bgcolor="#1F2733", paper_bgcolor="#1F2733", legend_title="Klaster")
        st.plotly_chart(fig, use_container_width=True)
    with cR:
        sizes = clustered["cluster"].value_counts().sort_index()
        fig = px.bar(x=[f"Klaster {i}" for i in sizes.index], y=sizes.values,
                     title="Jumlah Produk per Klaster", color=[f"Klaster {i}" for i in sizes.index],
                     color_discrete_sequence=["#5E7CE2", "#F2A93B", "#4FA69A", "#E4572E", "#9B7EDE", "#3AAFA9", "#D6A2E8", "#E88D67"])
        fig.update_layout(template="plotly_dark", plot_bgcolor="#1F2733", paper_bgcolor="#1F2733", showlegend=False,
                           xaxis_title="", yaxis_title="Jumlah Produk")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Profil Rata-rata Fitur per Klaster")
    profile = clustered.groupby("cluster")[FEATURE_COLS].mean().round(3)
    profile.index = [f"Klaster {i}" for i in profile.index]
    st.dataframe(profile.T, use_container_width=True)

# --------------------------------------------------------------------------------
# PAGE 4 — INTERPRETASI & INSIGHT BISNIS
# --------------------------------------------------------------------------------
elif page == "💡 Interpretasi & Insight Bisnis":
    st.markdown("# 💡 Interpretasi Hasil & Insight Bisnis")
    rivet_divider()
    st.write(
        f"Model K-Means dengan **K = {k_choice}** (silhouette score **{sil_score:.3f}**) menghasilkan "
        f"{k_choice} kelompok produk dengan karakteristik cacat yang berbeda. Berikut interpretasi "
        "tiap klaster beserta rekomendasi tindakan untuk tim Quality Control:"
    )

    for c in sorted(narratives.keys()):
        nar = narratives[c]
        with st.container():
            st.markdown(
                f"""<div class="insight-box">
                <h4>Klaster {c} — {nar['n']} produk &nbsp; {risk_tag(nar['risk'])}</h4>
                <p><b>{nar['title']}</b></p>
                <p style="color:#8A94A6; font-size:0.9rem;">
                Rata-rata {nar['avg_defects']:.1f} cacat/produk &middot;
                biaya rata-rata ${nar['avg_cost']:.0f}/kejadian &middot;
                total biaya klaster ${nar['total_cost']:.0f} &middot;
                severity dominan <b>{nar['dom_sev']}</b> &middot;
                jenis cacat dominan <b>{nar['dom_type']}</b> &middot;
                lokasi dominan <b>{nar['dom_loc']}</b> &middot;
                metode inspeksi dominan <b>{nar['dom_insp']}</b>
                </p>
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown("**Rekomendasi tindakan:**")
            for r in nar["rekomendasi"]:
                st.markdown(f"- {r}")
            st.write("")

    rivet_divider()
    st.markdown("### Ringkasan Insight Bisnis")
    high_risk = [c for c, n in narratives.items() if n["risk"] == "Tinggi"]
    total_cost_all = clustered["total_repair_cost"].sum()
    hr_cost = clustered[clustered["cluster"].isin(high_risk)]["total_repair_cost"].sum() if high_risk else 0
    pct_hr_cost = (hr_cost / total_cost_all * 100) if total_cost_all else 0

    st.markdown(
        f"- **{len(high_risk)} dari {k_choice} klaster** tergolong **risiko tinggi**, "
        f"menyumbang sekitar **{pct_hr_cost:.1f}%** dari total biaya perbaikan seluruh produk.\n"
        "- Produk dengan proporsi *Manual Testing* tinggi dan severity *Critical* tinggi sebaiknya "
        "menjadi prioritas untuk migrasi ke *Automated Testing* guna deteksi dini.\n"
        "- Cacat **Structural** umumnya berbiaya tinggi dan berulang — perlu ditelusuri ke tahap desain "
        "atau pemilihan material, bukan hanya diperbaiki di tahap akhir.\n"
        "- Klaster dengan frekuensi & biaya rendah dapat dijadikan **benchmark proses produksi** "
        "untuk direplikasi ke lini produk lain."
    )

# --------------------------------------------------------------------------------
# PAGE 5 — TENTANG APLIKASI
# --------------------------------------------------------------------------------
else:
    st.markdown("# ℹ️ Tentang Aplikasi")
    rivet_divider()
    st.markdown(
        """
        **ANALISIS CLUSTERING CACAT PRODUK INDUSTRI MANUFAKTUR** dibuat sebagai proyek deployment
        untuk tugas mata kuliah (Pertemuan ke-12), mengikuti alur tutorial deployment Streamlit:

        - Kode disusun dalam `app.py` beserta `requirements.txt` berisi dependensi
          (`streamlit`, `pandas`, `numpy`, `scikit-learn`, `plotly`) — tanpa pustaka yang tidak
          terpakai (mis. `pickle`) agar proses build di Streamlit Cloud tidak error.
        - Repository dihubungkan ke **Streamlit Community Cloud** melalui akun GitHub, dengan
          branch `main` dan entry point `app.py`.
        - Setelah deploy, server di-*reboot* bila diperlukan hingga status konsisten, lalu
          menghasilkan URL publik yang bisa diuji langsung (mis. mengganti nilai K klaster).

        **Metode analisis:** K-Means Clustering pada level produk (bukan level kejadian cacat),
        dengan 16 fitur hasil rekayasa dari kolom `defect_type`, `defect_location`, `severity`,
        `inspection_method`, dan `repair_cost`. Validasi jumlah klaster memakai *elbow method*
        dan *silhouette score*.

        **Sumber data:** `defects_data.csv` — 1.000 catatan cacat dari 100 produk manufaktur.
        """
    )
    st.caption("Dibuat dengan Streamlit • scikit-learn • Plotly")
