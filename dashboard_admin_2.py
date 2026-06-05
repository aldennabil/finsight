import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import pickle
import umap
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="FinSight Admin Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Injeksi CSS Kustom untuk Gaya Dark Mode Profesional
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #F1F3F5 !important;
    }
    
    .stApp {
        background-color: #0B0F19;
    }
    
    [data-testid="stSidebar"] {
        background-color: #151B26 !important;
        border-right: 1px solid #1E2638;
    }
    
    .kpi-card {
        background-color: #1E2638;
        padding: 12px 15px;
        border-radius: 6px;
        border: 1px solid #2D3748;
        border-top: 3px solid #3B82F6;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .kpi-value {
        font-size: 18px;
        font-weight: 700;
        color: #3B82F6;
        margin: 2px 0;
    }
    
    .kpi-label {
        font-size: 11px;
        color: #A0AEC0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-subtext {
        font-size: 10px;
        color: #718096;
    }
    
    .section-header {
        color: #3B82F6;
        border-left: 3px solid #3B82F6;
        padding-left: 8px;
        margin-top: 15px;
        margin-bottom: 10px;
        font-weight: 600;
        font-size: 16px;
    }
    
    .filter-box {
        background-color: #1E2638;
        padding: 12px 15px;
        border-radius: 6px;
        border: 1px solid #2D3748;
        margin-bottom: 12px;
    }
    
    .ai-box {
        background-color: #151B26;
        border: 1px solid #2D3748;
        padding: 12px 15px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    
    /* Custom style for form inputs in darkmode */
    div[data-baseweb="select"] > div {
        background-color: #1E2638 !important;
        color: #F1F3F5 !important;
        border-color: #2D3748 !important;
    }
    
    div[role="listbox"] {
        background-color: #1E2638 !important;
        color: #F1F3F5 !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #F1F3F5 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Memuat Data dengan Caching Optimal
@st.cache_data(ttl=3600)
def load_clean_data():
    try:
        df_nasabah = pd.read_csv("data/df_nasabah.csv")
        df_transaksi = pd.read_csv("data/Data Transaksi.csv")
        df_behavior = pd.read_csv("data/Analysis ready.csv")
        
        # Parse timestamp
        df_transaksi['timestamp'] = pd.to_datetime(df_transaksi['timestamp'])
        
        return df_nasabah, df_transaksi, df_behavior
    except FileNotFoundError as e:
        st.error(f"File data tidak ditemukan: {e}. Harap pastikan semua data ada di folder 'data/'.")
        return None, None, None

df_nasabah, df_transaksi, df_behavior = load_clean_data()

# Memuat model NLP Naive Bayes
@st.cache_resource
def load_nlp_model():
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('nlp_model_nb.pkl', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError:
        return None, None

vectorizer, nlp_model = load_nlp_model()

if df_nasabah is not None and df_transaksi is not None and df_behavior is not None:
    
    # Siapkan data gabungan untuk transaksi agar bisa difilter berdasarkan atribut nasabah
    df_transaksi_joined = df_transaksi.merge(
        df_nasabah[['id_user', 'nama_nasabah', 'segmen_demografi', 'persona_dasar', 'gaji_bulanan']],
        on='id_user',
        how='left'
    )
    
    # Opsi filter
    month_options = sorted(df_transaksi_joined['bulan'].dropna().unique().tolist())
    demo_options = sorted(df_nasabah['segmen_demografi'].dropna().unique().tolist())
    persona_options = sorted(df_nasabah['persona_dasar'].dropna().unique().tolist())
    status_options = ['Normal', 'Anomali']
    
    # Manajemen Session State untuk Reset
    if 'selected_months' not in st.session_state:
        st.session_state.selected_months = month_options
    if 'selected_demos' not in st.session_state:
        st.session_state.selected_demos = demo_options
    if 'selected_personas' not in st.session_state:
        st.session_state.selected_personas = persona_options
    if 'selected_status' not in st.session_state:
        st.session_state.selected_status = status_options
        
    # ==========================================
    # SIDEBAR: NAVIGASI & FILTER
    # ==========================================
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio(
        "Pilih Halaman:",
        options=[
            "Ringkasan Portofolio",
            "Analisis Transaksi Terfilter",
            "Analisis Lanjutan (ML & NLP)"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Analisis")
    
    selected_months = st.sidebar.multiselect(
        "Pilih Bulan Transaksi:",
        options=month_options,
        key='selected_months'
    )
    
    selected_demos = st.sidebar.multiselect(
        "Pilih Segmen Demografi:",
        options=demo_options,
        key='selected_demos'
    )
    
    selected_personas = st.sidebar.multiselect(
        "Pilih Persona Dasar Nasabah:",
        options=persona_options,
        key='selected_personas'
    )
    
    selected_status = st.sidebar.multiselect(
        "Pilih Status Transaksi:",
        options=status_options,
        key='selected_status'
    )
    
    if st.sidebar.button("Reset Filter", use_container_width=True):
        st.session_state.selected_months = month_options
        st.session_state.selected_demos = demo_options
        st.session_state.selected_personas = persona_options
        st.session_state.selected_status = status_options
        st.rerun()

    # Menyaring Data Transaksi
    filt_months = selected_months if selected_months else month_options
    filt_demos = selected_demos if selected_demos else demo_options
    filt_personas = selected_personas if selected_personas else persona_options
    filt_status = selected_status if selected_status else status_options
    
    # Konversi status ke label_anomali
    anomali_filters = []
    if 'Normal' in filt_status:
        anomali_filters.append(0)
    if 'Anomali' in filt_status:
        anomali_filters.append(1)
        
    df_tx_filtered = df_transaksi_joined[
        (df_transaksi_joined['bulan'].isin(filt_months)) &
        (df_transaksi_joined['segmen_demografi'].isin(filt_demos)) &
        (df_transaksi_joined['persona_dasar'].isin(filt_personas)) &
        (df_transaksi_joined['label_anomali'].isin(anomali_filters))
    ]
    
    # Menyaring Data Nasabah (untuk KPI dan behavior)
    df_nb_filtered = df_nasabah[
        (df_nasabah['segmen_demografi'].isin(filt_demos)) &
        (df_nasabah['persona_dasar'].isin(filt_personas))
    ]

    # Header Utama Aplikasi
    st.markdown("<h1 style='text-align: center; color: #3B82F6; margin-top: 0;'>FinSight Admin Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #A0AEC0; font-size: 14px;'>Monitoring Transaksi Mutasi Nasabah & Analisis Perilaku Keuangan Berbasis K-Means & NLP</p>", unsafe_allow_html=True)
    st.markdown('<hr style="margin: 10px 0; border: none; border-top: 1px solid #1E2638;">', unsafe_allow_html=True)
    
    # Info Jumlah Data Terfilter
    st.markdown(f"<p style='color: #A0AEC0; font-size: 13px;'>Menampilkan <b>{len(df_tx_filtered):,}</b> transaksi dari <b>{len(df_nb_filtered):,}</b> nasabah terfilter.</p>", unsafe_allow_html=True)

    # ==========================================
    # HALAMAN 1: RINGKASAN PORTOFOLIO
    # ==========================================
    if page == "Ringkasan Portofolio":
        st.markdown("<h2 class='section-header'>Ringkasan Portofolio</h2>", unsafe_allow_html=True)
        
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        total_nasabah_all = len(df_nasabah)
        total_trx_all = len(df_transaksi)
        total_volume_all = df_transaksi['nominal'].sum()
        avg_gaji_all = df_nasabah['gaji_bulanan'].mean()
        
        with col_kpi1:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Total Nasabah</div>
                    <div class="kpi-value">{total_nasabah_all:,}</div>
                    <div class="kpi-subtext">Nasabah terdaftar aktif</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col_kpi2:
            st.markdown(f"""
                <div class="kpi-card" style="border-top-color: #60A5FA;">
                    <div class="kpi-label">Total Transaksi</div>
                    <div class="kpi-value">{total_trx_all:,}</div>
                    <div class="kpi-subtext">Mutasi transaksi terproses</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col_kpi3:
            st.markdown(f"""
                <div class="kpi-card" style="border-top-color: #34D399;">
                    <div class="kpi-label">Volume Transaksi</div>
                    <div class="kpi-value">Rp {total_volume_all:,.0f}</div>
                    <div class="kpi-subtext">Total perputaran dana</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col_kpi4:
            st.markdown(f"""
                <div class="kpi-card" style="border-top-color: #FBBF24;">
                    <div class="kpi-label">Rata-rata Gaji</div>
                    <div class="kpi-value">Rp {avg_gaji_all:,.0f}</div>
                    <div class="kpi-subtext">Pendapatan bulanan nasabah</div>
                </div>
            """, unsafe_allow_html=True)
            
        # Visualisasi Umum (Overview)
        col_chart1, col_chart2 = st.columns([3, 2])
        
        with col_chart1:
            demo_counts = df_nasabah['segmen_demografi'].value_counts().reset_index()
            demo_counts.columns = ['Segmen Demografi', 'Jumlah Nasabah']
            fig_demo = px.bar(
                demo_counts,
                x='Segmen Demografi',
                y='Jumlah Nasabah',
                title="Distribusi Nasabah berdasarkan Segmen Demografi",
                color_discrete_sequence=['#3B82F6'],
                template='plotly_dark'
            )
            fig_demo.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=280, font=dict(color='#F1F3F5'), margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_demo, use_container_width=True)
            
        with col_chart2:
            persona_counts = df_nasabah['persona_dasar'].value_counts().reset_index()
            persona_counts.columns = ['Persona Dasar', 'Jumlah']
            fig_persona = px.pie(
                persona_counts,
                values='Jumlah',
                names='Persona Dasar',
                title="Proporsi Persona Dasar Nasabah",
                color_discrete_sequence=['#60A5FA', '#94A3B8', '#F87171'],
                hole=0.4,
                template='plotly_dark'
            )
            fig_persona.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=280, font=dict(color='#F1F3F5'), margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_persona, use_container_width=True)

    # ==========================================
    # HALAMAN 2: ANALISIS TRANSAKSI TERFILTER
    # ==========================================
    elif page == "Analisis Transaksi Terfilter":
        st.markdown("<h2 class='section-header'>Analisis Transaksi Terfilter</h2>", unsafe_allow_html=True)
        
        if len(df_tx_filtered) > 0:
            # KPI khusus terfilter
            col_fkpi1, col_fkpi2, col_fkpi3, col_fkpi4 = st.columns(4)
            
            filtered_debit = df_tx_filtered[df_tx_filtered['tipe_mutasi'] == 'Debit']['nominal'].sum()
            filtered_kredit = df_tx_filtered[df_tx_filtered['tipe_mutasi'] == 'Kredit']['nominal'].sum()
            filtered_anomaly_pct = (df_tx_filtered['label_anomali'] == 1).mean() * 100
            
            with col_fkpi1:
                st.markdown(f"<div style='text-align: center; border: 1px solid #2D3748; padding: 10px; border-radius: 8px; background-color: #1E2638;'><b>Transaksi Terfilter:</b> <br><span style='color: #3B82F6; font-weight: bold; font-size: 20px;'>{len(df_tx_filtered):,}</span></div>", unsafe_allow_html=True)
            with col_fkpi2:
                st.markdown(f"<div style='text-align: center; border: 1px solid #2D3748; padding: 10px; border-radius: 8px; background-color: #1E2638;'><b>Total Pengeluaran (Debit):</b> <br><span style='color: #F87171; font-weight: bold; font-size: 20px;'>Rp {filtered_debit:,.0f}</span></div>", unsafe_allow_html=True)
            with col_fkpi3:
                st.markdown(f"<div style='text-align: center; border: 1px solid #2D3748; padding: 10px; border-radius: 8px; background-color: #1E2638;'><b>Total Pemasukan (Kredit):</b> <br><span style='color: #34D399; font-weight: bold; font-size: 20px;'>Rp {filtered_kredit:,.0f}</span></div>", unsafe_allow_html=True)
            with col_fkpi4:
                st.markdown(f"<div style='text-align: center; border: 1px solid #2D3748; padding: 10px; border-radius: 8px; background-color: #1E2638;'><b>Rasio Transaksi Anomali:</b> <br><span style='color: #FBBF24; font-weight: bold; font-size: 20px;'>{filtered_anomaly_pct:.2f}%</span></div>", unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 2 Kolom Chart Detail Terfilter
            col_vis1, col_vis2 = st.columns(2)
            
            with col_vis1:
                # Chart 1: Nominal per kategori detail (Debit)
                df_debit_cat = df_tx_filtered[df_tx_filtered['tipe_mutasi'] == 'Debit'].groupby('kategori_detail')['nominal'].sum().reset_index()
                df_debit_cat = df_debit_cat.sort_values(by='nominal', ascending=True)
                fig_debit_cat = px.bar(
                    df_debit_cat,
                    y='kategori_detail',
                    x='nominal',
                    orientation='h',
                    title="Porsi Pengeluaran per Kategori Detail (Debit)",
                    labels={'kategori_detail': 'Kategori Detail', 'nominal': 'Total Pengeluaran (Rp)'},
                    color_discrete_sequence=['#F87171'],
                    template='plotly_dark'
                )
                fig_debit_cat.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=280, font=dict(color='#F1F3F5'), margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_debit_cat, use_container_width=True)
                
            with col_vis2:
                # Chart 2: Tren harian volume transaksi
                df_tx_filtered['tanggal'] = df_tx_filtered['timestamp'].dt.date
                df_daily = df_tx_filtered.groupby(['tanggal', 'tipe_mutasi'])['nominal'].sum().reset_index()
                
                fig_daily = px.line(
                    df_daily,
                    x='tanggal',
                    y='nominal',
                    color='tipe_mutasi',
                    title="Tren Harian Volume Transaksi",
                    labels={'tanggal': 'Tanggal', 'nominal': 'Volume Transaksi (Rp)', 'tipe_mutasi': 'Tipe Mutasi'},
                    color_discrete_map={'Debit': '#EF4444', 'Kredit': '#10B981'},
                    template='plotly_dark'
                )
                fig_daily.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=280, font=dict(color='#F1F3F5'), margin=dict(l=20, r=20, t=40, b=20))
                fig_daily.update_traces(line=dict(width=3))
                st.plotly_chart(fig_daily, use_container_width=True)
                
            # Tabel Pencarian Transaksi
            st.markdown("#### Pencarian Transaksi")
            search_query = st.text_input("Masukkan kata kunci untuk deskripsi mutasi atau catatan:", "")
            
            df_table = df_tx_filtered.copy()
            if search_query:
                df_table = df_table[
                    df_table['deskripsi_mutasi'].str.contains(search_query, case=False, na=False) |
                    df_table['catatan_mutasi'].str.contains(search_query, case=False, na=False)
                ]
                
            # Format kolom nominal & sisa saldo untuk tampilan tabel
            df_table_show = df_table[[
                'id_transaksi', 'nama_nasabah', 'timestamp', 'tipe_mutasi', 
                'nominal', 'sisa_saldo', 'kategori_detail', 'catatan_mutasi', 'label_anomali'
            ]].copy()
            
            # Rename columns to Indonesian
            df_table_show.columns = [
                'ID Transaksi', 'Nama Nasabah', 'Waktu', 'Tipe', 
                'Nominal', 'Sisa Saldo', 'Kategori', 'Catatan', 'Anomali'
            ]
            
            st.dataframe(
                df_table_show.head(1000).style.format({
                    'Nominal': 'Rp {:,.0f}',
                    'Sisa Saldo': 'Rp {:,.0f}',
                    'Anomali': lambda x: 'Anomali' if x == 1 else 'Normal'
                }),
                use_container_width=True
            )
        else:
            st.warning("Tidak ada data transaksi yang cocok dengan filter saat ini.")
            
        st.markdown("---")
        
        # Perbandingan Karakteristik Perilaku
        st.markdown("<h2 class='section-header'>Perbandingan Karakteristik Perilaku</h2>", unsafe_allow_html=True)
        st.markdown("Bandingkan rasio keuangan, kecenderungan berbelanja, tabungan, dan volatilitas saldo antar kelompok nasabah.")
        
        df_behavior_joined = df_behavior.merge(
            df_nasabah[['id_user', 'segmen_demografi', 'persona_dasar', 'gaji_bulanan']],
            on='id_user',
            how='left'
        )
        
        df_bh_filtered = df_behavior_joined[
            (df_behavior_joined['bulan'].isin(filt_months)) &
            (df_behavior_joined['segmen_demografi'].isin(filt_demos)) &
            (df_behavior_joined['persona_dasar'].isin(filt_personas))
        ]
        
        var_pembanding = st.selectbox(
            "Pilih Kategori untuk Dibandingkan:",
            options=[
                ("segmen_demografi", "Segmen Demografi"),
                ("persona_dasar", "Persona Dasar Nasabah"),
                ("bulan", "Bulan Agregasi")
            ],
            format_func=lambda x: x[1]
        )[0]
        
        if len(df_bh_filtered) > 0:
            df_compare = df_bh_filtered.groupby(var_pembanding).agg(
                Rata_Wants_Ratio=('wants_ratio', lambda x: x.mean() * 100),
                Rata_Savings_Rate=('savings_rate', lambda x: x.mean() * 100),
                Rata_Wants_Freq=('wants_frequency', 'mean'),
                Rata_Volatilitas_Saldo=('balance_volatility', lambda x: x.mean() * 100)
            ).reset_index()
            
            label_map = {
                "segmen_demografi": "Segmen Demografi",
                "persona_dasar": "Persona Dasar",
                "bulan": "Bulan"
            }
            name_x = label_map[var_pembanding]
            df_compare.rename(columns={var_pembanding: name_x}, inplace=True)
            
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                fig_comp_wants = px.bar(
                    df_compare,
                    x=name_x,
                    y='Rata_Wants_Ratio',
                    title=f"Rata-rata Wants Ratio (%) per {name_x}",
                    labels={'Rata_Wants_Ratio': 'Wants Ratio (%)'},
                    color_discrete_sequence=['#F87171'],
                    template='plotly_dark'
                )
                fig_comp_wants.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=240, font=dict(color='#F1F3F5'), margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_comp_wants, use_container_width=True)
                
                fig_comp_freq = px.bar(
                    df_compare,
                    x=name_x,
                    y='Rata_Wants_Freq',
                    title=f"Rata-rata Frekuensi Belanja Wants per {name_x}",
                    labels={'Rata_Wants_Freq': 'Frekuensi Transaksi Wants (kali/bulan)'},
                    color_discrete_sequence=['#F59E0B'],
                    template='plotly_dark'
                )
                fig_comp_freq.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=240, font=dict(color='#F1F3F5'), margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_comp_freq, use_container_width=True)
                
            with col_comp2:
                fig_comp_save = px.bar(
                    df_compare,
                    x=name_x,
                    y='Rata_Savings_Rate',
                    title=f"Rata-rata Savings Rate (%) per {name_x}",
                    labels={'Rata_Savings_Rate': 'Savings Rate (%)'},
                    color_discrete_sequence=['#34D399'],
                    template='plotly_dark'
                )
                fig_comp_save.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=240, font=dict(color='#F1F3F5'), margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_comp_save, use_container_width=True)
                
                fig_comp_vol = px.bar(
                    df_compare,
                    x=name_x,
                    y='Rata_Volatilitas_Saldo',
                    title=f"Rata-rata Volatilitas Saldo (%) per {name_x}",
                    labels={'Rata_Volatilitas_Saldo': 'Volatilitas Saldo (% dari Gaji)'},
                    color_discrete_sequence=['#A78BFA'],
                    template='plotly_dark'
                )
                fig_comp_vol.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=240, font=dict(color='#F1F3F5'), margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_comp_vol, use_container_width=True)
                
            st.markdown(f"#### Tabel Perbandingan Karakteristik Rata-rata per {name_x}")
            st.dataframe(
                df_compare.style.format({
                    'Rata_Wants_Ratio': '{:.2f}%',
                    'Rata_Savings_Rate': '{:.2f}%',
                    'Rata_Wants_Freq': '{:.1f} kali',
                    'Rata_Volatilitas_Saldo': '{:.2f}%'
                }),
                use_container_width=True
            )
        else:
            st.warning("Tidak ada data perilaku yang cocok dengan filter saat ini.")

    # ==========================================
    # HALAMAN 3: ANALISIS LANJUTAN (ML & NLP)
    # ==========================================
    elif page == "Analisis Lanjutan (ML & NLP)":
        st.markdown("<h2 class='section-header'>Analisis Lanjutan (ML & NLP)</h2>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Klasifikasi Transaksi (NLP)", "Segmentasi Nasabah (GMM & UMAP)"])
        
        # TAB 1: NLP
        with tab1:
            st.markdown("### Klasifikasi Transaksi Ambigu (P2P Transfer) Menggunakan Naive Bayes")
            st.markdown("""
                Pada sistem keuangan perbankan, transaksi transfer antarpengguna (P2P) seringkali tidak memiliki kategori yang jelas. 
                Modul ini menggunakan model **Multinomial Naive Bayes + TF-IDF** (dilatih pada notebook `Exploration and NLP.ipynb`) 
                untuk mengklasifikasikan deskripsi mutasi & catatan transfer ke dalam kategori spesifik secara otomatis.
            """)
            
            col_nlp_input, col_nlp_metrics = st.columns([3, 2])
            
            with col_nlp_input:
                st.markdown("#### Uji Coba Model Klasifikasi Real-time")
                user_text = st.text_input(
                    "Masukkan Catatan atau Deskripsi Mutasi:",
                    value="transfer gopay jajan seblak bareng temen kosan"
                )
                
                classify_btn = st.button("Klasifikasikan Transaksi", type="primary")
                
                if classify_btn:
                    if vectorizer is not None and nlp_model is not None:
                        text_lower = user_text.lower()
                        text_clean = re.sub(r'[^a-z0-9\s]', ' ', text_lower)
                        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
                        
                        vec = vectorizer.transform([text_clean])
                        pred_class = nlp_model.predict(vec)[0]
                        
                        probs = nlp_model.predict_proba(vec)[0]
                        class_idx = np.where(nlp_model.classes_ == pred_class)[0][0]
                        confidence = probs[class_idx] * 100
                        
                        category_map = {
                            'Transportasi': 'Needs',
                            'Tagihan & Utilitas': 'Needs',
                            'Kesehatan & Perawatan Diri': 'Needs',
                            'Groceries & Kebutuhan Pokok': 'Needs',
                            'Belanja Online & Fashion': 'Wants',
                            'Produktivitas & Digital': 'Wants',
                            'F&B dan Nongkrong': 'Wants',
                            'Hiburan & Langganan': 'Wants',
                            'Investasi & Finansial': 'Savings',
                            'Pendapatan Bulanan': 'Income',
                            'Pemasukan Tambahan': 'Income',
                            'Transfer P2P': 'Wants'
                        }
                        kat_besar = category_map.get(pred_class, 'Wants')
                        kat_color = '#34D399' if kat_besar in ['Needs', 'Savings', 'Income'] else '#F87171'
                        
                        st.markdown(f"""
                            <div class="ai-box">
                                <h4 style="margin-top:0; color: #F1F3F5;">Hasil Prediksi:</h4>
                                <p><b>Kategori Detail:</b> <span style="font-size:18px; color:#60A5FA; font-weight:bold;">{pred_class}</span></p>
                                <p><b>Kategori Besar:</b> <span style="font-size:18px; color:{kat_color}; font-weight:bold;">{kat_besar}</span></p>
                                <p><b>Confidence Score:</b> <span style="font-size:16px; font-weight:bold; color: #F1F3F5;">{confidence:.2f}%</span></p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Model NLP (Pickle) tidak berhasil dimuat. Pastikan file 'tfidf_vectorizer.pkl' dan 'nlp_model_nb.pkl' ada di root directory.")
            
            with col_nlp_metrics:
                st.markdown("#### Informasi & Performa Model")
                st.markdown("""
                    - **Algoritma**: Multinomial Naive Bayes
                    - **Ekstraksi Fitur**: TF-IDF Vectorizer (N-Gram 1-2)
                    - **Data Latih**: Mutasi berlabel 'Transfer P2P' (Winsorized)
                    - **Distribusi Klasifikasi**:
                      * *F&B dan Nongkrong* (Dominan jajan / gaya hidup)
                      * *Tagihan & Utilitas* (Kewajiban rutin)
                      * *Belanja Online & Fashion* (E-commerce)
                      * *Groceries & Kebutuhan Pokok* (Belanja dapur)
                """)
                st.info("Model Naive Bayes sangat cepat dan tangguh dalam memproses variasi diksi catatan transfer slang/non-formal khas mahasiswa.")

        # TAB 2: CLUSTERING & UMAP
        with tab2:
            st.markdown("### Segmentasi Perilaku Nasabah (Gaussian Mixture + UMAP)")
            st.markdown("""
                Menggunakan **StandardScaler**, reduksi dimensi **UMAP** (seluruh fitur), dan model **Gaussian Mixture Model (GMM)** untuk mengelompokkan nasabah ke dalam 3 persona makro:
                **Tightwad** (Sangat Hemat), **Unconflicted** (Wajar), dan **Spendthrift** (Boros).
            """)
            
            # Fit models on the fly
            feature_cols = ['wants_ratio', 'fixed_costs_ratio', 'savings_rate', 'wants_frequency', 
                            'small_leaks_ratio', 'night_owl_spending', 'weekend_surge', 
                            'early_month_depletion', 'balance_volatility', 'survival_mode_days']
            df_transformed = df_behavior.copy()
            
            # Log transform
            cols_to_log = ['wants_frequency', 'weekend_surge', 'balance_volatility', 'survival_mode_days']
            for col in cols_to_log:
                df_transformed[col] = np.log1p(df_transformed[col])
                
            # Winsorize
            for col in feature_cols:
                df_transformed[col] = winsorize(df_transformed[col], limits=[0.01, 0.01])
                
            # Fit scaler, UMAP, and Gaussian Mixture
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_transformed[feature_cols].values)
            
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_umap = reducer.fit_transform(X_scaled)
            
            gmm = GaussianMixture(n_components=3, random_state=42)
            cluster_labels = gmm.fit_predict(X_umap)
            
            # Calculate Silhouette Score (Manhattan metric as in the notebook)
            sil_score = silhouette_score(X_umap, cluster_labels, metric='manhattan')
            st.metric("Silhouette Score (Manhattan)", f"{sil_score:.6f}")
            
            # Merge back
            df_ml_plot = df_behavior.copy()
            df_ml_plot['UMAP1'] = X_umap[:, 0]
            df_ml_plot['UMAP2'] = X_umap[:, 1]
            df_ml_plot['Gaussian Mixture Cluster'] = cluster_labels.astype(str)
            
            # Merge with Ground Truth Persona dari df_nasabah
            df_gt_melted = df_nasabah.melt(
                id_vars=['id_user'],
                value_vars=['persona_bulan_1', 'persona_bulan_2', 'persona_bulan_3'],
                var_name='bulan_raw',
                value_name='Ground Truth Persona'
            )
            df_gt_melted['bulan'] = df_gt_melted['bulan_raw'].str.extract(r'(\d+)').astype(int)
            df_ml_plot = df_ml_plot.merge(
                df_gt_melted[['id_user', 'bulan', 'Ground Truth Persona']],
                on=['id_user', 'bulan'],
                how='left'
            )
            
            # UI controls for UMAP Plot
            col_ml_control, col_ml_chart = st.columns([1, 3])
            
            with col_ml_control:
                st.markdown("#### Kontrol Visualisasi")
                color_option = st.radio(
                    "Pewarnaan Titik Scatter:",
                    options=['Gaussian Mixture Cluster', 'Ground Truth Persona']
                )
                
                st.info("""
                    Model menggunakan **Gaussian Mixture Model (GMM)** pada ruang fitur tereduksi **UMAP** (All Features). 
                    Skor Silhouette dievaluasi menggunakan metrik Manhattan sesuai dengan hasil terbaik pada notebook analisis.
                """)
                
            with col_ml_chart:
                # Plotly UMAP Scatter Plot
                fig_umap = px.scatter(
                    df_ml_plot,
                    x='UMAP1',
                    y='UMAP2',
                    color=color_option,
                    title=f"UMAP Scatter Plot - Diberi warna berdasarkan {color_option}",
                    labels={'UMAP1': 'UMAP Component 1', 'UMAP2': 'UMAP Component 2'},
                    hover_data=['id_user', 'bulan', 'wants_ratio', 'savings_rate'],
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    template='plotly_dark'
                )
                
                fig_umap.update_layout(plot_bgcolor='rgba(30,38,56,0.9)', paper_bgcolor='rgba(0,0,0,0)', height=380, font=dict(color='#F1F3F5'), margin=dict(l=20, r=20, t=40, b=20))
                fig_umap.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
                fig_umap.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
                st.plotly_chart(fig_umap, use_container_width=True)
                
            # Analisis Karakteristik Centroid Cluster
            st.markdown("#### Karakteristik Klaster Centroid (Profil Psikologis Keuangan)")
            df_centroids = df_ml_plot.groupby('Gaussian Mixture Cluster').agg(
                Rata_Wants_Ratio=('wants_ratio', lambda x: x.mean() * 100),
                Rata_Wants_Freq=('wants_frequency', 'mean'),
                Rata_Savings_Rate=('savings_rate', lambda x: x.mean() * 100)
            ).reset_index()
            
            # Tambahkan label deskriptif berdasarkan rata-rata rasio
            def label_cluster(row):
                if row['Rata_Wants_Ratio'] > 50 and row['Rata_Savings_Rate'] < 0:
                    return "Spendthrift (Konsumtif / Dissaving)"
                elif row['Rata_Wants_Ratio'] < 35:
                    return "Tightwad (Super Hemat / Saving High)"
                else:
                    return "Unconflicted (Moderat / Seimbang)"
                    
            df_centroids['Karakteristik Persona'] = df_centroids.apply(label_cluster, axis=1)
            st.dataframe(
                df_centroids.style.format({
                    'Rata_Wants_Ratio': '{:.2f}%',
                    'Rata_Wants_Freq': '{:.1f} kali',
                    'Rata_Savings_Rate': '{:.2f}%'
                }),
                use_container_width=True
            )
