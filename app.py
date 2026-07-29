import streamlit as st
import base64
import os

import sqlite3
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
import io

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

import os
import gdown

MODEL_PATH = "resnet50_strokesense_best.h5"
GDRIVE_FILE_ID = "18Y9zL4RHiUdPhAuDoSyob1e8bCuIUGMA"

if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)

LABELS = ["hemoragik", "iskemik", "normal"]
IMG_SIZE = 224  

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="StrokeSense - RSU Aulia",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Fungsi untuk Membaca Gambar Lokal ke Base64 ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Mengambil base64 logo (pastikan file logo.png ada di folder yang sama)
logo_path = "logo.png"
if os.path.exists(logo_path):
    img_base64 = get_base64_of_bin_file(logo_path)
    logo_html = f'data:image/png;base64,{img_base64}'
else:
    # Fallback jika file tidak ditemukan
    logo_html = None

# --- Custom CSS (Tema Hijau-Putih) ---
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

:root, .stApp {{
    --primary-color: #2E7D32 !important;
    --background-color: #F3FAF2 !important;
    --secondary-background-color: #FFFFFF !important;
    --text-color: #1F2A24 !important;
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    color-scheme: light !important;
}}

* {{
    font-family: 'Plus Jakarta Sans', sans-serif;
}}

body, [data-testid="stAppViewContainer"] {{
    background: linear-gradient(160deg, #F3FAF2 0%, #EAF5E8 45%, #F7FBF6 100%) !important;
    background-attachment: fixed !important;
    color: #1F2A24 !important;
}}

[data-testid="stHeader"] {{ background: rgba(0,0,0,0) !important; }}
#MainMenu, footer {{ visibility: hidden; }}

.block-container {{
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 780px;
}}

h1, h2, h3, h4, h5, h6, p, span, label, div {{
    color: #1F2A24 !important;
}}

.login-header {{
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
    padding: 2.75rem 2rem;
    text-align: center;
    border-radius: 20px 20px 0 0;
    box-shadow: 0 10px 30px rgba(27,94,32,0.18);
}}
.hospital-name {{ font-family:'Lora', serif; color:#ffffff !important; font-size:17px; font-weight:600; margin:0.9rem 0 0.2rem; letter-spacing:0.02em; }}
.app-title {{ font-family:'Lora', serif; color:#ffffff !important; font-size:26px; font-weight:700; margin:0.4rem 0 0; }}
.app-subtitle {{ color:#C8E6C9 !important; font-size:11px; letter-spacing:0.14em; text-transform:uppercase; margin-top:0.3rem; }}
.divider {{ width:46px; height:3px; background:#A5D6A7; margin:1.1rem auto; border-radius:2px; }}

.logo-container {{
    background:#ffffff; width:96px; height:96px; border-radius:50%;
    margin:0 auto; display:flex; align-items:center; justify-content:center;
    box-shadow:0 6px 16px rgba(0,0,0,0.18);
    overflow:hidden;
}}
.logo-img {{ width:82%; height:auto; object-fit:contain; }}

div[data-testid="stForm"] {{
    background:#ffffff !important;
    border-radius: 0 0 20px 20px;
    padding: 2rem 1.8rem 1.6rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    border: 1px solid #EAF2E9;
}}

[data-testid="stSidebar"] {{
    background: #F4FAF3 !important;
    border-right: 1px solid #E1EEDF;
}}
[data-testid="stSidebar"] * {{ color:#1F2A24 !important; }}

.stTextInput input, .stSelectbox div[data-baseweb="select"], .stTextArea textarea {{
    background:#ffffff !important;
    border:1.5px solid #DCEBDA !important;
    border-radius:10px !important;
    color:#1F2A24 !important;
    padding:0.55rem 0.75rem !important;
}}
.stTextInput input:focus {{
    border-color:#2E7D32 !important;
    box-shadow:0 0 0 3px rgba(46,125,50,0.14) !important;
}}

div.stButton > button:first-child,
div.stFormSubmitButton > button:first-child {{
    background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
    color:#ffffff !important;
    border-radius:10px;
    border:none;
    padding:0.6rem 1.1rem;
    font-weight:600;
    transition: all 0.2s ease;
    box-shadow:0 4px 12px rgba(46,125,50,0.22);
}}
div.stButton > button:hover,
div.stFormSubmitButton > button:hover {{
    transform: translateY(-1px);
    box-shadow:0 6px 16px rgba(27,94,32,0.3);
}}

[data-testid="stSidebar"] .stRadio > div {{ gap:0.35rem; }}
[data-testid="stSidebar"] .stRadio label {{
    background:#ffffff;
    border:1px solid #E1EEDF;
    border-radius:10px;
    padding:0.5rem 0.7rem !important;
    width:100%;
    transition: all 0.15s ease;
}}
[data-testid="stSidebar"] .stRadio label:hover {{ border-color:#2E7D32; background:#F0F8EF; }}

.streamlit-expanderHeader, [data-testid="stExpander"] summary {{
    background:#ffffff !important;
    border-radius:10px !important;
    border:1px solid #E1EEDF !important;
    color:#1F2A24 !important;
    font-weight:600;
}}

[data-testid="stDataFrame"] {{
    border-radius:12px;
    overflow:hidden;
    border:1px solid #E1EEDF;
}}

div[data-testid="stAlert"] {{ border-radius:10px !important; }}

div[data-testid="stProgress"] > div > div > div {{ background-color:#2E7D32 !important; }}

[data-testid="stFileUploaderDropzone"] {{
    background:#ffffff !important;
    border:1.5px dashed #A5D6A7 !important;
    border-radius:12px !important;
}}

.disclaimer-box {{
    background:#FFF8E1 !important;
    border-left:5px solid #FBC02D;
    padding:18px 20px;
    margin-top:22px;
    font-size:14px;
    color:#5C4A1E !important;
    border-radius:10px;
    box-shadow:0 2px 10px rgba(0,0,0,0.05);
}}
.disclaimer-box * {{ color:#5C4A1E !important; }}
.disclaimer-title {{ color:#E65100 !important; font-weight:700; text-transform:uppercase; display:block; margin-bottom:5px; font-size:12px; letter-spacing:0.04em; }}

.info-card {{
    background:#ffffff;
    border:1px solid #E1EEDF;
    border-radius:14px;
    padding:1.2rem 1.4rem;
    box-shadow:0 4px 14px rgba(0,0,0,0.05);
    margin-bottom:1rem;
}}
.info-card h5 {{ margin:0 0 0.8rem; font-size:14px; font-weight:700; color:#1B5E20 !important; }}
.info-row {{ display:flex; justify-content:space-between; padding:0.35rem 0; border-bottom:1px dashed #EDF3EC; font-size:14px; }}
.info-row:last-child {{ border-bottom:none; }}
.info-label {{ color:#5B6B60 !important; }}
.info-value {{ font-weight:600; color:#1F2A24 !important; }}
.badge-hasil {{ display:inline-block; padding:0.3rem 0.8rem; border-radius:999px; font-weight:700; font-size:13px; letter-spacing:0.02em; }}
.badge-normal {{ background:#E8F5E9; color:#1B5E20 !important; }}
.badge-stroke {{ background:#FDECEA; color:#B71C1C !important; }}

div[data-testid="stTextInput"] > div,
div[data-testid="stTextInput"] > div > div,
[data-baseweb="input"],
[data-baseweb="base-input"] {{
    background:#ffffff !important;
    border-radius:10px !important;
    border-color:#DCEBDA !important;
}}
div[data-testid="stTextInput"] input {{
    background:#ffffff !important;
    color:#1F2A24 !important;
}}
div[data-testid="stTextInput"] button,
[data-testid="stTextInputRootElement"] button {{
    background:#ffffff !important;
    border:none !important;
}}
div[data-testid="stTextInput"] button svg {{
    fill:#2E7D32 !important;
}}

div[data-baseweb="select"] > div {{
    background:#ffffff !important;
    border-color:#DCEBDA !important;
    color:#1F2A24 !important;
}}
div[data-baseweb="select"]:focus-within > div {{
    border-color:#2E7D32 !important;
    box-shadow:0 0 0 3px rgba(46,125,50,0.14) !important;
}}
[data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"], li[role="option"] {{
    background:#ffffff !important;
    color:#1F2A24 !important;
}}
li[role="option"]:hover {{
    background:#F0F8EF !important;
}}

[data-testid="stFileUploaderDropzone"] {{
    background:#ffffff !important;
    border:1.5px dashed #A5D6A7 !important;
    border-radius:12px !important;
}}
[data-testid="stFileUploaderDropzone"] section {{
    background:transparent !important;
}}
[data-testid="stFileUploaderDropzone"] button {{
    background:#2E7D32 !important;
    color:#ffffff !important;
    border:none !important;
    border-radius:8px !important;
}}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {{
    color:#5B6B60 !important;
}}

div.stButton button *,
div.stFormSubmitButton button *,
[data-testid="stBaseButton-primary"] *,
[data-testid="stBaseButton-secondary"] * {{
    color:#ffffff !important;
}}

.table-wrap {{
    border:1px solid #E1EEDF;
    border-radius:12px;
    overflow:hidden;
    box-shadow:0 4px 14px rgba(0,0,0,0.05);
    margin-bottom:1rem;
}}
.custom-table {{
    width:100%;
    border-collapse:collapse;
    background:#ffffff;
    font-size:14px;
}}
.custom-table th {{
    background:#2E7D32;
    color:#ffffff !important;
    text-align:left;
    padding:10px 14px;
    font-weight:600;
}}
.custom-table td {{
    padding:9px 14px;
    border-bottom:1px solid #EDF3EC;
    color:#1F2A24 !important;
}}
.custom-table tr:last-child td {{ border-bottom:none; }}
.custom-table tr:hover td {{ background:#F6FBF5; }}
[data-testid="stFileUploaderDropzone"] * {{
    background-color: transparent !important;
    color: #1F2A24 !important;
}}

[data-testid="stFileUploaderDropzone"] {{
    background: #ffffff !important;
}}
[data-testid="stFileUploaderDropzone"] svg {{
    fill: #2E7D32 !important;
    stroke: #2E7D32 !important;
}}

button[data-testid="stBaseButton-secondary"] {{
    background: #2E7D32 !important;
    border: none !important;
    color: #ffffff !important;
}}
button[data-testid="stBaseButton-secondary"] * {{
    color: #ffffff !important;
}}
button[data-testid="stBaseButton-secondary"] + {{
    fill: #ffffff !important;
    stroke: #ffffff !important;
}}

[data-testid="stFileChips"] {{
    background: transparent !important;
}}
[data-testid="stFileChips"] * {{
    color: #1F2A24 !important;
}}
[data-testid="stFileChips"] svg {{
    fill: #ffffff !important;
    stroke: #2E7D32 !important;
}}           

[data-testid="stFileUploaderDropzone"] {{
    background: #ffffff !important;
}}

button[data-testid="stBaseButton-secondary"] svg,
[data-testid="stFileUploaderDeleteBtn"] svg {{
    filter: brightness(0) invert(1) !important;
}}

[data-testid="stFileUploaderDropzone"]:has([data-testid="stFileUploaderFile"]) button[data-testid="stBaseButton-secondary"] {{
display: none !important;
}}
            
</style>
""", unsafe_allow_html=True)

# --- Inisialisasi Session State ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""

# --- import database ---
from database import (
    get_nama_dokter, init_db, get_user, get_all_users,
    add_user, delete_user, hash_password, verify_password, update_password,
    simpan_riwayat, get_riwayat_dokter, get_riwayat_semua
) 

# Inisialisasi database saat app pertama jalan
init_db()


# ============================================================
# FUNGSI TABLE
# ============================================================
def render_table(df):
    if df.empty:
        st.info("Belum ada data.")
        return
    html = df.to_html(index=False, border=0, classes="custom-table")
    st.markdown(f'<div class="table-wrap">{html}</div>', unsafe_allow_html=True)

# ============================================================
# FUNGSI BACA DICOM
# ============================================================
def baca_dicom(uploaded_file):
    """Membaca file DICOM, mengembalikan gambar PNG (array) dan metadata pasien."""
    dicom_data = pydicom.dcmread(uploaded_file)

    # --- Ekstrak metadata pasien ---
    nama_pasien = str(dicom_data.get("PatientName", "Tidak diketahui"))
    id_pasien   = str(dicom_data.get("PatientID", "Tidak diketahui"))
    gender_raw  = str(dicom_data.get("PatientSex", "")).upper()
    gender      = "Laki-laki" if gender_raw == "M" else "Perempuan" if gender_raw == "F" else "Tidak diketahui"

    # Hitung usia dari PatientAge jika ada, atau dari PatientBirthDate
    usia = dicom_data.get("PatientAge", None)
    if usia:
        usia = str(usia).replace("Y", "").lstrip("0") or "0"
    else:
        usia = "Tidak diketahui"

    # --- Konversi pixel ke Hounsfield Unit ---
    pixel_array = dicom_data.pixel_array.astype(np.int16)
    intercept = float(dicom_data.get("RescaleIntercept", 0))
    slope     = float(dicom_data.get("RescaleSlope", 1))
    hu_image  = pixel_array * slope + intercept

    # --- Windowing otak (WL=40, WW=80) ---
    wl, ww = 40, 80
    img_min = wl - ww // 2
    img_max = wl + ww // 2
    windowed = np.clip(hu_image, img_min, img_max)
    windowed = ((windowed - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)

    # --- Konversi ke PNG (PIL Image) ---
    png_image = Image.fromarray(windowed)

    metadata = {
        "nama": nama_pasien,
        "id": id_pasien,
        "usia": usia,
        "gender": gender,
    }

    return png_image, metadata

# ============================================================
# FUNGSI PREDIKSI
# ============================================================
def prediksi_stroke(png_image):
    """Preprocessing citra dan prediksi menggunakan model ResNet-50."""
    # Resize ke ukuran input model
    img_resized = png_image.resize((IMG_SIZE, IMG_SIZE))

    # Konversi grayscale ke RGB (3 channel), karena ResNet-50 butuh 3 channel
    img_rgb = img_resized.convert("RGB")

    # Konversi ke array dan preprocessing standar ResNet-50
    img_array = np.array(img_rgb).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediksi
    pred = model.predict(img_array, verbose=0)[0]

    idx = np.argmax(pred)
    label = LABELS[idx]
    confidence = float(pred[idx]) * 100

    return label, confidence

# ============================================================
# FUNGSI HALAMAN LOGIN
# ============================================================
def show_login_page():
    with st.container():
        # Memuat logo di header login
        logo_display = f'<img src="{logo_html}" class="logo-img">' if logo_html else '<span style="font-size:40px;">🏥</span>'
        
        st.markdown(f"""
        <div class="login-header">
            <div class="logo-container">
                {logo_display}
            </div>
            <p class="hospital-name">Rumah Sakit Umum Aulia</p>
            <div class="divider"></div>
            <p class="app-title">StrokeSense</p>
            <p class="app-subtitle">Sistem Bantu Diagnosis Stroke</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Masukkan username")
            password = st.text_input("Password", type="password", placeholder="Masukkan password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            row = get_user(username)
            if row and row[1] == hash_password(password):
                st.session_state.authenticated = True
                st.session_state.username = row[0]
                st.session_state.role = row[2]
                st.rerun()
            else:
                st.error("Username atau password salah.")

        st.markdown("""
        <p style="text-align: center; color: #666; font-size: 12px; margin-top: 1rem;">
            © 2026 Andira Putri Nirmala - Universitas Gunadarma
        </p>
        """, unsafe_allow_html=True)

# ============================================================
# FUNGSI SIDEBAR
# ============================================================
def show_sidebar():
    with st.sidebar:
        # Menampilkan logo kecil di sidebar jika tersedia
        if logo_html:
            st.markdown(f"""
            <div style="text-align:center;">
                <img src="{logo_html}" style="width:60px; margin-bottom:10px;">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center; font-size:40px;">🏥</div>', unsafe_allow_html=True)
            
        st.markdown(f"""
        <div style="text-align:center; padding-bottom: 1rem;">
            <p style="font-weight:600; font-size:18px; margin:0; color:#1B5E20;">StrokeSense</p>
            <p style="font-size:11px; color:dark-gray; margin:0; letter-spacing:0.05em;">RSU AULIA</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown(f"""
        <div style="background:#DCEDC8; border-radius:8px; padding:10px 12px; margin-bottom:1rem;">
            <p style="font-size:11px; color:#33691E; margin:0;">Login sebagai:</p>
            <p style="font-size:14px; font-weight:600; color:#1B5E20; margin:0;">
                👤 {st.session_state.username}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Menu**")
        menu = st.radio(
            label="navigasi",
            options=["🏠  Halaman Utama", "📋  Riwayat Pasien", "🔑 Ganti Password"],
            label_visibility="collapsed"
        )

        st.divider()

        if st.button("🚪 Keluar", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()

        return menu

#=============================================================
# FUNGSI SIDEBAR ADMIN
#=============================================================
def show_sidebar_admin():
    with st.sidebar:
        if logo_html:
            st.markdown(f'<div style="text-align:center;"><img src="{logo_html}" style="width:60px; margin-bottom:10px;"></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center; font-size:40px;">🏥</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; padding-bottom: 1rem;">
            <p style="font-weight:600; font-size:18px; margin:0; color:#1B5E20;">StrokeSense</p>
            <p style="font-size:11px; color:gray; margin:0; letter-spacing:0.05em;">RSU AULIA </p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown(f"""
        <div style="background:#DCEDC8; border-radius:8px; padding:10px 12px; margin-bottom:1rem;">
            <p style="font-size:11px; color:#33691E; margin:0;">Login sebagai:</p>
            <p style="font-size:14px; font-weight:600; color:#1B5E20; margin:0;">🛡️ Admin</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Menu Admin**")
        menu = st.radio(
            label="navigasi",
            options=["👥  Kelola Pengguna", "📋  Riwayat Klasifikasi", "🔑 Ganti Password"],
            label_visibility="collapsed"
        )

        st.divider()

        if st.button("🚪 Keluar", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.role = ""
            st.rerun()

        return menu

#=============================================================
# FUNGSI HALAMAN DIAGNOSIS DOKTER
#=============================================================
def show_diagnosis():    
    st.subheader(f"Selamat datang, dr. {get_nama_dokter(st.session_state.username)}")
    st.write("Silakan unggah berkas DICOM untuk memulai klasifikasi citra.")

    with st.form("form_klasifikasi"):
        st.markdown("##### 📂 Berkas CT Scan (DICOM)")
        uploaded_file = st.file_uploader("Unggah berkas DICOM (.dcm)", type=["dcm"])
        submit_btn = st.form_submit_button("Mulai Proses Klasifikasi", use_container_width=True)

    if submit_btn:
        if not uploaded_file:
            st.warning("Mohon unggah berkas DICOM terlebih dahulu.")
        else:
            try:
                with st.spinner("Membaca berkas DICOM dan menganalisis citra..."):
                    png_image, metadata = baca_dicom(uploaded_file)

                    # --- Placeholder hasil model, ganti dengan output model asli nanti ---
                    hasil, confidence = prediksi_stroke(png_image)

                    simpan_riwayat(
                        username=st.session_state.username,
                        nama_pasien=metadata["nama"],
                        id_rekam=metadata["id"],
                        usia=metadata["usia"],
                        jenis_kelamin=metadata["gender"],
                        hasil=hasil,
                        confidence=confidence
                    )

                st.success("Analisis selesai.")

                # --- Tampilkan hasil ---
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(png_image, caption="Citra CT Scan (PNG)", use_container_width=True)

                with col2:
                    badge_class = "badge-normal" if hasil == "normal" else "badge-stroke"
                    st.markdown(f"""
                    <div class="info-card">
                        <h5>👤 Informasi Pasien</h5>
                        <div class="info-row"><span class="info-label">Nama</span><span class="info-value">{metadata['nama']}</span></div>
                        <div class="info-row"><span class="info-label">ID Rekam Medis</span><span class="info-value">{metadata['id']}</span></div>
                        <div class="info-row"><span class="info-label">Usia</span><span class="info-value">{metadata['usia']}</span></div>
                        <div class="info-row"><span class="info-label">Jenis Kelamin</span><span class="info-value">{metadata['gender']}</span></div>
                    </div>
                    <div class="info-card">
                        <h5>🩺 Hasil Diagnosis</h5>
                        <div class="info-row"><span class="info-label">Klasifikasi</span><span class="badge-hasil {badge_class}">{hasil.upper()}</span></div>
                        <div class="info-row"><span class="info-label">Tingkat Kepercayaan</span><span class="info-value">{confidence:.1f}%</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(int(confidence))

            except Exception as e:
                st.error(f"Gagal membaca berkas DICOM: {e}")

    st.markdown("""
    <div class="disclaimer-box">
        <span class="disclaimer-title">⚠️ Peringatan</span>
        Aplikasi <strong>StrokeSense</strong> ini bersifat sebagai alat bantu diagnosis berbasis AI.<br>
        Hasil klasifikasi tidak dapat digunakan sebagai diagnosis tunggal dan <strong>wajib diverifikasi kembali</strong> oleh dokter atau tenaga medis yang bersangkutan.
    </div>
    """, unsafe_allow_html=True)

#=============================================================
# FUNGSI HALAMAN KELOLA AKUN
#=============================================================
def get_kelola_user():
    st.title("👥 Kelola Pengguna")
    st.write("Tambah atau hapus akun dokter di sini.")

    # --- Form Tambah User ---
    # Inisialisasi pesan notifikasi
    if "notif_tambah" not in st.session_state:
        st.session_state.notif_tambah = None

    with st.expander("➕ Tambah User Baru", expanded=True):
        with st.form("form_tambah_user"):
            new_name     = st.text_input("Nama Dokter")
            new_username = st.text_input("Username Baru")
            new_password = st.text_input("Password", type="password")
            add_btn      = st.form_submit_button("Tambah User", use_container_width=True)

        if add_btn:
            if new_username and new_password:
                success, msg = add_user(new_name, new_username, new_password)
                st.session_state.notif_tambah = ("success" if success else "warning", msg)
                st.rerun()
            else:
                st.session_state.notif_tambah = ("warning", "Username dan password tidak boleh kosong.")
                st.rerun()

        # Tampilkan notifikasi di luar expander
        if st.session_state.notif_tambah:
            tipe, pesan = st.session_state.notif_tambah
            if tipe == "success":
                st.success(pesan)
            else:
                st.warning(pesan)
            st.session_state.notif_tambah = None

    # --- Form Hapus User ---
    if "notif_hapus" not in st.session_state:
        st.session_state.notif_hapus = None

    with st.expander("🗑️ Hapus User"):
        with st.form("form_hapus_user"):
            user_list    = [u[0] for u in get_all_users() if u[0] != st.session_state.username]
            hapus_target = st.selectbox("Pilih User", user_list)
            del_btn      = st.form_submit_button("Hapus User", use_container_width=True)

        if del_btn:
            delete_user(hapus_target)
            st.session_state.notif_hapus = ("success", f"User '{hapus_target}' berhasil dihapus.")
            st.rerun()

    if st.session_state.notif_hapus:
        tipe, pesan = st.session_state.notif_hapus
        if tipe == "success":
            st.success(pesan)
        else:
            st.warning(pesan)
            st.session_state.notif_hapus = None

    # Tabel daftar user
    with st.expander("Daftar User"):
        rows = get_all_users()
        df_users = pd.DataFrame(rows, columns=["Nama Dokter", "Role"])
        render_table(df_users)

#=============================================================
# FUNGSI HALAMAN GANTI PASSWORD
#=============================================================
def show_ganti_password():
    st.title("🔑 Ganti Password")

    if "notif_password" not in st.session_state:
        st.session_state.notif_password = None

    with st.form("form_ganti_password"):
        password_lama  = st.text_input("Password Lama", type="password")
        password_baru  = st.text_input("Password Baru", type="password")
        password_ulang = st.text_input("Konfirmasi Password Baru", type="password")
        simpan_btn     = st.form_submit_button("Simpan Password Baru", use_container_width=True)

    if simpan_btn:
        if not password_lama or not password_baru or not password_ulang:
            st.session_state.notif_password = ("warning", "Semua field harus diisi.")
        elif not verify_password(st.session_state.username, password_lama):
            st.session_state.notif_password = ("error", "Password lama salah.")
        elif password_baru != password_ulang:
            st.session_state.notif_password = ("warning", "Konfirmasi password tidak cocok.")
        elif len(password_baru) < 6:
            st.session_state.notif_password = ("warning", "Password baru minimal 6 karakter.")
        else:
            update_password(st.session_state.username, password_baru)
            st.session_state.notif_password = ("success", "Password berhasil diubah.")
        st.rerun()

    if st.session_state.notif_password:
        tipe, pesan = st.session_state.notif_password
        if tipe == "success":
            st.success(pesan)
        elif tipe == "error":
            st.error(pesan)
        else:
            st.warning(pesan)
        st.session_state.notif_password = None

# ============================================================
# FUNGSI HALAMAN DOKTER
# ============================================================
def show_main_app():
    selected_menu = show_sidebar()

    if selected_menu == "🏠  Halaman Utama":
        show_diagnosis()

    elif selected_menu == "📋  Riwayat Pasien":
        st.title("📋 Riwayat Pasien Saya")
        rows = get_riwayat_dokter(st.session_state.username)
        if rows:
            df = pd.DataFrame(rows, columns=["Nama Pasien", "ID Rekam", "Usia", "Kelamin", "Hasil", "Confidence", "Tanggal"])
            df = df.drop(columns=["Kelamin"])
            df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
            render_table(df)
        else:
            st.info("Belum ada riwayat diagnosis.")

    elif selected_menu == "🔑 Ganti Password":
        show_ganti_password()

#=============================================================
# FUNGSI HALAMAN ADMIN
#=============================================================
def show_admin_app():
    selected_menu = show_sidebar_admin()

    if selected_menu == "👥  Kelola Pengguna":
        get_kelola_user()
        
    elif selected_menu == "📋  Riwayat Klasifikasi":
        st.title("📋 Riwayat Semua Klasifikasi")

        rows = get_riwayat_semua()
        df = pd.DataFrame(rows, columns=["Dokter", "Nama Pasien", "ID Rekam", "Usia", "Kelamin", "Hasil", "Confidence", "Tanggal"])
        df = df.drop(columns=["Usia", "Kelamin"])
        df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
        render_table(df)

    elif selected_menu == "🔑 Ganti Password":
        show_ganti_password()

# ============================================================
# ROUTING
# ============================================================
if st.session_state.authenticated:
    if st.session_state.role == "admin":
        show_admin_app()
    else:
        show_main_app()
else:
    show_login_page()