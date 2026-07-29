import sqlite3
import hashlib
from datetime import datetime

DB_PATH = "strokesense.db"

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # Tabel users
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role     TEXT NOT NULL DEFAULT 'dokter'
        )
    """)

    # Tabel riwayat diagnosis
    cur.execute("""
        CREATE TABLE IF NOT EXISTS riwayat (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT NOT NULL,
            nama_pasien  TEXT NOT NULL,
            id_rekam     TEXT NOT NULL,
            usia         INTEGER,
            jenis_kelamin TEXT,
            hasil        TEXT,
            confidence   REAL,
            tanggal      TEXT NOT NULL
        )
    """)

    # Migrasi kolom confidence untuk database lama yang sudah ada sebelum perubahan ini
    try:
        cur.execute("ALTER TABLE riwayat ADD COLUMN confidence REAL")
    except sqlite3.OperationalError:
        pass

    # Seed akun admin default jika belum ada
    hashed = hashlib.sha256("admin123".encode()).hexdigest()
    cur.execute("""
        INSERT OR IGNORE INTO users (name, username, password, role)
        VALUES (?, ?, ?, ?)
    """, ("Admin", "admin", hashed, "admin"))

    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- Fungsi Users ---
def get_user(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT username, password, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row  # (username, password, role) atau None

def get_all_users():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT username, role FROM users ORDER BY role, username")
    rows = cur.fetchall()
    conn.close()
    return rows

def add_user(name, username, password):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (name, username, password) VALUES (?, ?, ?)",
            (name, username, hash_password(password))
        )
        conn.commit()
        conn.close()
        return True, "User berhasil ditambahkan."
    except sqlite3.IntegrityError:
        return False, "Username sudah ada."

def delete_user(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()

# --- Fungsi Riwayat ---
def simpan_riwayat(username, nama_pasien, id_rekam, usia, jenis_kelamin, hasil, confidence):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO riwayat (username, nama_pasien, id_rekam, usia, jenis_kelamin, hasil, confidence, tanggal)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (username, nama_pasien, id_rekam, usia, jenis_kelamin, hasil, confidence, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()

def get_riwayat_dokter(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT nama_pasien, id_rekam, usia, jenis_kelamin, hasil, confidence, tanggal
        FROM riwayat WHERE username = ? ORDER BY tanggal DESC
    """, (username,))
    rows = cur.fetchall()
    conn.close()
    return rows

def get_riwayat_semua():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT u.name, r.nama_pasien, r.id_rekam, r.usia, r.jenis_kelamin, r.hasil, r.confidence, r.tanggal
        FROM riwayat r
        JOIN users u ON r.username = u.username
        ORDER BY r.tanggal DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return rows

def get_nama_dokter(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM users WHERE username = ?", (username,))
    row = cur.fetchone()  
    conn.close()
    return row[0] if row else username 

def update_password(username, password_baru):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET password = ? WHERE username = ?",
        (hash_password(password_baru), username)
    )
    conn.commit()
    conn.close()

def verify_password(username, password):
    row = get_user(username)
    if row and row[1] == hash_password(password):
        return True
    return False
