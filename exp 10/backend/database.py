import sqlite3
import datetime
from backend.config import DB_PATH

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(filename, prediction, confidence):
    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    now_str = datetime.datetime.now().strftime('%d/%m/%Y, %I:%M:%S %p')
    cur.execute(
        'INSERT INTO history (filename, prediction, confidence, created_at) VALUES (?, ?, ?, ?)',
        (filename, prediction, confidence, now_str)
    )
    conn.commit()
    conn.close()

def get_all_history():
    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, filename, prediction, confidence, created_at FROM history ORDER BY id DESC')
    rows = cur.fetchall()
    history = [dict(row) for row in rows]
    conn.close()
    return history

def clear_all_history():
    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('DELETE FROM history')
    conn.commit()
    conn.close()
