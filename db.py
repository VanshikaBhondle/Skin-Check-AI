"""Storage layer. Uses MySQL/TiDB when MYSQL_HOST is set, otherwise a local SQLite file.

Privacy: photos are NEVER stored - only anonymous analysis metadata for analytics.

MySQL env vars: MYSQL_HOST, MYSQL_PORT (3306), MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB (skin).
Encrypted connections (needed by hosted MySQL such as TiDB Cloud):
  MYSQL_SSL=true            SSL verified against the system CA store
  MYSQL_SSL_CA="-----BEGIN CERTIFICATE-----..."  provider CA certificate (PEM text)
"""
from __future__ import annotations

import os
import sqlite3
import ssl
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

USE_MYSQL = bool(os.getenv("MYSQL_HOST"))
SQLITE_PATH = os.getenv("SQLITE_PATH", os.path.join(os.path.dirname(__file__), "skin_data.db"))
PH = "%s" if USE_MYSQL else "?"
AUTO_ID = "INT AUTO_INCREMENT PRIMARY KEY" if USE_MYSQL else "INTEGER PRIMARY KEY AUTOINCREMENT"

ANALYSIS_COLS = ["timestamp", "age_band", "gender", "skin_type", "body_area", "duration", "symptoms",
                 "image_usable", "top_condition", "top_confidence", "conditions", "ai_urgency",
                 "final_urgency", "rule_flags", "model", "helpful", "model_top", "model_confidence"]
FEEDBACK_COLS = ["timestamp", "name", "rating", "comments"]


def _ssl_context():
    ca = os.getenv("MYSQL_SSL_CA", "").strip()
    if not ca and os.getenv("MYSQL_SSL", "").lower() not in ("1", "true", "yes"):
        return None
    return ssl.create_default_context(cadata=ca) if ca else ssl.create_default_context()


@contextmanager
def connect():
    if USE_MYSQL:
        import pymysql
        conn = pymysql.connect(host=os.environ["MYSQL_HOST"], port=int(os.getenv("MYSQL_PORT", 3306)),
                               user=os.getenv("MYSQL_USER", "root"), password=os.getenv("MYSQL_PASSWORD", ""),
                               database=os.getenv("MYSQL_DB", "skin"), charset="utf8mb4", ssl=_ssl_context())
    else:
        conn = sqlite3.connect(SQLITE_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as c:
        cur = c.cursor()
        cur.execute(f"""CREATE TABLE IF NOT EXISTS skin_analyses (
            id {AUTO_ID}, timestamp VARCHAR(32), age_band VARCHAR(16), gender VARCHAR(32),
            skin_type VARCHAR(32), body_area VARCHAR(64), duration VARCHAR(32), symptoms TEXT,
            image_usable INT, top_condition VARCHAR(128), top_confidence INT, conditions TEXT,
            ai_urgency VARCHAR(32), final_urgency VARCHAR(32), rule_flags TEXT, model VARCHAR(64),
            helpful INT, model_top VARCHAR(128), model_confidence REAL)""")
        for col, typ in (("model_top", "VARCHAR(128)"), ("model_confidence", "REAL")):  # upgrade older tables
            try:
                cur.execute(f"ALTER TABLE skin_analyses ADD COLUMN {col} {typ}")
            except Exception:
                pass  # column already exists
        cur.execute(f"""CREATE TABLE IF NOT EXISTS skin_feedback (
            id {AUTO_ID}, timestamp VARCHAR(32), name VARCHAR(100), rating INT, comments TEXT)""")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def age_band(age: int | None) -> str:
    if age is None:
        return "Not given"
    for hi, label in ((1, "Under 1"), (13, "1-12"), (18, "13-17"), (30, "18-29"), (45, "30-44"), (60, "45-59")):
        if age < hi:
            return label
    return "60+"


def insert_analysis(**row) -> int:
    sql = f"INSERT INTO skin_analyses ({', '.join(ANALYSIS_COLS)}) VALUES ({', '.join([PH] * len(ANALYSIS_COLS))})"
    with connect() as c:
        cur = c.cursor()
        cur.execute(sql, [row.get(k) for k in ANALYSIS_COLS])
        return int(cur.lastrowid)


def set_helpful(analysis_id: int, helpful: int) -> None:
    with connect() as c:
        c.cursor().execute(f"UPDATE skin_analyses SET helpful = {PH} WHERE id = {PH}", (helpful, analysis_id))


def insert_feedback(**row) -> None:
    sql = f"INSERT INTO skin_feedback ({', '.join(FEEDBACK_COLS)}) VALUES ({', '.join([PH] * len(FEEDBACK_COLS))})"
    with connect() as c:
        c.cursor().execute(sql, [row.get(k) for k in FEEDBACK_COLS])


def count_today() -> int:
    with connect() as c:
        cur = c.cursor()
        cur.execute(f"SELECT COUNT(*) FROM skin_analyses WHERE timestamp >= {PH}",
                    (datetime.now().strftime("%Y-%m-%d 00:00:00"),))
        return int(cur.fetchone()[0])


def _frame(table: str) -> pd.DataFrame:
    with connect() as c:
        cur = c.cursor()
        cur.execute(f"SELECT * FROM {table} ORDER BY id DESC")
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(list(cur.fetchall()), columns=cols)


def analyses() -> pd.DataFrame:
    return _frame("skin_analyses")


def feedback() -> pd.DataFrame:
    return _frame("skin_feedback")
