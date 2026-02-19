# db.py

import sqlite3
import time

class ExperimentDB:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            model TEXT,
            temperature REAL,
            policy_id INTEGER,
            start_time REAL,
            status TEXT
        )
        """)

        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            problem_id INTEGER,
            iteration INTEGER,
            prompt TEXT,
            reasoning TEXT,
            energy REAL,
            grounding REAL,
            stability REAL,
            temperature REAL,
            action TEXT,
            foreign_ratio REAL,
            accuracy REAL,
            token_count INTEGER,
            length_growth REAL,
            repetition_ratio REAL,
            timestamp REAL
        )
        """)

        self.conn.commit()

    def start_run(self, run_id, model, temperature, policy_id):
        self.conn.execute("""
        INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)
        """, (run_id, model, temperature, policy_id, time.time(), "running"))
        self.conn.commit()

    def log_step(self, row):
        self.conn.execute("""
        INSERT INTO steps 
        (run_id, problem_id, iteration, prompt, reasoning, energy, grounding,
         stability, temperature, action, foreign_ratio, accuracy, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, row)
        self.conn.commit()

    def finish_run(self, run_id):
        self.conn.execute("""
        UPDATE runs SET status='completed' WHERE run_id=?
        """, (run_id,))
        self.conn.commit()
