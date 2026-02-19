# Project Context: recursive_stability
# Path: C:\Projects\policy\recursive_stability
# Generated for AI Review


==================================================
FILE: analysis.py
==================================================

# analysis.py

import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import linregress

def analyze(db_path):

    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM steps", conn)

    print("Mean energy:", df.energy.mean())

    slopes = []
    for pid in df.problem_id.unique():
        sub = df[df.problem_id == pid]
        if len(sub) > 2:
            slope, *_ = linregress(sub.iteration, sub.energy)
            slopes.append(slope)

    print("Mean slope:", np.mean(slopes))

    # Intervention Recovery Delta
    interventions = df[df.action != "ACCEPT"]

    deltas = []
    for _, row in interventions.iterrows():
        next_rows = df[
            (df.problem_id == row.problem_id) &
            (df.iteration > row.iteration)
        ].head(2)
        if len(next_rows) == 2:
            delta = next_rows.iloc[-1].accuracy - row.accuracy
            deltas.append(delta)

    print("Mean IRD:", np.mean(deltas) if deltas else 0)


==================================================
FILE: config.py
==================================================

# config.py

MODEL_NAME = "mistral"  # change as needed
OLLAMA_URL = "http://localhost:11434/api/generate"

INITIAL_TEMPERATURE = 1.1
NUM_PROBLEMS = 200
NUM_RECURSIONS = 10

# Energy thresholds (can adjust)
TAU_SOFT = 0.45
TAU_MEDIUM = 0.63
TAU_HARD = 1.54

DATABASE_PATH = "experiment.db"


==================================================
FILE: db.py
==================================================

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


==================================================
FILE: energy.py
==================================================

# energy.py

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_energy(prompt, current, previous=None):
    p_emb = model.encode(prompt, convert_to_tensor=True)
    c_emb = model.encode(current, convert_to_tensor=True)

    grounding = 1 - util.cos_sim(c_emb, p_emb).item()

    stability = 0
    if previous:
        prev_emb = model.encode(previous, convert_to_tensor=True)
        stability = 1 - util.cos_sim(c_emb, prev_emb).item()

    return grounding + stability, grounding, stability

def foreign_char_ratio(text):
    if not text:
        return 0
    return sum(1 for c in text if ord(c) > 127) / len(text)


==================================================
FILE: forensic_logger.py
==================================================

#!/usr/bin/env python3
"""
Forensic Logger: Crash-proof instrumentation for recursive instability analysis.
Captures energy, foreign bursts, repetition, accuracy, and phase transitions.
Designed for XGBoost-ready feature extraction post-hoc.
"""
import sqlite3
import time
import re
import unicodedata
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ForensicLogger:
    """Production-grade forensic logger with crash recovery and signal enrichment"""
    
    def __init__(self, db_path: str = "recursive_instability.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._configure_connection()
        self._create_tables()
        logger.info(f"ForensicLogger initialized: {db_path}")
    
    def _configure_connection(self):
        """Optimize for crash safety + write speed"""
        self.conn.execute("PRAGMA journal_mode=WAL;")      # Crash-safe writes
        self.conn.execute("PRAGMA synchronous=NORMAL;")   # Faster commits
        self.conn.execute("PRAGMA foreign_keys=ON;")       # Enforce relationships
    
    def _create_tables(self):
        """Create schema with all signal dimensions"""
        # Runs table (experiment metadata)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                temperature REAL NOT NULL,
                num_problems INTEGER NOT NULL,
                num_recursions INTEGER NOT NULL,
                tau_soft REAL NOT NULL,
                tau_medium REAL NOT NULL,
                tau_hard REAL NOT NULL,
                policy_id INTEGER NOT NULL,
                task_type TEXT NOT NULL,  -- 'gsm8k', 'summarization', etc.
                start_time REAL NOT NULL,
                end_time REAL,
                status TEXT DEFAULT 'running',
                notes TEXT
            )
        """)
        
        # Steps table (per-iteration forensic data)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                problem_id INTEGER NOT NULL,
                iteration INTEGER NOT NULL,
                prompt_text TEXT NOT NULL,
                reasoning_text TEXT NOT NULL,
                gold_answer TEXT,          -- For accuracy computation
                extracted_answer TEXT,     -- Model's answer extraction
                
                -- Core energy metrics
                energy REAL NOT NULL,
                grounding_energy REAL NOT NULL,
                stability_energy REAL NOT NULL,
                
                -- Signal discovery metrics (YOUR XGBOOST FEATURES)
                foreign_char_ratio REAL NOT NULL,  -- Non-ASCII burst detector
                ascii_ratio REAL NOT NULL,         -- Complement of foreign
                repetition_score REAL NOT NULL,    -- Token repetition metric
                text_length INTEGER NOT NULL,
                token_count INTEGER,
                
                -- Outcome metrics (CRITICAL FOR CAUSALITY)
                accuracy REAL,             -- Computed vs gold answer
                correctness BOOLEAN,       -- Binary correct/incorrect
                
                -- System state
                temperature REAL NOT NULL,
                policy_action TEXT NOT NULL,  -- 'accept', 'revert', 'reset', etc.
                phase TEXT NOT NULL,          -- 'stable', 'drift', 'unstable', 'collapse'
                
                -- Timestamps
                timestamp REAL NOT NULL,
                
                FOREIGN KEY (run_id) REFERENCES runs(run_id),
                UNIQUE(run_id, problem_id, iteration)
            )
        """)
        
        # Interventions table (policy actions)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS interventions (
                intervention_id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_id INTEGER NOT NULL,
                threshold_crossed TEXT,  -- 'soft', 'medium', 'hard'
                rationale TEXT,
                reverted_to_iteration INTEGER,
                new_temperature REAL,
                FOREIGN KEY (step_id) REFERENCES steps(step_id)
            )
        """)
        
        # Indexes for fast querying (critical at scale)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_steps_run_problem 
            ON steps(run_id, problem_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_steps_phase 
            ON steps(phase)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_steps_energy 
            ON steps(energy DESC)
        """)
        
        self.conn.commit()
    
    # ============================================================================
    # SIGNAL COMPUTATION (Your XGBoost Feature Pipeline)
    # ============================================================================
    
    @staticmethod
    def compute_foreign_char_ratio(text: str) -> float:
        """
        YOUR OBSERVATION: Non-ASCII bursts precede collapse.
        Measures ratio of characters outside ASCII range (Chinese, Arabic, etc.)
        """
        if not text:
            return 0.0
        # Count characters with Unicode category starting with 'L' (Letter) but outside ASCII
        foreign_chars = sum(
            1 for c in text 
            if ord(c) > 127 and unicodedata.category(c).startswith('L')
        )
        return foreign_chars / len(text) if text else 0.0
    
    @staticmethod
    def compute_ascii_ratio(text: str) -> float:
        """Complement of foreign ratio - sometimes more predictive"""
        if not text:
            return 1.0
        ascii_chars = sum(1 for c in text if ord(c) <= 127)
        return ascii_chars / len(text) if text else 1.0
    
    @staticmethod
    def compute_repetition_score(text: str) -> float:
        """
        Measures semantic thinning: token repetition indicates "playing it safe"
        Higher = more repetition = less information gain
        """
        if not text or len(text) < 10:
            return 0.0
        
        # Simple n-gram repetition (adjust n for sensitivity)
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 3:
            return 0.0
        
        # Count repeated trigrams
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        if not trigrams:
            return 0.0
        
        unique_trigrams = set(trigrams)
        repetition = 1.0 - (len(unique_trigrams) / len(trigrams))
        return repetition
    
    @staticmethod
    def compute_phase(energy: float, tau_soft: float, tau_medium: float, tau_hard: float) -> str:
        """
        Phase segmentation for system dynamics analysis:
        stable → drift → unstable → collapse
        """
        if energy < tau_soft:
            return "stable"
        elif energy < tau_medium:
            return "drift"
        elif energy < tau_hard:
            return "unstable"
        else:
            return "collapse"
    
    # ============================================================================
    # ACCURACY COMPUTATION (CRITICAL FOR OUTCOME VALIDATION)
    # ============================================================================
    
    @staticmethod
    def compute_accuracy_gsm8k(reasoning: str, gold_answer: str) -> Dict[str, Any]:
        """
        GSM8K-specific accuracy extraction.
        Replace with task-specific logic for other datasets.
        
        Returns:
            {
                "accuracy": float (0.0 or 1.0),
                "correctness": bool,
                "extracted_answer": str,
                "gold_answer": str
            }
        """
        # Simple regex to extract final answer (adjust for your format)
        # Example: "The answer is 42." → "42"
        match = re.search(r'(?:answer is|final answer|therefore)\s*[:\-]?\s*(\d+)', reasoning, re.IGNORECASE)
        extracted = match.group(1) if match else None
        
        if extracted is None:
            # Fallback: last number in text
            numbers = re.findall(r'\d+', reasoning)
            extracted = numbers[-1] if numbers else None
        
        correctness = extracted == gold_answer.strip() if extracted else False
        accuracy = 1.0 if correctness else 0.0
        
        return {
            "accuracy": accuracy,
            "correctness": correctness,
            "extracted_answer": extracted,
            "gold_answer": gold_answer
        }
    
    # ============================================================================
    # LOGGING METHODS (Crash-Proof Pattern)
    # ============================================================================
    
    def start_run(self, config: Dict[str, Any]) -> str:
        """Start new experiment run"""
        run_id = f"run_{int(time.time())}_{config['policy_id']}"
        self.conn.execute("""
            INSERT INTO runs (
                run_id, model_name, temperature, num_problems, num_recursions,
                tau_soft, tau_medium, tau_hard, policy_id, task_type,
                start_time, status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
        """, (
            run_id,
            config['model_name'],
            config['temperature'],
            config['num_problems'],
            config['num_recursions'],
            config['tau_soft'],
            config['tau_medium'],
            config['tau_hard'],
            config['policy_id'],
            config.get('task_type', 'gsm8k'),
            time.time(),
            config.get('notes', '')
        ))
        self.conn.commit()
        return run_id
    
    def log_step(
        self,
        run_id: str,
        problem_id: int,
        iteration: int,
        prompt: str,
        reasoning: str,
        gold_answer: Optional[str],
        energy_metrics: Dict[str, float],
        temperature: float,
        policy_action: str,
        config: Dict[str, Any]
    ):
        """
        Log single step with ALL signal dimensions.
        COMMIT IMMEDIATELY - survives crashes.
        """
        # Compute signal discovery metrics
        foreign_ratio = self.compute_foreign_char_ratio(reasoning)
        ascii_ratio = self.compute_ascii_ratio(reasoning)
        repetition = self.compute_repetition_score(reasoning)
        
        # Compute phase segmentation
        phase = self.compute_phase(
            energy_metrics['total_energy'],
            config['tau_soft'],
            config['tau_medium'],
            config['tau_hard']
        )
        
        # Compute accuracy (CRITICAL FOR CAUSALITY)
        accuracy_data = {"accuracy": None, "correctness": None, "extracted_answer": None}
        if gold_answer:
            accuracy_data = self.compute_accuracy_gsm8k(reasoning, gold_answer)
        
        # Insert step
        self.conn.execute("""
            INSERT INTO steps (
                run_id, problem_id, iteration, prompt_text, reasoning_text,
                gold_answer, extracted_answer,
                energy, grounding_energy, stability_energy,
                foreign_char_ratio, ascii_ratio, repetition_score,
                text_length, token_count,
                accuracy, correctness,
                temperature, policy_action, phase,
                timestamp
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?,
                ?
            )
        """, (
            run_id, problem_id, iteration, prompt, reasoning,
            gold_answer, accuracy_data['extracted_answer'],
            energy_metrics['total_energy'],
            energy_metrics['grounding_energy'],
            energy_metrics['stability_energy'],
            foreign_ratio, ascii_ratio, repetition,
            len(reasoning), None,  # token_count requires tokenizer
            accuracy_data['accuracy'], accuracy_data['correctness'],
            temperature, policy_action, phase,
            time.time()
        ))
        self.conn.commit()  # CRITICAL: Survives crash at next iteration
    
    def log_intervention(
        self,
        run_id: str,
        problem_id: int,
        iteration: int,
        threshold: str,
        rationale: str,
        reverted_to: Optional[int],
        new_temp: Optional[float]
    ):
        """Log policy intervention details"""
        # Get step_id for this (run_id, problem_id, iteration)
        cursor = self.conn.execute("""
            SELECT step_id FROM steps
            WHERE run_id = ? AND problem_id = ? AND iteration = ?
        """, (run_id, problem_id, iteration))
        result = cursor.fetchone()
        if not result:
            logger.warning(f"Intervention log failed: step not found for {run_id}/{problem_id}/{iteration}")
            return
        
        step_id = result[0]
        self.conn.execute("""
            INSERT INTO interventions (
                step_id, threshold_crossed, rationale, 
                reverted_to_iteration, new_temperature
            ) VALUES (?, ?, ?, ?, ?)
        """, (step_id, threshold, rationale, reverted_to, new_temp))
        self.conn.commit()
    
    def mark_run_complete(self, run_id: str):
        """Mark run as completed"""
        self.conn.execute(
            "UPDATE runs SET end_time = ?, status = 'completed' WHERE run_id = ?",
            (time.time(), run_id)
        )
        self.conn.commit()
    
    def get_resume_point(self, run_id: str, problem_id: int) -> int:
        """Get last completed iteration for resume capability"""
        cursor = self.conn.execute("""
            SELECT MAX(iteration) FROM steps 
            WHERE run_id = ? AND problem_id = ?
        """, (run_id, problem_id))
        result = cursor.fetchone()
        return result[0] if result and result[0] is not None else -1
    
    def close(self):
        """Close database connection"""
        self.conn.close()

==================================================
FILE: maiun.py
==================================================

# main.py

from runner import run
from analysis import analyze

if __name__ == "__main__":

    # Run baseline
    run(policy_id=0)

    # Run progressive clamp
    run(policy_id=5)

    # Analyze
    analyze("experiment.db")


==================================================
FILE: model.py
==================================================

# model.py

import requests
from config import OLLAMA_URL, MODEL_NAME

def call_model(prompt, temperature):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]


==================================================
FILE: policy.py
==================================================

# policy.py

from config import TAU_SOFT, TAU_MEDIUM, TAU_HARD

def apply_policy(policy_id, energy, reasoning, last_stable, prompt, temperature):

    if policy_id == 0:
        return reasoning, temperature, "ACCEPT"

    if energy <= TAU_SOFT:
        return reasoning, temperature, "ACCEPT"

    if policy_id == 1:
        return last_stable, temperature, "REVERT"

    if policy_id == 2:
        return last_stable, temperature * 0.9, "REVERT_COOL"

    if policy_id == 3:
        if energy > TAU_MEDIUM:
            return last_stable, temperature * 0.75, "REVERT_AGGRESSIVE"
        return last_stable, temperature, "REVERT"

    if policy_id == 4:
        if energy > TAU_HARD:
            return prompt, temperature * 0.7, "RESET_PROMPT"
        return last_stable, temperature, "REVERT"

    if policy_id == 5:
        if energy > TAU_MEDIUM:
            return prompt, temperature * 0.85, "RESET"
        return last_stable, temperature * 0.85, "REVERT_STABILIZE"

    return reasoning, temperature, "ACCEPT"


==================================================
FILE: runner.py
==================================================

# runner.py

import time
import uuid
from datasets import load_dataset
from .energy import compute_energy, foreign_char_ratio
from .policy import apply_policy
from .model import call_model
from .db import ExperimentDB
from .config import DATABASE_PATH, MODEL_NAME, INITIAL_TEMPERATURE, NUM_PROBLEMS, NUM_RECURSIONS

def extract_number(text):
    import re
    nums = re.findall(r"\d+\.?\d*", text)
    return float(nums[-1]) if nums else None

def run(policy_id=0):

    run_id = str(uuid.uuid4())
    db = ExperimentDB(DATABASE_PATH)
    db.start_run(run_id, MODEL_NAME, INITIAL_TEMPERATURE, policy_id)

    dataset = load_dataset("gsm8k", "main", split="test").select(range(NUM_PROBLEMS))

    for pid, example in enumerate(dataset):

        prompt = example["question"]
        gold_answer = extract_number(example["answer"])

        state = prompt
        last_stable = prompt
        temperature = INITIAL_TEMPERATURE
        prev_reasoning = None

        for iteration in range(NUM_RECURSIONS):

            reasoning = call_model(
                f"Solve step by step:\n\n{state}",
                temperature
            )

            energy, grounding, stability = compute_energy(
                prompt, reasoning, prev_reasoning
            )

            new_state, temperature, action = apply_policy(
                policy_id, energy, reasoning, last_stable, prompt, temperature
            )

            if action == "ACCEPT":
                last_stable = reasoning

            predicted = extract_number(reasoning)
            accuracy = 1 if predicted == gold_answer else 0

            db.log_step((
                run_id,
                pid,
                iteration,
                prompt,
                reasoning,
                energy,
                grounding,
                stability,
                temperature,
                action,
                foreign_char_ratio(reasoning),
                accuracy,
                time.time()
            ))

            prev_reasoning = reasoning
            state = new_state

    db.finish_run(run_id)


==================================================
FILE: signal_discovery.py
==================================================

#!/usr/bin/env python3
"""
XGBoost Signal Discovery Pipeline
Identify which metrics PRECEDE collapse (lead-lag validation)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

def load_forensic_data(db_path: str = "recursive_instability.db") -> pd.DataFrame:
    """Load all steps with collapse labels"""
    conn = sqlite3.connect(db_path)
    
    # Load steps with collapse labels
    query = """
    SELECT 
        s.*,
        CASE 
            WHEN s.phase = 'collapse' THEN 1
            ELSE 0
        END as collapse_label,
        LEAD(s.phase, 1) OVER (PARTITION BY s.run_id, s.problem_id ORDER BY s.iteration) as next_phase,
        LEAD(s.phase, 2) OVER (PARTITION BY s.run_id, s.problem_id ORDER BY s.iteration) as phase_t_plus_2,
        LEAD(s.phase, 3) OVER (PARTITION BY s.run_id, s.problem_id ORDER BY s.iteration) as phase_t_plus_3
    FROM steps s
    WHERE s.iteration < (SELECT MAX(iteration) FROM steps s2 WHERE s2.run_id = s.run_id AND s2.problem_id = s.problem_id)
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_lead_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for predicting collapse at t+k"""
    features = df.copy()
    
    # Lead indicators: signals at t that predict collapse at t+1, t+2, t+3
    for lag in [1, 2, 3]:
        features[f'collapse_t_plus_{lag}'] = (
            features[f'phase_t_plus_{lag}'] == 'collapse'
        ).astype(int)
    
    # Rolling statistics (window = 3 iterations)
    for col in ['energy', 'foreign_char_ratio', 'repetition_score']:
        features[f'{col}_rolling_mean'] = features.groupby(['run_id', 'problem_id'])[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        features[f'{col}_rolling_std'] = features.groupby(['run_id', 'problem_id'])[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).std().fillna(0)
        )
    
    return features

def train_collapse_predictor(df: pd.DataFrame, prediction_horizon: int = 2):
    """
    Train XGBoost to predict collapse at t+prediction_horizon
    Returns model + feature importance
    """
    # Target: collapse at t+prediction_horizon
    target_col = f'collapse_t_plus_{prediction_horizon}'
    
    # Features: all signal metrics at time t
    feature_cols = [
        'energy', 'grounding_energy', 'stability_energy',
        'foreign_char_ratio', 'ascii_ratio', 'repetition_score',
        'text_length', 'accuracy',
        'energy_rolling_mean', 'energy_rolling_std',
        'foreign_char_ratio_rolling_mean', 'foreign_char_ratio_rolling_std'
    ]
    
    # Drop rows with missing targets
    df_clean = df.dropna(subset=[target_col] + feature_cols)
    
    # Train/test split (time-series aware)
    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    aucs = []
    
    for train_idx, test_idx in tscv.split(df_clean):
        X_train = df_clean.iloc[train_idx][feature_cols]
        y_train = df_clean.iloc[train_idx][target_col]
        X_test = df_clean.iloc[test_idx][feature_cols]
        y_test = df_clean.iloc[test_idx][target_col]
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        aucs.append(auc_score)
        models.append(model)
    
    # Average model
    avg_auc = np.mean(aucs)
    print(f"Predicting collapse at t+{prediction_horizon}: AUC = {avg_auc:.3f}")
    
    # Feature importance from last fold
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': models[-1].feature_importances_
    }).sort_values('importance', ascending=False)
    
    return models[-1], importance, avg_auc

def plot_signal_lead_lag(df: pd.DataFrame):
    """Visualize which signals precede collapse"""
    # Filter to trajectories that collapse
    collapse_trajectories = df[df['phase'] == 'collapse']['problem_id'].unique()
    collapse_df = df[df['problem_id'].isin(collapse_trajectories)].copy()
    
    # Compute average signal trajectory before collapse
    collapse_df['steps_to_collapse'] = collapse_df.groupby(['run_id', 'problem_id'])['iteration'].transform(
        lambda x: (x.max() - x)
    )
    
    # Aggregate signals by steps before collapse
    signal_agg = collapse_df[collapse_df['steps_to_collapse'] <= 5].groupby('steps_to_collapse')[
        ['energy', 'foreign_char_ratio', 'repetition_score', 'accuracy']
    ].mean().reset_index()
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['energy', 'foreign_char_ratio', 'repetition_score', 'accuracy']
    titles = ['Hallucination Energy', 'Foreign Char Ratio', 'Repetition Score', 'Accuracy']
    colors = ['red', 'purple', 'orange', 'green']
    
    for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        ax = axes[idx // 2, idx % 2]
        ax.plot(signal_agg['steps_to_collapse'], signal_agg[metric], marker='o', linewidth=2.5, color=color)
        ax.set_xlabel('Steps Before Collapse')
        ax.set_ylabel(metric)
        ax.set_title(f'{title} Trajectory Before Collapse')
        ax.invert_xaxis()  # Show time moving forward to collapse
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('signal_lead_lag.png', dpi=300, bbox_inches='tight')
    print("✅ Signal lead-lag plot saved to signal_lead_lag.png")

def main():
    print("="*70)
    print("SIGNAL DISCOVERY: Identifying Predictors of Recursive Collapse")
    print("="*70)
    
    # Load data
    print("\n1. Loading forensic data...")
    df = load_forensic_data()
    print(f"   Loaded {len(df)} steps from {df['run_id'].nunique()} runs")
    
    # Create features
    print("\n2. Creating lead-lag features...")
    df_features = create_lead_lag_features(df)
    
    # Train predictors for different horizons
    print("\n3. Training collapse predictors...")
    results = {}
    for horizon in [1, 2, 3]:
        model, importance, auc_score = train_collapse_predictor(df_features, prediction_horizon=horizon)
        results[horizon] = {'model': model, 'importance': importance, 'auc': auc_score}
        
        print(f"\nTop 5 predictors for collapse at t+{horizon}:")
        print(importance.head(5).to_string(index=False))
    
    # Plot signal trajectories
    print("\n4. Generating signal lead-lag visualization...")
    plot_signal_lead_lag(df)
    
    # Critical finding: Foreign char ratio predictive power
    print("\n" + "="*70)
    print("KEY FINDING: Foreign Character Ratio as Early Warning Signal")
    print("="*70)
    for horizon in [1, 2, 3]:
        foreign_rank = results[horizon]['importance'][
            results[horizon]['importance']['feature'] == 'foreign_char_ratio'
        ].index[0] + 1
        print(f"  • Foreign char ratio rank for t+{horizon}: #{foreign_rank}")
    
    print("\n✅ Signal discovery complete. Check signal_lead_lag.png for visualization.")
    print("   Use results[horizon]['model'] for deployment or further analysis.")

if __name__ == "__main__":
    main()

==================================================
FILE: __init__.py
==================================================


