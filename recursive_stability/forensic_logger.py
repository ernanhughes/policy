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