# runner.py

import time
import uuid
from datasets import load_dataset

from .energy import compute_energy, foreign_char_ratio
from .policy import apply_policy
from .model import call_model
from .db import ExperimentDB
from .forensic_logger import ForensicLogger
from .config import DATABASE_PATH, MODEL_NAME, INITIAL_TEMPERATURE, NUM_PROBLEMS, NUM_RECURSIONS


logger = ForensicLogger(DATABASE_PATH)

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

            token_count = len(reasoning.split())
            if prev_reasoning:
                prev_tokens = prev_reasoning.split()
                length_growth = token_count / len(prev_tokens)
                rep_ratio = len(set(reasoning.split()) & set(prev_reasoning.split())) / max(token_count, 1)
            else:
                length_growth = 1.0
                rep_ratio = 0.0

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
