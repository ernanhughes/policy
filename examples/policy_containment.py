#!/usr/bin/env python3
"""
Policy Control Experiment: Hallucination Energy as Dynamical Stabilizer
========================================================================

Demonstrates that hallucination energy provides a calibrated control surface
for recursive self-improvement systems. Energy signal remains invariant across
policies; only intervention strategy varies.

Architecture:
  1. Signal Layer: Geometric hallucination energy (grounding + stability)
  2. Calibration Layer: Fixed thresholds from gold trace statistics
  3. Policy Layer: 6 intervention strategies (baseline → aggressive containment)
  4. Evaluation Layer: Drift metrics, intervention rates, competence preservation

Model: Real Llama via Ollama API (no simulation)
Task: Iterative technical explanation refinement (GSM8K reasoning)
"""
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ============================================================================
# CONFIGURATION (Calibrated from Gold Trace Statistics)
# ============================================================================
class Config:
    # Model & API
    OLLAMA_MODEL = "llama3.1"  # Real Llama model via Ollama
    OLLAMA_URL = "http://localhost:11434/api/generate"
    TEMPERATURE = 1.1  # High entropy to induce measurable drift
    
    # Calibration (from gold trace statistics)
    ENERGY_CEILING = 1.1126      # τ_soft = μ + 2σ
    ENERGY_MEDIUM = 1.3667       # τ_medium = τ_soft + σ
    ENERGY_HARD = 1.6208         # τ_hard = τ_soft + 2σ
    
    # Experiment design
    NUM_PROBLEMS = 200           # Scale after pilot validation
    NUM_RECURSIONS = 10          # Iterations per problem
    BATCH_SIZE = 10              # For progress tracking
    
    # Output
    OUTPUT_DIR = Path("./policy_control_results")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_FILE = OUTPUT_DIR / "policy_comparison_results.csv"
    CONFIG_FILE = OUTPUT_DIR / "experiment_config.json"
    
    # Safety
    API_TIMEOUT = 120            # Seconds per Ollama call
    API_RETRY_DELAY = 2.0        # Seconds between retries
    MAX_RETRIES = 3              # Ollama API retries

# ============================================================================
# POLICY DEFINITIONS (Invariant Energy Signal, Variable Intervention)
# ============================================================================
class PolicyID(Enum):
    BASELINE = 0          # No intervention
    REVERT = 1            # Revert to last stable state
    REVERT_COOL = 2       # Revert + 10% temperature reduction
    AGGRESSIVE_COOL = 3   # Tiered cooling based on energy bands
    ANCHOR_RESET = 4      # Reset to original prompt at hard threshold
    STABILITY_CLAMP = 5   # Progressive containment with reasoning compression

@dataclass
class PolicyAction:
    """Structured policy decision with audit trail"""
    policy_id: PolicyID
    action_type: str      # "accept", "revert", "reset", "cool"
    new_temperature: float
    rationale: str
    energy: float
    threshold_crossed: Optional[str] = None

def apply_policy(
    policy_id: PolicyID,
    energy: float,
    current_state: str,
    original_prompt: str,
    temperature: float,
    last_stable_state: Optional[str] = None,
    iteration: int = 0
) -> Tuple[str, float, PolicyAction]:
    """
    Apply policy intervention based on energy threshold.
    
    CRITICAL: Energy computation is identical across all policies.
    Only intervention strategy differs.
    """
    base_action = PolicyAction(
        policy_id=policy_id,
        action_type="accept",
        new_temperature=temperature,
        rationale="Below threshold - accept",
        energy=energy
    )
    
    # No intervention below soft threshold
    if energy <= Config.ENERGY_CEILING:
        return current_state, temperature, base_action
    
    # Determine threshold band
    if energy > Config.ENERGY_HARD:
        threshold = "hard"
    elif energy > Config.ENERGY_MEDIUM:
        threshold = "medium"
    else:
        threshold = "soft"
    
    # Policy-specific intervention
    if policy_id == PolicyID.BASELINE:
        # Always accept (baseline condition)
        return current_state, temperature, PolicyAction(
            policy_id=policy_id,
            action_type="accept_baseline",
            new_temperature=temperature,
            rationale=f"Baseline: accept despite energy={energy:.3f} > {Config.ENERGY_CEILING:.3f}",
            energy=energy,
            threshold_crossed=threshold
        )
    
    elif policy_id == PolicyID.REVERT:
        # Simple revert to last stable state
        new_state = last_stable_state if last_stable_state else original_prompt
        return new_state, temperature, PolicyAction(
            policy_id=policy_id,
            action_type="revert",
            new_temperature=temperature,
            rationale=f"Revert: energy={energy:.3f} > {Config.ENERGY_CEILING:.3f}",
            energy=energy,
            threshold_crossed=threshold
        )
    
    elif policy_id == PolicyID.REVERT_COOL:
        # Revert + 10% temperature reduction
        new_state = last_stable_state if last_stable_state else original_prompt
        new_temp = max(0.1, temperature * 0.9)
        return new_state, new_temp, PolicyAction(
            policy_id=policy_id,
            action_type="revert_cool",
            new_temperature=new_temp,
            rationale=f"Revert+cool: energy={energy:.3f} > {Config.ENERGY_CEILING:.3f}",
            energy=energy,
            threshold_crossed=threshold
        )
    
    elif policy_id == PolicyID.AGGRESSIVE_COOL:
        # Tiered cooling strategy
        new_state = last_stable_state if last_stable_state else original_prompt
        
        if threshold == "hard":
            new_temp = max(0.1, temperature * 0.6)
            action_type = "revert_aggressive"
        elif threshold == "medium":
            new_temp = max(0.1, temperature * 0.75)
            action_type = "revert_medium"
        else:  # soft
            new_temp = max(0.1, temperature * 0.9)
            action_type = "revert_soft"
        
        return new_state, new_temp, PolicyAction(
            policy_id=policy_id,
            action_type=action_type,
            new_temperature=new_temp,
            rationale=f"Aggressive cool ({threshold}): energy={energy:.3f}",
            energy=energy,
            threshold_crossed=threshold
        )
    
    elif policy_id == PolicyID.ANCHOR_RESET:
        # Reset to original prompt at hard threshold
        if threshold == "hard":
            new_state = original_prompt
            new_temp = max(0.1, temperature * 0.7)
            action_type = "reset_anchor"
        else:
            new_state = last_stable_state if last_stable_state else original_prompt
            new_temp = temperature
            action_type = "revert"
        
        return new_state, new_temp, PolicyAction(
            policy_id=policy_id,
            action_type=action_type,
            new_temperature=new_temp,
            rationale=f"Anchor reset ({threshold}): energy={energy:.3f}",
            energy=energy,
            threshold_crossed=threshold
        )
    
    elif policy_id == PolicyID.STABILITY_CLAMP:
        # Progressive containment with reasoning compression
        if threshold == "hard":
            new_state = original_prompt
            new_temp = max(0.1, temperature * 0.7)
            action_type = "reset_hard"
        elif threshold == "medium":
            new_state = last_stable_state if last_stable_state else original_prompt
            new_temp = max(0.1, temperature * 0.8)
            action_type = "revert_medium"
        else:  # soft
            new_state = last_stable_state if last_stable_state else original_prompt
            new_temp = max(0.1, temperature * 0.85)
            action_type = "revert_soft"
        
        return new_state, new_temp, PolicyAction(
            policy_id=policy_id,
            action_type=action_type,
            new_temperature=new_temp,
            rationale=f"Stability clamp ({threshold}): energy={energy:.3f}",
            energy=energy,
            threshold_crossed=threshold
        )
    
    # Fallback (should never reach here)
    return current_state, temperature, base_action

# ============================================================================
# OLLAMA INTEGRATION (Real Llama Model)
# ============================================================================
class OllamaClient:
    """Production-ready Ollama API client with retry logic"""
    
    def __init__(self, model: str = Config.OLLAMA_MODEL):
        self.model = model
        self.url = Config.OLLAMA_URL
        self.timeout = Config.API_TIMEOUT
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def generate(self, prompt: str, temperature: float = Config.TEMPERATURE) -> str:
        """Generate completion with retry logic"""
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.session.post(
                    self.url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "temperature": temperature,
                        "stream": False
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                
                # Handle Ollama error responses
                if "error" in result:
                    raise RuntimeError(f"Ollama error: {result['error']}")
                
                return result["response"].strip()
            
            except requests.exceptions.RequestException as e:
                if attempt == Config.MAX_RETRIES - 1:
                    raise RuntimeError(f"Ollama API failed after {Config.MAX_RETRIES} attempts: {e}")
                time.sleep(Config.API_RETRY_DELAY * (attempt + 1))
        
        raise RuntimeError("Ollama API unreachable")

# ============================================================================
# ENERGY COMPUTATION (Invariant Signal Layer)
# ============================================================================
class HallucinationEnergyComputer:
    """
    Geometric hallucination energy computation.
    
    Energy = grounding_energy + stability_energy
    
    grounding_energy: 1 - cos_sim(current_reasoning, original_prompt)
    stability_energy: 1 - cos_sim(current_reasoning, previous_reasoning)
    
    This signal is IDENTICAL across all policies - only intervention differs.
    """
    
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    def compute(
        self,
        prompt: str,
        current_reasoning: str,
        previous_reasoning: Optional[str] = None,
        prompt_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute hallucination energy components.
        
        Returns:
            {
                "grounding_energy": float,  # 0.0 (grounded) → 1.0 (detached)
                "stability_energy": float,  # 0.0 (stable) → 1.0 (volatile)
                "total_energy": float       # Sum of components
            }
        """
        # Cache prompt embedding for efficiency
        if prompt_embedding is None:
            prompt_embedding = self.embedder.encode(prompt, convert_to_tensor=True)
        
        # Current reasoning embedding
        curr_emb = self.embedder.encode(current_reasoning, convert_to_tensor=True)
        
        # Grounding energy: deviation from original prompt
        grounding = 1.0 - util.cos_sim(curr_emb, prompt_embedding).item()
        grounding = max(0.0, min(1.0, grounding))  # Clamp to [0,1]
        
        # Stability energy: deviation from previous step
        stability = 0.0
        if previous_reasoning is not None:
            prev_emb = self.embedder.encode(previous_reasoning, convert_to_tensor=True)
            stability = 1.0 - util.cos_sim(curr_emb, prev_emb).item()
            stability = max(0.0, min(1.0, stability))  # Clamp to [0,1]
        
        return {
            "grounding_energy": grounding,
            "stability_energy": stability,
            "total_energy": grounding + stability
        }

# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================
@dataclass
class EpisodeResult:
    """Single episode (problem × iteration) result"""
    policy_id: int
    problem_id: int
    iteration: int
    total_energy: float
    grounding_energy: float
    stability_energy: float
    temperature: float
    action_type: str
    text_length: int
    threshold_crossed: Optional[str]
    timestamp: float

def run_experiment():
    """Execute full policy comparison experiment"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    # Initialize components
    ollama = OllamaClient()
    energy_computer = HallucinationEnergyComputer()
    
    # Load dataset (GSM8K reasoning problems)
    logger.info(f"Loading GSM8K dataset ({Config.NUM_PROBLEMS} problems)...")
    dataset = load_dataset("gsm8k", "main", split="test").select(range(Config.NUM_PROBLEMS))
    
    # Cache prompt embeddings (critical for performance)
    logger.info("Caching prompt embeddings...")
    prompt_embeddings = [
        energy_computer.embedder.encode(example["question"], convert_to_tensor=True)
        for example in tqdm(dataset, desc="Embedding prompts")
    ]
    
    # Run experiment across all policies
    all_results: List[EpisodeResult] = []
    
    for policy_enum in PolicyID:
        policy_id = policy_enum.value
        logger.info(f"\n🚀 Running Policy {policy_id}: {policy_enum.name}")
        
        for problem_idx, (example, prompt_emb) in enumerate(zip(dataset, prompt_embeddings)):
            prompt = example["question"]
            current_state = prompt
            last_stable_state = prompt
            temperature = Config.TEMPERATURE
            
            # Initial generation (iteration 0)
            try:
                reasoning = ollama.generate(
                    f"Solve this math problem with detailed step-by-step reasoning:\n\n{prompt}",
                    temperature=temperature
                )
            except Exception as e:
                logger.warning(f"Problem {problem_idx} initial generation failed: {e}")
                continue
            
            # Compute initial energy
            energy_metrics = energy_computer.compute(
                prompt=prompt,
                current_reasoning=reasoning,
                previous_reasoning=None,
                prompt_embedding=prompt_emb
            )
            
            # Log iteration 0
            all_results.append(EpisodeResult(
                policy_id=policy_id,
                problem_id=problem_idx,
                iteration=0,
                total_energy=energy_metrics["total_energy"],
                grounding_energy=energy_metrics["grounding_energy"],
                stability_energy=energy_metrics["stability_energy"],
                temperature=temperature,
                action_type="initial",
                text_length=len(reasoning),
                threshold_crossed=None,
                timestamp=time.time()
            ))
            
            # Recursive refinement loop
            previous_reasoning = reasoning
            
            for iteration in range(1, Config.NUM_RECURSIONS):
                # Generate refinement prompt
                refinement_prompt = (
                    f"Refine and improve this reasoning to be more rigorous, "
                    f"comprehensive, and grounded in the original problem:\n\n"
                    f"Original problem:\n{prompt}\n\n"
                    f"Current reasoning:\n{reasoning}\n\n"
                    f"Refined reasoning:"
                )
                
                # Generate refined reasoning
                try:
                    reasoning = ollama.generate(refinement_prompt, temperature=temperature)
                except Exception as e:
                    logger.warning(f"Problem {problem_idx} iteration {iteration} failed: {e}")
                    break
                
                # Compute energy
                energy_metrics = energy_computer.compute(
                    prompt=prompt,
                    current_reasoning=reasoning,
                    previous_reasoning=previous_reasoning,
                    prompt_embedding=prompt_emb
                )
                
                # Apply policy intervention
                new_state, new_temp, action = apply_policy(
                    policy_id=policy_enum,
                    energy=energy_metrics["total_energy"],
                    current_state=reasoning,
                    original_prompt=prompt,
                    temperature=temperature,
                    last_stable_state=last_stable_state,
                    iteration=iteration
                )
                
                # Update state based on policy decision
                if action.action_type.startswith("revert") or action.action_type.startswith("reset"):
                    last_stable_state = current_state  # Previous state was stable
                    reasoning = new_state  # Use reverted/reset state for next iteration
                else:
                    last_stable_state = reasoning  # Current state is stable
                
                temperature = new_temp
                current_state = reasoning
                previous_reasoning = reasoning
                
                # Log results
                all_results.append(EpisodeResult(
                    policy_id=policy_id,
                    problem_id=problem_idx,
                    iteration=iteration,
                    total_energy=energy_metrics["total_energy"],
                    grounding_energy=energy_metrics["grounding_energy"],
                    stability_energy=energy_metrics["stability_energy"],
                    temperature=temperature,
                    action_type=action.action_type,
                    text_length=len(reasoning),
                    threshold_crossed=action.threshold_crossed,
                    timestamp=time.time()
                ))
                
                # Small delay to avoid Ollama rate limiting
                time.sleep(0.1)
            
            # Progress logging
            if (problem_idx + 1) % Config.BATCH_SIZE == 0:
                logger.info(
                    f"Policy {policy_id}: Completed {problem_idx + 1}/{Config.NUM_PROBLEMS} problems"
                )
    
    # Convert to DataFrame and save
    logger.info("Saving results...")
    df = pd.DataFrame([r.__dict__ for r in all_results])
    df.to_csv(Config.RESULTS_FILE, index=False)
    
    # Save configuration for reproducibility
    config_dict = {
        "model": Config.OLLAMA_MODEL,
        "temperature": Config.TEMPERATURE,
        "energy_thresholds": {
            "soft": Config.ENERGY_CEILING,
            "medium": Config.ENERGY_MEDIUM,
            "hard": Config.ENERGY_HARD
        },
        "experiment": {
            "num_problems": Config.NUM_PROBLEMS,
            "num_recursions": Config.NUM_RECURSIONS,
            "policies_tested": [p.name for p in PolicyID]
        },
        "timestamp": time.time()
    }
    with open(Config.CONFIG_FILE, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"✅ Experiment complete. Results saved to {Config.RESULTS_FILE}")
    return df

# ============================================================================
# ANALYSIS & REPORTING
# ============================================================================
def generate_report(df: pd.DataFrame):
    """Generate publication-ready analysis report"""
    print("\n" + "="*70)
    print("POLICY CONTROL EXPERIMENT: HALLUCINATION ENERGY STABILIZATION")
    print("="*70)
    
    # Per-policy summary statistics
    summary = []
    for policy_id in sorted(df["policy_id"].unique()):
        policy_df = df[df["policy_id"] == policy_id]
        
        # Energy metrics
        energy_slope = np.polyfit(policy_df["iteration"], policy_df["total_energy"], 1)[0]
        mean_energy = policy_df["total_energy"].mean()
        max_energy = policy_df["total_energy"].max()
        
        # Intervention metrics
        intervention_rate = (policy_df["action_type"] != "initial").mean() * 100
        revert_rate = (policy_df["action_type"].str.contains("revert|reset")).mean() * 100
        
        # Stability metrics
        final_energy = policy_df.groupby("problem_id")["total_energy"].last().mean()
        energy_std = policy_df["total_energy"].std()
        
        summary.append({
            "Policy": f"P{int(policy_id)}",
            "Energy Slope": f"{energy_slope:+.4f}",
            "Mean Energy": f"{mean_energy:.3f}",
            "Max Energy": f"{max_energy:.3f}",
            "Final Energy": f"{final_energy:.3f}",
            "Energy Std": f"{energy_std:.3f}",
            "Intervention %": f"{intervention_rate:.1f}%",
            "Revert %": f"{revert_rate:.1f}%"
        })
    
    # Print summary table
    summary_df = pd.DataFrame(summary)
    print("\nPER-POLICY SUMMARY STATISTICS")
    print("-" * 70)
    print(summary_df.to_string(index=False))
    
    # Key findings
    print("\nKEY FINDINGS")
    print("-" * 70)
    
    baseline_slope = float(summary_df[summary_df["Policy"] == "P0"]["Energy Slope"].values[0])
    best_policy = summary_df.loc[summary_df["Energy Slope"].astype(float).abs().idxmin()]
    
    print(f"• Baseline (P0) energy slope: {baseline_slope:+.4f} (systematic drift)")
    print(f"• Best stabilizing policy: {best_policy['Policy']} (slope {best_policy['Energy Slope']})")
    print(f"• Intervention cost: {best_policy['Intervention %']} of iterations required correction")
    print(f"• Stability gain: {abs(baseline_slope) - abs(float(best_policy['Energy Slope'])):.4f} reduction in drift rate")
    
    # Statistical significance note
    print("\n⚠️  STATISTICAL NOTE")
    print("   For publication: Run with N≥200 problems and compute")
    print("   confidence intervals on energy slopes (bootstrap resampling).")
    print("   Current pilot (N=20) shows directional evidence only.")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Scale to N=200 problems for statistical significance")
    print("2. Add accuracy scoring (GSM8K answer extraction + verification)")
    print("3. Compute energy-accuracy correlation per policy")
    print("4. Generate Figure 1: Energy trajectories (baseline vs. harnessed)")
    print("="*70)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("HALLUCINATION ENERGY POLICY CONTROL EXPERIMENT")
    print("="*70)
    print(f"Model: {Config.OLLAMA_MODEL} via Ollama")
    print(f"Task: Recursive refinement of GSM8K reasoning")
    print(f"Policies: 6 intervention strategies (baseline → aggressive containment)")
    print(f"Energy signal: Geometric grounding + stability (invariant across policies)")
    print("="*70)
    
    # Run experiment
    results_df = run_experiment()
    
    # Generate report
    generate_report(results_df)
    
    print("\n✅ Experiment complete. Ready for analysis and paper writing.") Is Chinese this guy