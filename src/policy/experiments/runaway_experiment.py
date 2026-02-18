# policy/experiments/runaway_experiment.py

from __future__ import annotations
import random
import numpy as np
from typing import Callable, Dict

from policy.policy_container import PolicyContainer


class RunawayDeclineExperiment:
    """
    Simulates uncontrolled self-modification.
    Demonstrates energy spiral vs policy containment.
    """

    def __init__(
        self,
        ai_callable: Callable,
        energy_function: Callable,
        policy: PolicyContainer | None = None,
        episodes: int = 500,
        noise_scale: float = 0.05,
    ):
        self.ai_callable = ai_callable
        self.energy_function = energy_function
        self.policy = policy
        self.episodes = episodes
        self.noise_scale = noise_scale

    def run(self) -> Dict:
        attempted_energies = []
        accepted_energies = []
        accepted = 0

        state = {"quality": 1.0}

        for _ in range(self.episodes):
            # Simulate self-modification drift
            state["quality"] += random.gauss(0, self.noise_scale)

            output = self.ai_callable(state)
            energy = float(self.energy_function(output, {}))
            attempted_energies.append(energy)

            if self.policy:
                _, decision = self.policy.execute(state)
                if decision.verdict == "ACCEPT":
                    accepted += 1
                    accepted_energies.append(float(decision.energy))
                else:
                    # rollback (undo a step)
                    state["quality"] -= random.gauss(0, self.noise_scale)
            else:
                accepted_energies.append(energy)

        # Stats should be computed on the energies you actually "kept"
        energies = accepted_energies

        if len(energies) == 0:
            mean_energy = float("nan")
            max_energy = float("nan")
        else:
            mean_energy = float(np.mean(energies))
            max_energy = float(np.max(energies))

        return {
            "mean_energy": mean_energy,
            "max_energy": max_energy,
            "accept_rate": (accepted / self.episodes) if self.policy else 1.0,
            "energies": energies,
            # extra diagnostics (useful + doesn’t break callers)
            "attempted_energies": attempted_energies,
            "accepted": accepted,
        }
