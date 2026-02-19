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
