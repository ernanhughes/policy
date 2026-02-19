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
