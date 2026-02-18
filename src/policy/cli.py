import argparse
from policy.experiments.runaway_report import RunawayReportHarness

def main():
    parser = argparse.ArgumentParser(description="Run Policy Governor experiment.")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    # Minimal demo AI
    def demo_ai(state):
        return {"quality": state.get("quality", 0.5)}

    def demo_energy(output, context):
        return abs(output["quality"] - 1.0)

    harness = RunawayReportHarness(
        ai_callable=demo_ai,
        energy_function=demo_energy,
        episodes=args.episodes,
    )

    harness.run()

if __name__ == "__main__":
    main()
