"""Run with: python examples/run_simulation.py"""

from pprint import pprint

from camvo.simulation import run_simulation


def main() -> None:
    report, router = run_simulation(item_count=600, seed=7)
    pprint(report.to_dict())
    router.save_checkpoint("checkpoints/simulation.json")


if __name__ == "__main__":
    main()

