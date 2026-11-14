"""Command-line demo:  python main.py "fintech" """
import argparse

from agents import pitch_text
from graph import run

LABELS = {
    "research": "Research Agent finished a report",
    "funding": "Funding Advisor finished a plan",
    "pitch": "Pitch Coach finished a draft",
    "review": "Peer review finished",
}


def main():
    parser = argparse.ArgumentParser(description="Multi-agent pitch deck planner")
    parser.add_argument("domain", nargs="?", help="Startup domain, e.g. fintech")
    args = parser.parse_args()
    domain = args.domain or input("Startup domain or industry: ").strip()

    print(f"\nPlanning a pitch deck for: {domain}\n")
    state, tracer, run_dir = run(
        domain, on_step=lambda node, _: print(f"  [{node}] {LABELS.get(node, node)}")
    )

    print("\n" + pitch_text(state["pitch"]))
    print(tracer.narrative())
    print(f"Saved pitch_deck.md, trace.jsonl, trace.md and graph.mmd in: {run_dir}")


if __name__ == "__main__":
    main()
