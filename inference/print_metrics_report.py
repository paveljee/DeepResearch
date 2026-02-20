import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Print DeepResearch LLM metrics summary.")
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to llm_metrics_summary.json",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    totals = data.get("totals", {})
    print("=== Global Totals ===")
    print(json.dumps(totals, indent=2, ensure_ascii=False))

    print("\n=== By Model ===")
    print(json.dumps(data.get("by_model", {}), indent=2, ensure_ascii=False))

    print("\n=== Tool Invocations (Global) ===")
    print(json.dumps(data.get("tool_invocations_total", {}), indent=2, ensure_ascii=False))

    print("\n=== By Run ===")
    by_run = data.get("by_run", {})
    for run_id, run_data in by_run.items():
        print(f"\n-- {run_id} --")
        print(json.dumps(run_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

