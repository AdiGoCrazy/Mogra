"""Active Learning & RLHF Dataset Analysis CLI Tool for Mogra Movie Recommender Agent."""

import os
import json
import argparse
from typing import Dict, List, Any
from db.feedback_store import FeedbackStore, FeedbackRecord

def analyze_dataset(dataset_path: str, report_output_path: str) -> bool:
    """Analyze recorded RLHF feedback dataset and generate active learning fine-tuning report."""
    store = FeedbackStore(filepath=dataset_path)
    records = store.load_all_records()
    stats = store.get_summary_stats()

    print("\n" + "=" * 90)
    print("                    RLHF FEEDBACK & ACTIVE LEARNING DATASET AUDIT")
    print("=" * 90)
    print(f"  Dataset File Path               : {store.filepath}")
    print(f"  Total Prompts Evaluated         : {stats['total_records']}")
    print(f"  Approved Prompts (👍)            : {stats['approved_count']}")
    print(f"  Disapproved Prompts (👎)         : {stats['disapproved_count']}")
    print(f"  Unrated Prompts                 : {stats['unrated_count']}")
    print(f"  Overall Approval Rate %         : {stats['approval_rate_pct']:.1f}%")
    print("=" * 90 + "\n")

    failures = store.get_failure_records()

    # Generate Markdown Report
    os.makedirs(os.path.dirname(os.path.abspath(report_output_path)), exist_ok=True)
    report_lines = [
        "# RLHF Feedback & Active Learning Failure Report 📊",
        "",
        f"- **Dataset Path**: `{store.filepath}`",
        f"- **Total Prompts Recorded**: `{stats['total_records']}`",
        f"- **Approved Prompts**: `{stats['approved_count']}` (👍)",
        f"- **Disapproved Prompts**: `{stats['disapproved_count']}` (👎)",
        f"- **Approval Rate**: `{stats['approval_rate_pct']:.1f}%`",
        "",
        "## Disapproved Query Analysis (Failure Pipeline)",
        ""
    ]

    if not failures:
        report_lines.append("✅ No disapproved prompts recorded. System is operating at 100% human approval.")
        print("✅ No failure records found in dataset.")
    else:
        print("DISAPPROVED PROMPTS AUDIT & FAILURE BREAKDOWN:")
        print(f"{'#':<3} | {'Prompt':<45} | {'Primary Genre':<15} | {'Card Ratings':<15}")
        print("-" * 90)

        for idx, rec in enumerate(failures, start=1):
            intent = rec.parsed_intent
            p_genre = intent.get("hard_filters", {}).get("primary_genre", "None") if isinstance(intent.get("hard_filters"), dict) else "None"
            card_ratings_str = ", ".join([f"{k}:{v[0]}" for k, v in rec.card_ratings.items()]) or "All 👎"

            print(f"{idx:<3} | {rec.user_prompt[:45]:<45} | {str(p_genre):<15} | {card_ratings_str:<15}")

            report_lines.append(f"### Failure #{idx}: \"{rec.user_prompt}\"")
            report_lines.append(f"- **Record ID**: `{rec.record_id}`")
            report_lines.append(f"- **Timestamp**: `{rec.timestamp}`")
            report_lines.append(f"- **Parsed Intent**: `{json.dumps(intent)}`")
            report_lines.append(f"- **Recommended Movies**: {[m.get('title') for m in rec.recommended_movies]}")
            report_lines.append(f"- **Card Level Ratings**: `{rec.card_ratings}`")
            if rec.developer_notes:
                report_lines.append(f"- **Developer Notes**: {rec.developer_notes}")
            report_lines.append("")

    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n📄 Active Learning Failure Report generated -> {report_output_path}\n")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze RLHF Feedback Dataset for Movie Recommender Agent.")
    parser.add_argument("--dataset", type=str, default="data/rlhf_feedback_dataset.jsonl", help="Path to JSONL dataset file.")
    parser.add_argument("--report", type=str, default="reports/rlhf_feedback_analysis.md", help="Output Markdown report path.")
    args = parser.parse_args()

    analyze_dataset(args.dataset, args.report)
