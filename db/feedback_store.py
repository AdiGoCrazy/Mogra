import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from logger.unified_logger import get_logger, Subsystem

logger = get_logger(Subsystem.DB_FEEDBACK)

DATASET_FILE_PATH = os.path.abspath("data/rlhf_feedback_dataset.jsonl")

@dataclass
class FeedbackRecord:
    """Structured data record capturing a single user prompt, engine outputs, and human feedback."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_prompt: str = ""
    parsed_intent: Dict[str, Any] = field(default_factory=dict)
    recommended_movies: List[Dict[str, Any]] = field(default_factory=list)
    synthesis_explanation: str = ""
    overall_rating: str = "UNRATED"  # "APPROVED", "DISAPPROVED", "UNRATED"
    card_ratings: Dict[str, str] = field(default_factory=dict)  # tmdb_id/title -> "APPROVED" / "DISAPPROVED"
    developer_notes: str = ""
    telemetry_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackRecord":
        return cls(**data)

class FeedbackStore:
    """Manager providing persistent CRUD operations for RLHF feedback datasets."""

    def __init__(self, filepath: Optional[str] = None) -> None:
        self.filepath = os.path.abspath(filepath or DATASET_FILE_PATH)
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                pass  # Touch file

    def save_record(self, record: FeedbackRecord) -> str:
        """Append a new FeedbackRecord to the JSONL dataset file."""
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            logger.info(f"[FEEDBACK_SAVE] Saved record {record.record_id[:8]} for prompt: '{record.user_prompt[:40]}'")
            return record.record_id
        except Exception as e:
            logger.error(f"[FEEDBACK_ERROR] Failed to save record {record.record_id}: {e}", exc_info=True)
            raise

    def update_record_rating(
        self,
        record_id: str,
        overall_rating: str,
        card_ratings: Optional[Dict[str, str]] = None,
        notes: str = ""
    ) -> bool:
        """Update rating and notes for an existing record in place."""
        records = self.load_all_records()
        updated = False
        for rec in records:
            if rec.record_id == record_id:
                rec.overall_rating = overall_rating
                if card_ratings is not None:
                    rec.card_ratings.update(card_ratings)
                if notes:
                    rec.developer_notes = notes
                updated = True
                break

        if updated:
            # Overwrite file with updated records
            with open(self.filepath, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            logger.info(f"[FEEDBACK_UPDATE] Updated record {record_id[:8]} -> Rating: {overall_rating}")
            return True
        return False

    def load_all_records(self) -> List[FeedbackRecord]:
        """Load all feedback records from the JSONL dataset file."""
        records: List[FeedbackRecord] = []
        if not os.path.exists(self.filepath):
            return records

        with open(self.filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(FeedbackRecord.from_dict(data))
                except Exception as e:
                    logger.warning(f"[FEEDBACK_LOAD_WARN] Line {line_num} malformed JSON: {e}")
        return records

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute aggregated dataset metrics (total prompts, approval %, disapproval %)."""
        records = self.load_all_records()
        total = len(records)
        approved = sum(1 for r in records if r.overall_rating == "APPROVED")
        disapproved = sum(1 for r in records if r.overall_rating == "DISAPPROVED")
        unrated = sum(1 for r in records if r.overall_rating == "UNRATED")
        rated_total = approved + disapproved
        approval_rate = (approved / rated_total * 100.0) if rated_total > 0 else 0.0

        return {
            "total_records": total,
            "approved_count": approved,
            "disapproved_count": disapproved,
            "unrated_count": unrated,
            "rated_total": rated_total,
            "approval_rate_pct": round(approval_rate, 1)
        }

    def get_failure_records(self) -> List[FeedbackRecord]:
        """Retrieve all records rated as DISAPPROVED for fine-tuning analysis."""
        return [r for r in self.load_all_records() if r.overall_rating == "DISAPPROVED"]

# Singleton instance
feedback_store = FeedbackStore()
