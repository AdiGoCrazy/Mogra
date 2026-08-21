"""Pydantic v2 data transfer schemas for Mogra Movie Recommender Agent REST API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class RecommendationRequest(BaseModel):
    """Payload for submitting natural language movie recommendation queries."""
    prompt: str = Field(..., min_length=2, description="Natural language recommendation query string.")
    top_k: int = Field(default=1, ge=1, le=10, description="Maximum number of candidate recommendations to return.")
    min_similarity_threshold: float = Field(default=0.30, ge=0.0, le=1.0, description="Minimum acceptable multi-vector cosine similarity score.")

class MovieCandidateSchema(BaseModel):
    """Schema representing an individual candidate movie recommendation."""
    tmdb_id: Any = Field(..., description="Unique TMDB numeric or string identifier.")
    title: str = Field(..., description="Canonical movie title.")
    release_year: Optional[int] = Field(default=None, description="Movie release year.")
    imdb_rating: Optional[float] = Field(default=None, description="IMDb 10-point scale rating.")
    mpaa_rating: Optional[str] = Field(default="NR", description="MPAA rating certificate.")
    primary_genre: Optional[str] = Field(default=None, description="Primary genre classification.")
    subgenres: List[str] = Field(default_factory=list, description="Subgenre tags associated with movie.")
    synopsis: Optional[str] = Field(default="", description="Narrative synopsis.")

class TelemetryMetricsSchema(BaseModel):
    """Telemetry performance metrics for query processing pipeline."""
    parse_ms: float = Field(..., description="Intent parsing duration in milliseconds.")
    retrieval_ms: float = Field(..., description="GraphRAG + Vector retrieval duration in milliseconds.")
    total_ms: float = Field(..., description="Total pipeline latency in milliseconds.")
    candidate_count: int = Field(..., description="Number of candidate recommendations meeting criteria.")

class RecommendationResponse(BaseModel):
    """Comprehensive recommendation payload returned by the agent API."""
    record_id: str = Field(..., description="Unique RLHF feedback record ID (UUID).")
    user_prompt: str = Field(..., description="Original user prompt string.")
    parsed_intent: Dict[str, Any] = Field(..., description="Parsed intent payload dictionary.")
    recommended_movies: List[MovieCandidateSchema] = Field(..., description="Ranked list of movie recommendations.")
    synthesis_explanation: str = Field(..., description="Generated natural language explanation.")
    overall_rating: str = Field(default="UNRATED", description="Current human feedback rating status.")
    telemetry_metrics: TelemetryMetricsSchema = Field(..., description="Pipeline execution telemetry.")

class IntentParseRequest(BaseModel):
    """Request payload for standalone intent parsing."""
    prompt: str = Field(..., min_length=2, description="Natural language prompt string to parse.")

class FeedbackSubmissionRequest(BaseModel):
    """Payload for submitting human approval/disapproval ratings for a recommendation record."""
    record_id: str = Field(..., description="Unique record ID returned by recommendation endpoint.")
    overall_rating: str = Field(..., description="Feedback rating: 'APPROVED' or 'DISAPPROVED'.")
    card_ratings: Optional[Dict[str, str]] = Field(default=None, description="Optional mapping of tmdb_id -> 'APPROVED' / 'DISAPPROVED'.")
    developer_notes: Optional[str] = Field(default="", description="Optional developer evaluation notes.")

class FeedbackSummaryResponse(BaseModel):
    """Summary metrics of recorded RLHF feedback dataset."""
    total_records: int = Field(..., description="Total prompt queries evaluated.")
    approved_count: int = Field(..., description="Count of approved payloads (👍).")
    disapproved_count: int = Field(..., description="Count of disapproved payloads (👎).")
    unrated_count: int = Field(..., description="Count of unrated payloads.")
    approval_rate_pct: float = Field(..., description="Overall human approval percentage.")

class SystemStatsResponse(BaseModel):
    """System health, graph database node metrics, and vector store status."""
    status: str = Field(default="HEALTHY", description="System health status.")
    total_movies: int = Field(..., description="Total Movie nodes in Neo4j Graph DB.")
    total_genres: int = Field(..., description="Total Genre nodes in Neo4j Graph DB.")
    total_subgenres: int = Field(..., description="Total Subgenre nodes in Neo4j Graph DB.")
    total_settings: int = Field(..., description="Total Setting nodes in Neo4j Graph DB.")
    qdrant_status: str = Field(default="CONNECTED", description="Qdrant vector store connection status.")
    feedback_stats: Dict[str, Any] = Field(..., description="Aggregated RLHF dataset feedback statistics.")
