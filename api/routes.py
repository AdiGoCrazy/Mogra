"""FastAPI Route Handlers for Mogra Movie Recommender Agent REST API."""

import time
import logging
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
from engine.intent_parser import intent_parser
from engine.retrieval import hybrid_retriever
from engine.synthesizer import response_synthesizer
from db.feedback_store import FeedbackRecord, feedback_store
from db.neo4j_client import neo4j_client
from api.schemas import (
    FeedbackSubmissionRequest,
    FeedbackSummaryResponse,
    IntentParseRequest,
    MovieCandidateSchema,
    RecommendationRequest,
    RecommendationResponse,
    SystemStatsResponse,
    TelemetryMetricsSchema
)
from logger.unified_logger import get_logger, Subsystem

logger = get_logger(Subsystem.API_ROUTER)
router = APIRouter(prefix="/api/v1")

@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Movie Recommendations",
    description="Submit natural language prompt to perform GraphRAG multi-vector retrieval and synthesis."
)
def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """Execute end-to-end recommendation pipeline and record feedback dataset entry."""
    start_time = time.time()
    logger.info(f"[API_QUERY] Recommendation requested for prompt: '{request.prompt}'")

    try:
        # Step A: Intent Parsing
        t0 = time.time()
        intent = intent_parser.parse_query(request.prompt)
        dt_intent = time.time() - t0

        if not intent:
            logger.error(f"[QUERY_ERROR] Intent parser returned None for query: '{request.prompt}'")
            explanation = "Sorry, I could not find the movie requested."
            telemetry = TelemetryMetricsSchema(parse_ms=round(dt_intent * 1000.0, 2), retrieval_ms=0.0, total_ms=round((time.time() - start_time) * 1000.0, 2), candidate_count=0)
            rec = feedback_store.create_feedback_record(user_prompt=request.prompt, parsed_intent={}, recommended_movies=[], explanation=explanation, metrics={"parse_ms": telemetry.parse_ms, "retrieval_ms": 0.0, "total_ms": telemetry.total_ms, "candidate_count": 0})
            return RecommendationResponse(record_id=rec.record_id, user_prompt=request.prompt, parsed_intent={}, recommended_movies=[], synthesis_explanation=explanation, overall_rating=rec.overall_rating, telemetry_metrics=telemetry)

        # Step B: Retrieval
        t0 = time.time()
        raw_candidates = hybrid_retriever.retrieve_recommendations(
            intent,
            top_k=request.top_k,
            min_similarity_threshold=request.min_similarity_threshold
        )

        # Enrich candidates with synopsis & subgenres from Neo4j
        enriched_candidates: list[dict[str, Any]] = []
        for c in raw_candidates:
            tmdb_id = c["tmdb_id"]
            cypher = """
            MATCH (m:Movie {tmdb_id: $tmdb_id})
            OPTIONAL MATCH (m)-[:HAS_SUBGENRE]->(s:Subgenre)
            OPTIONAL MATCH (m)-[:BELONGS_TO_GENRE]->(g:Genre)
            RETURN m.title AS title, m.release_year AS release_year, m.imdb_rating AS imdb_rating,
                   m.mpaa_rating AS mpaa_rating, m.narrative_synopsis AS synopsis,
                   g.name AS primary_genre, collect(s.name) AS subgenres, m.tmdb_id AS tmdb_id
            """
            res = hybrid_retriever.neo4j.execute_query(cypher, {"tmdb_id": tmdb_id})
            if res:
                enriched_candidates.append(res[0])
            else:
                enriched_candidates.append(c)

        dt_retrieval = time.time() - t0

        if not enriched_candidates:
            logger.error(f"[QUERY_ERROR] Candidate retrieval found 0 candidates for query: '{request.prompt}'")
            explanation = "Sorry, I could not find the movie requested."
            telemetry = TelemetryMetricsSchema(parse_ms=round(dt_intent * 1000.0, 2), retrieval_ms=round(dt_retrieval * 1000.0, 2), total_ms=round((time.time() - start_time) * 1000.0, 2), candidate_count=0)
            rec = feedback_store.create_feedback_record(user_prompt=request.prompt, parsed_intent=intent.model_dump(), recommended_movies=[], explanation=explanation, metrics={"parse_ms": telemetry.parse_ms, "retrieval_ms": telemetry.retrieval_ms, "total_ms": telemetry.total_ms, "candidate_count": 0})
            return RecommendationResponse(record_id=rec.record_id, user_prompt=request.prompt, parsed_intent=intent.model_dump(), recommended_movies=[], synthesis_explanation=explanation, overall_rating=rec.overall_rating, telemetry_metrics=telemetry)

        # Step C: Synthesis
        t0 = time.time()
        explanation = response_synthesizer.synthesize_response(
            request.prompt,
            enriched_candidates,
            seed_anchors=intent.seed_anchors
        )
        dt_synthesis = time.time() - t0
        total_dt = time.time() - start_time

        telemetry = TelemetryMetricsSchema(
            parse_ms=round(dt_intent * 1000.0, 2),
            retrieval_ms=round(dt_retrieval * 1000.0, 2),
            total_ms=round(total_dt * 1000.0, 2),
            candidate_count=len(enriched_candidates)
        )

        # Save record to RLHF Feedback Store
        record = FeedbackRecord(
            user_prompt=request.prompt,
            parsed_intent=intent.model_dump(),
            recommended_movies=enriched_candidates,
            synthesis_explanation=explanation,
            overall_rating="UNRATED",
            telemetry_metrics=telemetry.model_dump()
        )
        record_id = feedback_store.save_record(record)

        candidates_schema = [
            MovieCandidateSchema(
                tmdb_id=m.get("tmdb_id", 0),
                title=m.get("title", "Unknown"),
                release_year=m.get("release_year"),
                imdb_rating=m.get("imdb_rating"),
                mpaa_rating=m.get("mpaa_rating", "NR"),
                primary_genre=m.get("primary_genre"),
                subgenres=m.get("subgenres", []),
                synopsis=m.get("synopsis", "")
            )
            for m in enriched_candidates
        ]

        return RecommendationResponse(
            record_id=record_id,
            user_prompt=request.prompt,
            parsed_intent=intent.model_dump(),
            recommended_movies=candidates_schema,
            synthesis_explanation=explanation,
            overall_rating="UNRATED",
            telemetry_metrics=telemetry
        )

    except Exception as e:
        logger.error(f"[API_ERROR] Failed processing recommendation query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation engine error: {str(e)}"
        )

@router.post(
    "/intent/parse",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Parse Prompt Intent",
    description="Parse natural language prompt into structured QueryIntentPayload JSON."
)
def parse_intent(request: IntentParseRequest) -> Dict[str, Any]:
    """Parse query intent directly."""
    try:
        intent = intent_parser.parse_query(request.prompt)
        return intent.model_dump()
    except Exception as e:
        logger.error(f"[API_ERROR] Failed parsing intent: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Intent parser error: {str(e)}"
        )

@router.post(
    "/feedback",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Submit RLHF Feedback",
    description="Submit 👍 Approve or 👎 Disapprove rating for a recommendation record."
)
def submit_feedback(request: FeedbackSubmissionRequest) -> Dict[str, Any]:
    """Record user approval rating."""
    if request.overall_rating not in ("APPROVED", "DISAPPROVED"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="overall_rating must be either 'APPROVED' or 'DISAPPROVED'."
        )

    success = feedback_store.update_record_rating(
        record_id=request.record_id,
        overall_rating=request.overall_rating,
        card_ratings=request.card_ratings,
        notes=request.developer_notes or ""
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Record ID '{request.record_id}' not found in dataset store."
        )

    return {
        "status": "SUCCESS",
        "record_id": request.record_id,
        "rating": request.overall_rating,
        "message": f"Recorded feedback rating '{request.overall_rating}' successfully."
    }

@router.get(
    "/feedback/summary",
    response_model=FeedbackSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get RLHF Feedback Stats",
    description="Retrieve aggregated RLHF dataset metrics and approval percentage."
)
def get_feedback_summary() -> FeedbackSummaryResponse:
    """Get feedback summary stats."""
    stats = feedback_store.get_summary_stats()
    return FeedbackSummaryResponse(
        total_records=stats["total_records"],
        approved_count=stats["approved_count"],
        disapproved_count=stats["disapproved_count"],
        unrated_count=stats["unrated_count"],
        approval_rate_pct=stats["approval_rate_pct"]
    )

@router.get(
    "/metadata/stats",
    response_model=SystemStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get System Statistics",
    description="Retrieve Neo4j Graph DB node counts and system status."
)
def get_system_stats() -> SystemStatsResponse:
    """Retrieve system health and database statistics."""
    try:
        movies_res = neo4j_client.execute_query("MATCH (m:Movie) RETURN count(m) AS count")
        genres_res = neo4j_client.execute_query("MATCH (g:Genre) RETURN count(g) AS count")
        subg_res = neo4j_client.execute_query("MATCH (s:Subgenre) RETURN count(s) AS count")
        sett_res = neo4j_client.execute_query("MATCH (st:Setting) RETURN count(st) AS count")

        m_count = movies_res[0]["count"] if movies_res else 0
        g_count = genres_res[0]["count"] if genres_res else 0
        s_count = subg_res[0]["count"] if subg_res else 0
        st_count = sett_res[0]["count"] if sett_res else 0

        feedback_stats = feedback_store.get_summary_stats()

        return SystemStatsResponse(
            status="HEALTHY",
            total_movies=m_count,
            total_genres=g_count,
            total_subgenres=s_count,
            total_settings=st_count,
            qdrant_status="CONNECTED",
            feedback_stats=feedback_stats
        )
    except Exception as e:
        logger.error(f"[API_ERROR] Failed fetching system stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed retrieving database statistics: {str(e)}"
        )
