# Mogra Movie Recommender Agent — REST API Documentation 🎬🌐

The **Mogra Movie Recommender Agent REST API** provides a production FastAPI RESTful interface for embedding movie recommendation capabilities into third-party web applications, mobile frontends, and automated workflows.

---

## 🌐 Web Interfaces & OpenAPI Docs

When the API server is running (`python3 run_api_server.py --port 8000`), the following interactive web interfaces are served automatically by FastAPI:

- **Interactive Swagger UI (Try out API endpoints in browser):**
  👉 [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Interactive Documentation:**
  👉 [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI 3.0 JSON Specification:**
  👉 [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## 📡 Base URL & Server Launch

- **Base URL:** `http://localhost:8000/api/v1`
- **CORS Support:** Enabled for all origins (`["*"]`) allowing cross-origin requests from web browsers.

### Starting the REST API Server
```bash
PYTHONPATH=. python3 run_api_server.py --host 0.0.0.0 --port 8000
```

---

## 🚀 Endpoint Reference

### 1. Execute Movie Recommendation Pipeline
- **Endpoint:** `POST /api/v1/recommendations`
- **Description:** Executes query intent parsing, GraphRAG Cypher pre-filtering, Qdrant multi-vector search, Reciprocal Rank Fusion (RRF), local LLM response synthesis, and logs telemetry.

#### Request Body (`RecommendationRequest`):
```json
{
  "prompt": "movies like Blade Runner or Alien with dark atmospheric vibe",
  "top_k": 3,
  "min_similarity_threshold": 0.35,
  "record_feedback_payload": true
}
```

#### Response Model (`RecommendationResponse`):
```json
{
  "prompt": "movies like Blade Runner or Alien with dark atmospheric vibe",
  "candidates": [
    {
      "tmdb_id": 202,
      "title": "2001: A Space Odyssey",
      "release_year": 1968,
      "imdb_rating": 8.3,
      "primary_genre": "Science Fiction"
    },
    {
      "tmdb_id": 125,
      "title": "Alien",
      "release_year": 1979,
      "imdb_rating": 8.5,
      "primary_genre": "Horror"
    }
  ],
  "synthesis": "Based on your query for dark atmospheric sci-fi, here are top recommendations...",
  "telemetry": {
    "parse_latency_ms": 145.2,
    "retrieval_latency_ms": 48.6,
    "total_latency_ms": 193.8,
    "candidate_count": 2
  },
  "feedback_record_id": "f83a2b1c-9012-4a8b-b671-123456789abc"
}
```

#### Example `curl` Command:
```bash
curl -X POST "http://localhost:8000/api/v1/recommendations" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "psychological thriller with high suspense",
       "top_k": 5
     }'
```

---

### 2. Parse Query Intent Payload
- **Endpoint:** `POST /api/v1/intent/parse`
- **Description:** Parses natural language input query into a structured `QueryIntentPayload` JSON object (extracting seed anchors, negative exclusions, genre filters, and vector weights).

#### Request Body (`IntentParseRequest`):
```json
{
  "prompt": "sci-fi space movie but not like Interstellar"
}
```

#### Response Model (`QueryIntentPayload`):
```json
{
  "raw_query": "sci-fi space movie but not like Interstellar",
  "normalized_summary": "sci-fi space movie but not like Interstellar",
  "seed_anchors": [],
  "negative_seed_anchors": ["Interstellar"],
  "dialogue_state": "NEW_RECOMMENDATION_QUERY",
  "is_contrasting_mix": false,
  "hard_filters": {
    "primary_genre": "Science Fiction",
    "subgenres": [],
    "excluded_genres": []
  },
  "negative_exclusions": ["Interstellar"],
  "weight_profile_name": "BALANCED"
}
```

#### Example `curl` Command:
```bash
curl -X POST "http://localhost:8000/api/v1/intent/parse" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "romantic comedy with witty dialogue"
     }'
```

---

### 3. Submit RLHF Human Rating Feedback
- **Endpoint:** `POST /api/v1/feedback`
- **Description:** Submits human feedback (`APPROVED` or `DISAPPROVED`) for a recommendation session payload, updating the RLHF dataset in `data/rlhf_feedback_dataset.jsonl`.

#### Request Body (`FeedbackSubmissionRequest`):
```json
{
  "record_id": "f83a2b1c-9012-4a8b-b671-123456789abc",
  "rating": "APPROVED",
  "notes": "Great recommendations matching space horror theme!"
}
```

#### Response Model:
```json
{
  "status": "success",
  "record_id": "f83a2b1c-9012-4a8b-b671-123456789abc",
  "rating": "APPROVED",
  "message": "Feedback submitted successfully."
}
```

#### Example `curl` Command:
```bash
curl -X POST "http://localhost:8000/api/v1/feedback" \
     -H "Content-Type: application/json" \
     -d '{
       "record_id": "f83a2b1c-9012-4a8b-b671-123456789abc",
       "rating": "APPROVED"
     }'
```

---

### 4. Get RLHF Feedback Summary Statistics
- **Endpoint:** `GET /api/v1/feedback/summary`
- **Description:** Returns aggregate approval rates and count metrics from the RLHF feedback store.

#### Response Model (`FeedbackSummaryResponse`):
```json
{
  "total_records": 42,
  "approved_count": 38,
  "disapproved_count": 4,
  "pending_count": 0,
  "approval_rate_percentage": 90.48
}
```

#### Example `curl` Command:
```bash
curl -X GET "http://localhost:8000/api/v1/feedback/summary"
```

---

### 5. Get System Graph & Vector Metadata Stats
- **Endpoint:** `GET /api/v1/metadata/stats`
- **Description:** Queries live connection status and node counts for Neo4j Graph DB and Qdrant Multi-Vector Store.

#### Response Model (`SystemStatsResponse`):
```json
{
  "neo4j_status": "connected",
  "movie_node_count": 80,
  "qdrant_status": "connected",
  "vector_point_count": 80,
  "known_movie_titles_count": 80
}
```

#### Example `curl` Command:
```bash
curl -X GET "http://localhost:8000/api/v1/metadata/stats"
```

---

### 6. Healthcheck Endpoint
- **Endpoint:** `GET /health`
- **Description:** Simple healthcheck verifying server status.

#### Response Model:
```json
{
  "status": "healthy",
  "service": "Mogra Movie Recommender API",
  "version": "1.0.0"
}
```

#### Example `curl` Command:
```bash
curl -X GET "http://localhost:8000/health"
```
