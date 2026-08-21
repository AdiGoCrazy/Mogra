"""Neo4j graph ingestion module converting MovieEnrichmentPayload to graph nodes and relationships."""

import logging
from typing import Optional
from db.neo4j_client import Neo4jClient, neo4j_client
from schemas.enrichment import MovieEnrichmentPayload

logger = logging.getLogger(__name__)

def ingest_movie_to_neo4j(payload: MovieEnrichmentPayload, client: Optional[Neo4jClient] = None) -> None:
    """Ingest a validated MovieEnrichmentPayload into Neo4j Graph DB using MERGE statements.

    Args:
        payload: Validated MovieEnrichmentPayload instance.
        client: Neo4j database client wrapper.
    """
    client = client or neo4j_client

    cypher = """
    // 1. Merge Movie Node
    MERGE (m:Movie {tmdb_id: $tmdb_id})
    SET m.title = $title,
        m.release_year = $release_year,
        m.synopsis = $synopsis,
        m.mpaa_rating = $mpaa_rating,
        m.imdb_rating = $imdb_rating,
        m.rt_critics_score = $rt_critics_score,
        m.vote_count = $vote_count,
        m.gore_level = $gore_level,
        m.has_body_horror = $has_body_horror,
        m.has_jump_scares = $has_jump_scares,
        m.romance_type = $romance_type,
        m.ending_tone = $ending_tone,
        m.has_romance = $has_romance,
        m.sci_fi_hardness = $sci_fi_hardness,
        m.mind_bend_level = $mind_bend_level,
        m.dystopian_severity = $dystopian_severity,
        m.violence_level = $violence_level,
        m.action_style = $action_style,
        m.pacing_speed = $pacing_speed,
        m.psychological_dread_level = $psychological_dread_level,
        m.monster_type = $monster_type,
        m.heist_complexity = $heist_complexity,
        m.moral_ambiguity_level = $moral_ambiguity_level,
        m.grittiness_level = $grittiness_level,
        m.humor_style = $humor_style,
        m.absurdity_level = $absurdity_level,
        m.relationship_dynamic = $relationship_dynamic,
        m.emotional_intensity = $emotional_intensity,
        m.narrative_structure = $narrative_structure,
        m.magic_system_style = $magic_system_style,
        m.world_building_scale = $world_building_scale,
        m.war_realism = $war_realism,
        m.anti_war_stance_level = $anti_war_stance_level,
        m.western_subgenre = $western_subgenre,
        m.historical_accuracy_level = $historical_accuracy_level,
        m.period_era = $period_era,
        m.animation_style = $animation_style,
        m.target_demographic = $target_demographic,
        m.twist_count = $twist_count,
        m.detective_type = $detective_type,
        m.musical_integration = $musical_integration,
        m.experimental_abstraction = $experimental_abstraction

    // 2. Connect Top Genre
    MERGE (g:Genre {name: $primary_genre})
    MERGE (m)-[:BELONGS_TO_GENRE]->(g)

    // 3. Connect Subgenres
    FOREACH (sub IN $subgenres |
      MERGE (s:Subgenre {name: sub})
      MERGE (m)-[:HAS_SUBGENRE]->(s)
    )

    // 4. Connect Directors
    FOREACH (dir IN $directors |
      MERGE (p:Person {name: dir})
      MERGE (p)-[:DIRECTED]->(m)
    )

    // 5. Connect Setting Nodes
    FOREACH (st IN $setting_tags |
      MERGE (sett:Setting {name: st})
      MERGE (m)-[:SET_IN]->(sett)
    )

    // 6. Connect Sequel Relationship if present
    WITH m
    WHERE $sequel_of_tmdb_id IS NOT NULL
    MATCH (parent:Movie {tmdb_id: $sequel_of_tmdb_id})
    MERGE (m)-[:IS_SEQUEL_TO]->(parent)
    """

    params = {
        "tmdb_id": payload.tmdb_id,
        "title": payload.title,
        "release_year": payload.release_year,
        "synopsis": getattr(payload, "synopsis", getattr(payload.vector_payloads, "narrative_synopsis", payload.title)),
        "mpaa_rating": payload.ratings.mpaa_rating,
        "imdb_rating": payload.ratings.imdb_rating,
        "rt_critics_score": payload.ratings.rt_critics_score,
        "vote_count": payload.ratings.vote_count,
        "gore_level": payload.content_and_romance.gore_level,
        "has_body_horror": payload.content_and_romance.has_body_horror,
        "has_jump_scares": payload.content_and_romance.has_jump_scares,
        "romance_type": payload.content_and_romance.romance_type,
        "ending_tone": payload.content_and_romance.ending_tone,
        "has_romance": payload.negative_flags.has_romance,
        "primary_genre": payload.taxonomy.primary_genre,
        "subgenres": payload.taxonomy.subgenres,
        "directors": payload.directors,
        "setting_tags": getattr(payload.content_and_romance, "setting_tags", []),
        "sequel_of_tmdb_id": payload.sequel_of_tmdb_id,
        "sci_fi_hardness": getattr(payload.content_and_romance, "sci_fi_hardness", None),
        "mind_bend_level": getattr(payload.content_and_romance, "mind_bend_level", None),
        "dystopian_severity": getattr(payload.content_and_romance, "dystopian_severity", None),
        "violence_level": getattr(payload.content_and_romance, "violence_level", None),
        "action_style": getattr(payload.content_and_romance, "action_style", None),
        "pacing_speed": getattr(payload.content_and_romance, "pacing_speed", None),
        "psychological_dread_level": getattr(payload.content_and_romance, "psychological_dread_level", None),
        "monster_type": getattr(payload.content_and_romance, "monster_type", None),
        "heist_complexity": getattr(payload.content_and_romance, "heist_complexity", None),
        "moral_ambiguity_level": getattr(payload.content_and_romance, "moral_ambiguity_level", None),
        "grittiness_level": getattr(payload.content_and_romance, "grittiness_level", None),
        "humor_style": getattr(payload.content_and_romance, "humor_style", None),
        "absurdity_level": getattr(payload.content_and_romance, "absurdity_level", None),
        "relationship_dynamic": getattr(payload.content_and_romance, "relationship_dynamic", None),
        "emotional_intensity": getattr(payload.content_and_romance, "emotional_intensity", None),
        "narrative_structure": getattr(payload.content_and_romance, "narrative_structure", None),
        "magic_system_style": getattr(payload.content_and_romance, "magic_system_style", None),
        "world_building_scale": getattr(payload.content_and_romance, "world_building_scale", None),
        "war_realism": getattr(payload.content_and_romance, "war_realism", None),
        "anti_war_stance_level": getattr(payload.content_and_romance, "anti_war_stance_level", None),
        "western_subgenre": getattr(payload.content_and_romance, "western_subgenre", None),
        "historical_accuracy_level": getattr(payload.content_and_romance, "historical_accuracy_level", None),
        "period_era": getattr(payload.content_and_romance, "period_era", None),
        "animation_style": getattr(payload.content_and_romance, "animation_style", None),
        "target_demographic": getattr(payload.content_and_romance, "target_demographic", None),
        "twist_count": getattr(payload.content_and_romance, "twist_count", None),
        "detective_type": getattr(payload.content_and_romance, "detective_type", None),
        "musical_integration": getattr(payload.content_and_romance, "musical_integration", None),
        "experimental_abstraction": getattr(payload.content_and_romance, "experimental_abstraction", None),
    }

    client.execute_query(cypher, params)
    logger.info(f"Successfully ingested movie '{payload.title}' (ID: {payload.tmdb_id}) into Neo4j.")
