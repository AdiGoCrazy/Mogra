"""Pydantic schemas for intent parsing and retrieval parameter standardization."""

from pydantic import BaseModel, Field
from typing import Optional, Any

class HardFilters(BaseModel):
    """Hard Cypher constraints extracted from user prompt."""
    mpaa_ratings: Optional[list[str]] = Field(None, description="Allowed MPAA ratings (e.g. ['G', 'PG'])")
    min_imdb_rating: Optional[float] = Field(None, description="Minimum IMDb score (e.g. 8.0)")
    min_rt_critics_score: Optional[int] = Field(None, description="Minimum Rotten Tomatoes score")
    max_vote_count: Optional[int] = Field(None, description="For hidden gems filtering")
    subgenres: Optional[list[str]] = Field(None, description="List of subgenres to match")
    primary_genre: Optional[str] = Field(None, description="Primary top-level genre")
    excluded_genres: Optional[list[str]] = Field(default_factory=list, description="Explicitly prohibited genres e.g. ['Horror']")
    gore_level: Optional[str] = Field(None, description="Set ONLY if user explicitly requests or limits gore (NONE, MILD, HIGH_GRAPHIC_GORE). Keep null/None unless user explicitly mentions gore!")
    romance_type: Optional[str] = Field(None, description="Required romance type")
    has_jump_scares: Optional[bool] = Field(None, description="False if jump scares prohibited")
    has_romance: Optional[bool] = Field(None, description="False if romance prohibited")

class QueryVectorPrompts(BaseModel):
    """Target prose prompts used to generate query embeddings across the 16 named vectors."""
    visual_aesthetic_prompt: str = Field("", description="Target visual vibe string")
    character_psychology_prompt: str = Field("", description="Target character interiority string")
    soundscape_prompt: str = Field("", description="Target audio/music vibe string")
    emotional_aftertaste_prompt: str = Field("", description="Target lingering mood string")
    philosophical_depth_prompt: str = Field("", description="Target philosophical theme string")
    tonal_arc_prompt: str = Field("", description="Target narrative pacing string")
    dialogue_and_wit_prompt: str = Field("", description="Target dialogue and wit vibe string")
    pacing_and_kinetic_rhythm_prompt: str = Field("", description="Target kinetic rhythm string")
    spatial_atmosphere_prompt: str = Field("", description="Target architecture and weather vibe string")
    cultural_historical_texture_prompt: str = Field("", description="Target period texture string")
    climactic_catharsis_prompt: str = Field("", description="Target climax payoff string")
    antagonist_threat_dynamics_prompt: str = Field("", description="Target villain threat string")
    thematic_subtext_allegory_prompt: str = Field("", description="Target allegorical subtext string")
    humor_and_irony_tone_prompt: str = Field("", description="Target satirical humor string")
    intimacy_and_chemistry_prompt: str = Field("", description="Target intimacy tension string")
    dread_suspense_escalation_prompt: str = Field("", description="Target dread escalation string")

class VectorWeights(BaseModel):
    """Normalized weights across the 16 vector fields summing to 1.0."""
    visual_aesthetic: float = Field(0.0625, description="Weight for visual vector")
    character_psychology: float = Field(0.0625, description="Weight for character psychology vector")
    emotional_aftertaste: float = Field(0.0625, description="Weight for emotional aftertaste vector")
    soundscape: float = Field(0.0625, description="Weight for soundscape vector")
    philosophical_depth: float = Field(0.0625, description="Weight for philosophical depth vector")
    tonal_arc: float = Field(0.0625, description="Weight for tonal arc vector")
    dialogue_and_wit: float = Field(0.0625, description="Weight for dialogue and wit vector")
    pacing_and_kinetic_rhythm: float = Field(0.0625, description="Weight for kinetic rhythm vector")
    spatial_atmosphere: float = Field(0.0625, description="Weight for spatial atmosphere vector")
    cultural_historical_texture: float = Field(0.0625, description="Weight for historical texture vector")
    climactic_catharsis: float = Field(0.0625, description="Weight for climactic catharsis vector")
    antagonist_threat_dynamics: float = Field(0.0625, description="Weight for antagonist threat vector")
    thematic_subtext_allegory: float = Field(0.0625, description="Weight for thematic subtext vector")
    humor_and_irony_tone: float = Field(0.0625, description="Weight for humor tone vector")
    intimacy_and_chemistry: float = Field(0.0625, description="Weight for intimacy vector")
    dread_suspense_escalation: float = Field(0.0625, description="Weight for dread escalation vector")

class QueryIntentPayload(BaseModel):
    """Full intent parsing output from Gemini Intent Engine."""
    raw_query: str = ""
    normalized_summary: str = ""
    seed_anchors: list[str] = Field(default_factory=list, description="Titles of seed movies mentioned in query")
    negative_seed_anchors: list[str] = Field(default_factory=list, description="Reference movies user dislikes or wants to avoid")
    dialogue_state: str = Field("NEW_RECOMMENDATION_QUERY", description="NEW_RECOMMENDATION_QUERY, CONVERSATIONAL_CHATTER, QUERY_REFINEMENT")
    is_contrasting_mix: bool = Field(False, description="True if query mixes contrasting aesthetic styles")
    hard_filters: HardFilters = Field(default_factory=HardFilters)
    negative_exclusions: list[str] = Field(default_factory=list, description="Explicitly prohibited elements or titles")
    vector_prompts: QueryVectorPrompts = Field(default_factory=QueryVectorPrompts)
    vector_weights: VectorWeights = Field(default_factory=VectorWeights)
    weight_profile_name: str = Field("BALANCED", description="VISUAL_HEAVY, CHARACTER_HEAVY, ATMOSPHERIC, BALANCED")
