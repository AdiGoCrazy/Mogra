"""Pydantic data schemas for movie enrichment payload extraction."""

from pydantic import BaseModel, Field
from typing import Optional

class QualityAndContentRatings(BaseModel):
    """Factual rating metrics across movie review platforms."""
    mpaa_rating: str = Field(..., description="G, PG, PG-13, R, NC-17, or UNRATED")
    imdb_rating: Optional[float] = Field(None, description="IMDb score from 0.0 to 10.0")
    rt_critics_score: Optional[int] = Field(None, description="Rotten Tomatoes critics percentage (0 to 100)")
    rt_audience_score: Optional[int] = Field(None, description="Rotten Tomatoes audience percentage (0 to 100)")
    metacritic_score: Optional[int] = Field(None, description="Metacritic score (0 to 100)")
    vote_count: Optional[int] = Field(None, description="Total number of user ratings/votes across platforms")

class GenreTaxonomy(BaseModel):
    """Categorical taxonomy classification."""
    primary_genre: str = Field(..., description="One of the 15 Top-Level Genres")
    subgenres: list[str] = Field(..., description="Specific subgenres from the 68 taxonomy list")

class CharacterDetail(BaseModel):
    """Detailed character psychology and archetype annotation."""
    character_name: str = Field(..., description="Name of the character")
    actor_name: str = Field(..., description="Name of actor playing the character")
    role_type: str = Field(..., description="PRIMARY_PROTAGONIST, SECONDARY_PROTAGONIST, ANTAGONIST, ENSEMBLE_LEAD")
    gender_identity: str = Field(..., description="Female, Male, Non-Binary, Non-Human/AI")
    archetype: str = Field(..., description="e.g., Lone Wanderer, Reluctant Mentor, Tragic Lover, Final Girl")
    arc_type: str = Field(..., description="e.g., Redemption Through Sacrifice, Descent into Madness, Unapologetic Awakening")
    psychological_summary: str = Field(..., description="Prose detailing internal flaws, trauma, and character motivation.")

class InterCharacterRelationship(BaseModel):
    """Relational edge between movie characters."""
    character_a: str = Field(..., description="Name of Character A")
    character_b: str = Field(..., description="Name of Character B")
    relationship_type: str = Field(..., description="BROTHER_OF, MENTOR_TO, RIVAL_OF, ROMANTIC_PARTNER, FOUND_FAMILY")

class ContentAndRomanceAttributes(BaseModel):
    """Content warning levels, intensity, genre constraints, and romance classification."""
    romance_type: str = Field(..., description="LGBTQ_QUEER, HETEROSEXUAL, PLATONIC_ONLY, NONE")
    romance_prominence: str = Field(..., description="PRIMARY_PLOT, SECONDARY_SUBPLOT, ABSENT")
    ending_tone: str = Field(..., description="HAPPY_CATHARTIC, SAD_HEARTBREAKING, BITTERSWEET, AMBIGUOUS_OPEN, NIHILISTIC")
    gore_level: str = Field(..., description="NONE, MILD, HIGH_GRAPHIC_GORE")
    has_body_horror: bool = Field(..., description="True if film contains extreme biological transformations or practical body horror.")
    has_jump_scares: bool = Field(..., description="True if horror elements rely heavily on sudden jump scares.")
    era_category: str = Field(..., description="GOLDEN_AGE_CLASSIC, SILVER_SCREEN_1950S, RETRO_80S_90S, MODERN_CONTEMPORARY")
    setting_tags: list[str] = Field(default_factory=list, description="Spatial setting keywords e.g. ['space', 'spaceship', 'motel', 'isolated_station']")
    
    # Genre-Specific Level Markings & Constraints
    sci_fi_hardness: Optional[str] = Field(None, description="HARD_SCIENCE, SOFT_SCI_FI, SPECULATIVE")
    mind_bend_level: Optional[str] = Field(None, description="NONE, MODERATE, EXTREME_PARADOX")
    dystopian_severity: Optional[str] = Field(None, description="NONE, MILD, BLEAK_TOTALITARIAN")
    violence_level: Optional[str] = Field(None, description="NONE, MILD, HIGH_GRAPHIC_VIOLENCE")
    action_style: Optional[str] = Field(None, description="MARTIAL_ARTS, GUN_FU, CAR_CHASE, EXPLOSIVE_BLOCKBUSTER, GRITTY_REALISM")
    pacing_speed: Optional[str] = Field(None, description="SLOW_BURN, MODERATE, RELENTLESS_NONSTOP")
    psychological_dread_level: Optional[str] = Field(None, description="NONE, MILD, SUFFOCATING_PARANOIA")
    monster_type: Optional[str] = Field(None, description="SUPERNATURAL_DEMON, BODY_MUTATION, SLASHER_KILLER, ALIEN_CREATURE, PSYCHOLOGICAL_ILLUSION")
    heist_complexity: Optional[str] = Field(None, description="NONE, SIMPLE, MULTI_PHASE_PUZZLE")
    moral_ambiguity_level: Optional[str] = Field(None, description="BLACK_AND_WHITE, GREY, PURE_NIHILISM")
    grittiness_level: Optional[str] = Field(None, description="GLAMORIZED, MODERATE, UNFLINCHING_RAW")
    humor_style: Optional[str] = Field(None, description="SLAPSTICK, DARK_SATIRE, DRY_DEADPAN, CRUDE_RAUNCHY, PARODY, WITTY_DIALOGUE")
    absurdity_level: Optional[str] = Field(None, description="GROUNDED, MODERATE, SURREAL_ABSURD")
    relationship_dynamic: Optional[str] = Field(None, description="ENEMIES_TO_LOVERS, FRIENDS_TO_LOVERS, SECOND_CHANCE, UNREQUITED_LONGING, FORBIDDEN_LOVE")
    emotional_intensity: Optional[str] = Field(None, description="LIGHT, MODERATE, DEVASTATING_CATHARSIS")
    narrative_structure: Optional[str] = Field(None, description="LINEAR, NON_LINEAR, REVERSE_CHRONOLOGICAL, SPLIT_PERSPECTIVE")
    magic_system_style: Optional[str] = Field(None, description="NONE, SOFT_MYSTICAL, HARD_RULE_BASED")
    world_building_scale: Optional[str] = Field(None, description="INTIMATE_LOCAL, REGIONAL, EPIC_WORLD")
    war_realism: Optional[str] = Field(None, description="HEROIC_PROPAGANDA, DRAMATIZED, UNFLINCHING_BRUTAL")
    anti_war_stance_level: Optional[str] = Field(None, description="NEUTRAL, STRONG_ANTI_WAR")
    western_subgenre: Optional[str] = Field(None, description="CLASSIC_TRADITIONAL, SPAGHETTI_WESTERN, REVISIONIST")
    historical_accuracy_level: Optional[str] = Field(None, description="LOOSELY_INSPIRED, HISTORICAL_FICTION, STRICT_FACTUAL")
    period_era: Optional[str] = Field(None, description="ANCIENT, MEDIEVAL, REGENCY_GEORGIAN, WWI_WWII, RETRO_50S_70S")
    animation_style: Optional[str] = Field(None, description="2D_HAND_DRAWN, 3D_CGI, STOP_MOTION, EXPERIMENTAL_HYBRID")
    target_demographic: Optional[str] = Field(None, description="KIDS_FAMILY, TEEN_ADULT, MATURE_ONLY")
    twist_count: Optional[str] = Field(None, description="NONE, SINGLE_SHOCK_TWIST, MULTI_LAYERED_DECEPTION")
    detective_type: Optional[str] = Field(None, description="AMATEUR, HARD_BOILED_PI, CORONER_FORENSIC, PSYCHOLOGICAL_INVESTIGATOR")
    musical_integration: Optional[str] = Field(None, description="DIEGETIC_PERFORMANCE, NON_DIEGETIC_BROADWAY")
    experimental_abstraction: Optional[str] = Field(None, description="GROUNDED, SURREAL, ABSTRACT_AVANT_GARDE")

class NegativeFlags(BaseModel):
    """Strict boolean flags used for negative constraint filtering."""
    has_romance: bool = Field(..., description="True if romantic relationships exist in any notable capacity.")
    has_jump_scares: bool = Field(..., description="True if horror elements rely on sudden jump scares.")
    is_bleak_ending: bool = Field(..., description="True if narrative finishes without hope, redemption, or catharsis.")
    has_female_lead: bool = Field(..., description="True if primary protagonist is female.")

class VectorTextPayloads(BaseModel):
    """16 rich prose text payloads embedded for multi-vector search."""
    visual_aesthetic_description: str = Field(..., description="Prose detailing cinematography, lighting, aspect ratio, FX texture.")
    character_psychology_description: str = Field(..., description="Prose detailing collective character interiority, emotional vulnerability.")
    emotional_aftertaste_description: str = Field(..., description="Prose describing viewer state during credits and lingering mood.")
    philosophical_depth_description: str = Field(..., description="Prose analyzing worldviews, conflict engines, social commentary.")
    soundscape_description: str = Field(..., description="Prose describing musical score, ambient audio profile, diegetic silence.")
    tonal_arc_description: str = Field(..., description="Prose detailing the emotional progression from act one to resolution.")
    
    # The 10 New Dense Vector Spaces
    dialogue_and_wit_description: str = Field("", description="Prose detailing script dialogue rhythm, banter, monologues, slang.")
    pacing_and_kinetic_rhythm_description: str = Field("", description="Prose detailing editing cuts, momentum, real-time urgency.")
    spatial_atmosphere_description: str = Field("", description="Prose detailing architecture, weather, environmental texture.")
    cultural_historical_texture_description: str = Field("", description="Prose detailing period costume realism, socio-political atmosphere.")
    climactic_catharsis_description: str = Field("", description="Prose detailing final act payoff, twist revelation shock.")
    antagonist_threat_dynamics_description: str = Field("", description="Prose detailing villain complexity, threat omnipresence.")
    thematic_subtext_allegory_description: str = Field("", description="Prose detailing underlying social commentary, symbolic motifs.")
    humor_and_irony_tone_description: str = Field("", description="Prose detailing satirical absurdity, black comedy, dry deadpan wit.")
    intimacy_and_chemistry_description: str = Field("", description="Prose detailing romantic warmth, unspoken eye tension, vulnerability.")
    dread_suspense_escalation_description: str = Field("", description="Prose detailing creeping claustrophobia, paranoia escalation.")

    narrative_synopsis: str = Field(..., description="Comprehensive plot summary capturing major narrative beats.")

class MovieEnrichmentPayload(BaseModel):
    """Complete composite movie enrichment payload for ingestion."""
    tmdb_id: int
    title: str
    release_year: int
    synopsis: str = Field(..., description="Full 300-500 word plot synopsis capturing major character arcs, setting, and plot twists")
    directors: list[str]
    cast: list[str]
    sequel_of_tmdb_id: Optional[int] = Field(None, description="TMDB ID of parent movie if this film is a sequel")
    taxonomy: GenreTaxonomy
    ratings: QualityAndContentRatings
    characters: list[CharacterDetail]
    inter_character_relationships: list[InterCharacterRelationship]
    content_and_romance: ContentAndRomanceAttributes
    negative_flags: NegativeFlags
    vector_payloads: VectorTextPayloads
