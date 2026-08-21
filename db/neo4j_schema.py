"""Neo4j schema initialization script establishing constraints, indexes, and taxonomy nodes."""

import logging
from db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Full Taxonomy: 15 Top Genres -> Granular Subgenres
TAXONOMY_MAP: dict[str, list[str]] = {
    "Action": [
        "Heroic Bloodshed", "Military Action", "Espionage", "Wuxia Action",
        "Disaster", "Adventure", "Superhero"
    ],
    "Animation": [
        "Traditional", "Stop Motion", "Claymation", "Cutout", "CGI (3D)",
        "Puppetry", "Live-Action Animation"
    ],
    "Comedy": [
        "Action-Adventure Comedy", "Action-Comedy", "Dark Comedy (Black Comedy)",
        "Romantic Comedy (Rom-Com)", "Buddy Comedy", "Road Comedy", "Slapstick",
        "Parody", "Spoof", "Satire", "Sitcom", "Sketch Comedy", "Mockumentary", "Prank"
    ],
    "Crime": [
        "Caper", "Heist", "Gangster", "Cop (Police)", "Detective", "Courtroom", "Procedural"
    ],
    "Drama": [
        "Melodrama", "Teen Drama", "Philosophical Drama", "Medical Drama",
        "Legal Drama", "Political Drama", "Anthropological Drama", "Religious Drama", "Docudrama"
    ],
    "Experimental": [
        "Surrealist", "Absurdist"
    ],
    "Fantasy": [
        "Contemporary Fantasy", "Urban Fantasy", "Dark Fantasy", "High Fantasy (Epic Fantasy)", "Myth"
    ],
    "Historical": [
        "Historical Event", "Biography (Biopic)", "Historical Epic",
        "Historical Fiction", "Period Piece", "Alternate History"
    ],
    "Horror": [
        "Ghost", "Monster", "Werewolf", "Vampire", "Occult",
        "Slasher", "Splatter", "Found Footage", "Zombie"
    ],
    "Romance": [
        "Romance Drama", "Romance Thriller", "Period Romance"
    ],
    "Science Fiction": [
        "Post-Apocalyptic", "Utopian", "Dystopian", "Cyberpunk",
        "Steampunk", "Tech Noir", "Space Opera", "Contemporary Sci-Fi", "Military Sci-Fi"
    ],
    "Thriller": [
        "Psychological Thriller", "Mystery", "Techno-Thriller", "Film Noir (Neo-Noir)"
    ],
    "Western": [
        "Neo-Western", "Epic Western", "Empire Western", "Marshal Western",
        "Outlaw Western", "Revenge Western", "Revisionist Western", "Spaghetti Western"
    ],
    "Musical": [
        "Classical Musical", "Diegetic Performance", "Stage Adaptation", "Jukebox Musical"
    ],
    "War": [
        "Combat Operations", "Psychological War", "POW/Escape", "Homefront Conflict"
    ],
}


def initialize_neo4j_schema(client: Neo4jClient) -> None:
    """Initialize constraints, indexes, and taxonomy nodes in Neo4j.

    Args:
        client: Neo4jClient connection instance.
    """
    logger.info("Initializing Neo4j Schema & Constraints...")

    schema_queries = [
        # Constraints
        "CREATE CONSTRAINT movie_tmdb_id_unique IF NOT EXISTS FOR (m:Movie) REQUIRE m.tmdb_id IS UNIQUE;",
        "CREATE CONSTRAINT genre_name_unique IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE;",
        "CREATE CONSTRAINT subgenre_name_unique IF NOT EXISTS FOR (s:Subgenre) REQUIRE s.name IS UNIQUE;",
        "CREATE CONSTRAINT person_name_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE;",
        
        # Indexes
        "CREATE INDEX movie_ratings_idx IF NOT EXISTS FOR (m:Movie) ON (m.imdb_rating, m.rt_critics_score, m.vote_count);",
        "CREATE INDEX movie_content_flags_idx IF NOT EXISTS FOR (m:Movie) ON (m.mpaa_rating, m.gore_level, m.romance_type);",
        "CREATE INDEX movie_release_year_idx IF NOT EXISTS FOR (m:Movie) ON (m.release_year);"
    ]

    for cypher in schema_queries:
        client.execute_query(cypher)

    logger.info("Populating Taxonomy Nodes & Parent-Child Relationships...")
    for genre_name, subgenres in TAXONOMY_MAP.items():
        # Create Top-Level Genre Node
        client.execute_query("MERGE (g:Genre {name: $name})", {"name": genre_name})
        
        for sub_name in subgenres:
            # Create Subgenre Node and Link to Parent Genre
            client.execute_query(
                """
                MERGE (s:Subgenre {name: $sub_name})
                WITH s
                MATCH (g:Genre {name: $genre_name})
                MERGE (s)-[:CHILD_OF_GENRE]->(g)
                """,
                {"sub_name": sub_name, "genre_name": genre_name}
            )

    logger.info("Neo4j Schema Initialization Complete.")
