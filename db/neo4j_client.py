"""Neo4j client wrapper for managing Graph DB connection sessions and Cypher execution with unified logging."""

import time
from typing import Any, Optional
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, Neo4jError
from config import config
from logger.unified_logger import get_logger, Subsystem

logger = get_logger(Subsystem.DB_NEO4J)

class Neo4jClient:
    """Manager class for Neo4j database connections and query execution."""

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> None:
        """Initialize Neo4j database driver connection.

        Args:
            uri: Neo4j Bolt connection URI.
            user: Username credential.
            password: Password credential.
        """
        self.uri = uri or config.NEO4J_URI
        self.user = user or config.NEO4J_USER
        self.password = password or config.NEO4J_PASSWORD
        self.driver: Optional[Driver] = None

    def connect(self) -> Optional[Driver]:
        """Establish connection to Neo4j database.

        Returns:
            Active Neo4j Driver instance or None if connection fails.
        """
        if not self.driver:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                logger.info(f"[DB_CONNECT] Connected to Neo4j Graph DB at {self.uri}")
            except Exception as e:
                logger.warning(f"[DB_CONNECT_FAILED] Could not connect to Neo4j at {self.uri}: {e}")
                return None
        return self.driver

    def close(self) -> None:
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("[DB_CLOSE] Neo4j database driver connection closed.")

    def execute_query(self, query: str, parameters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Execute a Cypher query against Neo4j and return records.

        Args:
            query: Cypher query string.
            parameters: Dictionary of query parameters.

        Returns:
            List of dictionaries representing record outputs.
        """
        if not self.driver:
            driver = self.connect()
            if not driver:
                logger.warning("[CYPHER_EXEC_SKIPPED] Neo4j database offline. Returning empty query result.")
                return []

        start_time = time.time()
        try:
            with self.driver.session() as session:  # type: ignore
                result = session.run(query, parameters or {})
                records = [record.data() for record in result]
                dt = time.time() - start_time
                logger.info(f"[CYPHER_EXEC] ({dt:.3f}s) Returned {len(records)} records | Query: '{query.strip()[:100]}...' | Params: {parameters}")
                return records
        except (ServiceUnavailable, Neo4jError, OSError) as e:
            logger.warning(f"[CYPHER_ERROR] Neo4j query execution failed: {e}")
            return []

neo4j_client = Neo4jClient()
