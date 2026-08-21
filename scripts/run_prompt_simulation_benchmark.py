#!/usr/bin/env python3
"""Automated Prompt Simulation Benchmark Runner for MograRecommenderAgent.

Evaluates 36 benchmark prompts across 6 core categories against ground-truth seed targets.
Calculates Top-1, Top-3, Top-5, Top-10 retrieval accuracy, exclusion constraints compliance,
genre precision %, and end-to-end latency (ms).
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from db.neo4j_client import neo4j_client
from db.qdrant_client import qdrant_wrapper
from engine.cache import intent_cache
from engine.intent_parser import intent_parser
from engine.retrieval import hybrid_retriever


@dataclass
class BenchmarkPromptCase:
    """Benchmark test prompt case definition.
    
    Attributes:
        id: Unique identifier (e.g. Q01).
        category: Benchmark prompt category.
        prompt: Simulated user query prompt string.
        expected_targets: List of expected ground-truth target movie titles.
        expected_genres: List of expected primary or relevant genre names.
        required_exclusions: List of movie titles that MUST NOT appear in recommendations.
    """
    id: str
    category: str
    prompt: str
    expected_targets: List[str]
    expected_genres: List[str]
    required_exclusions: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Detailed evaluation result for a single benchmark prompt case.
    
    Attributes:
        case_id: Benchmark prompt identifier.
        category: Benchmark prompt category.
        prompt: Simulated user prompt text.
        expected_targets: Expected ground-truth targets.
        retrieved_titles: List of top retrieved candidate titles.
        retrieved_genres: List of primary genres of retrieved candidates.
        top_1_hit: Whether target movie was found at rank 1.
        top_3_hit: Whether target movie was found within top 3.
        top_5_hit: Whether target movie was found within top 5.
        top_10_hit: Whether target movie was found within top 10.
        rank_found: 1-indexed rank of first target match, or None if not found in top 10.
        genre_precision_pct: Percentage of top 5 retrieved candidates matching expected genres.
        parse_latency_ms: Latency of query intent parsing in milliseconds.
        retrieval_latency_ms: Latency of hybrid recommendation retrieval in milliseconds.
        total_latency_ms: Total latency in milliseconds.
        exclusions_passed: Whether all required exclusions were satisfied.
        status: Overall status ("PASS" if top_5_hit and exclusions_passed else "FAIL").
    """
    case_id: str
    category: str
    prompt: str
    expected_targets: List[str]
    retrieved_titles: List[str]
    retrieved_genres: List[str]
    top_1_hit: bool
    top_3_hit: bool
    top_5_hit: bool
    top_10_hit: bool
    rank_found: Optional[int]
    genre_precision_pct: float
    parse_latency_ms: float
    retrieval_latency_ms: float
    total_latency_ms: float
    exclusions_passed: bool
    status: str


BENCHMARK_SUITE: List[BenchmarkPromptCase] = [
    # Category 1: Simple Top-Level Genre Requests
    BenchmarkPromptCase(
        id="Q01",
        category="Top-Level Genre",
        prompt="Can you recommend a classic sci-fi movie about space exploration and a mind-bending cosmic journey?",
        expected_targets=["2001: A Space Odyssey"],
        expected_genres=["Science Fiction"],
    ),
    BenchmarkPromptCase(
        id="Q02",
        category="Top-Level Genre",
        prompt="I am in the mood for a pure romantic drama set in Europe about two strangers connecting.",
        expected_targets=["Before Sunrise"],
        expected_genres=["Romance"],
    ),
    BenchmarkPromptCase(
        id="Q03",
        category="Top-Level Genre",
        prompt="Show me an intense psychological thriller centered on an FBI investigation of a serial killer.",
        expected_targets=["The Silence of the Lambs"],
        expected_genres=["Thriller"],
    ),
    BenchmarkPromptCase(
        id="Q04",
        category="Top-Level Genre",
        prompt="I want to watch an epic war film about World War I soldiers delivering a message across enemy lines.",
        expected_targets=["1917"],
        expected_genres=["War"],
    ),
    BenchmarkPromptCase(
        id="Q05",
        category="Top-Level Genre",
        prompt="Looking for a classic high fantasy epic movie about a quest to destroy a dark lord ring.",
        expected_targets=["The Lord of the Rings: The Fellowship of the Ring"],
        expected_genres=["Fantasy"],
    ),
    BenchmarkPromptCase(
        id="Q06",
        category="Top-Level Genre",
        prompt="Give me a high-octane action movie featuring non-stop vehicular combat across a desert wasteland.",
        expected_targets=["Mad Max: Fury Road"],
        expected_genres=["Action"],
    ),

    # Category 2: Subgenre Taxonomy & Hybrid Queries
    BenchmarkPromptCase(
        id="Q07",
        category="Subgenre Taxonomy",
        prompt="Looking for a Hong Kong Heroic Bloodshed action movie with fast-paced gun-fu shootouts and hitman honor.",
        expected_targets=["The Killer"],
        expected_genres=["Action"],
    ),
    BenchmarkPromptCase(
        id="Q08",
        category="Subgenre Taxonomy",
        prompt="Recommend a British dark zombie comedy (zom-com) set in a local pub during an undead apocalypse.",
        expected_targets=["Shaun of the Dead"],
        expected_genres=["Comedy"],
    ),
    BenchmarkPromptCase(
        id="Q09",
        category="Subgenre Taxonomy",
        prompt="I want a cyberpunk tech noir movie with rain-slicked neon streets, corporate dystopia, and synthetic replicants.",
        expected_targets=["Blade Runner"],
        expected_genres=["Science Fiction"],
    ),
    BenchmarkPromptCase(
        id="Q10",
        category="Subgenre Taxonomy",
        prompt="Show me a revisionist western about an aging retired outlaw taking on one last bounty in a rainy frontier town.",
        expected_targets=["Unforgiven"],
        expected_genres=["Western"],
    ),
    BenchmarkPromptCase(
        id="Q11",
        category="Subgenre Taxonomy",
        prompt="Looking for a hand-drawn 2D traditional animation film based on Japanese folklore, bathhouse spirits, and myths.",
        expected_targets=["Spirited Away"],
        expected_genres=["Animation"],
    ),
    BenchmarkPromptCase(
        id="Q12",
        category="Subgenre Taxonomy",
        prompt="Recommend a courtroom drama focusing entirely on 12 jurors deliberating a murder verdict in a single claustrophobic room.",
        expected_targets=["12 Angry Men"],
        expected_genres=["Drama"],
    ),

    # Category 3: Spatial Setting & Environmental Tags
    BenchmarkPromptCase(
        id="Q13",
        category="Spatial Setting",
        prompt="I want a horror film set in an isolated Antarctic research station during a freezing polar snowstorm.",
        expected_targets=["The Thing"],
        expected_genres=["Horror"],
    ),
    BenchmarkPromptCase(
        id="Q14",
        category="Spatial Setting",
        prompt="Looking for an epic space sci-fi movie set primarily on the desert planet Arrakis with spice harvesting and sandworms.",
        expected_targets=["Dune"],
        expected_genres=["Science Fiction"],
    ),
    BenchmarkPromptCase(
        id="Q15",
        category="Spatial Setting",
        prompt="Recommend a horror movie set in an isolated, snowbound hotel in the mountains with a winter snow maze.",
        expected_targets=["The Shining"],
        expected_genres=["Horror"],
    ),
    BenchmarkPromptCase(
        id="Q16",
        category="Spatial Setting",
        prompt="Give me a historical samurai movie set in a remote feudal Japanese farming village defended against bandits.",
        expected_targets=["Seven Samurai"],
        expected_genres=["Historical"],
    ),
    BenchmarkPromptCase(
        id="Q17",
        category="Spatial Setting",
        prompt="I want a modern detective caper comedy set on a tech billionaire luxury private island in Greece.",
        expected_targets=["Glass Onion: A Knives Out Mystery"],
        expected_genres=["Comedy"],
    ),
    BenchmarkPromptCase(
        id="Q18",
        category="Spatial Setting",
        prompt="Looking for a gritty crime heist drama set in urban Los Angeles featuring armed bank robberies and highway shootouts.",
        expected_targets=["Heat"],
        expected_genres=["Crime"],
    ),

    # Category 4: Content & Level Constraints
    BenchmarkPromptCase(
        id="Q19",
        category="Content Constraints",
        prompt="Recommend a wholesome PG-rated stop-motion animated movie with zero gore, no jump scares, and family-friendly heist humor.",
        expected_targets=["Fantastic Mr. Fox"],
        expected_genres=["Animation"],
    ),
    BenchmarkPromptCase(
        id="Q20",
        category="Content Constraints",
        prompt="I want an R-rated psychological horror film with high graphic gore, intense jump scares, and an occult demon plot, but NO romantic subplots.",
        expected_targets=["Hereditary"],
        expected_genres=["Horror"],
    ),
    BenchmarkPromptCase(
        id="Q21",
        category="Content Constraints",
        prompt="Looking for a dark satirical black comedy set in a political Pentagon War Room with a nihilistic nuclear ending and no romance.",
        expected_targets=["Dr. Strangelove"],
        expected_genres=["Comedy"],
    ),
    BenchmarkPromptCase(
        id="Q22",
        category="Content Constraints",
        prompt="Give me an R-rated sci-fi thriller about AI consciousness with mild gore, no jump scares, and platonic-only relationships.",
        expected_targets=["Ex Machina"],
        expected_genres=["Science Fiction"],
    ),
    BenchmarkPromptCase(
        id="Q23",
        category="Content Constraints",
        prompt="Recommend a romantic comedy with heterosexual romance, a happy cathartic ending, and zero gore set in New York City.",
        expected_targets=["When Harry Met Sally..."],
        expected_genres=["Romance"],
    ),
    BenchmarkPromptCase(
        id="Q24",
        category="Content Constraints",
        prompt="Show me a visceral body horror movie with high graphic gore about a scientist turning into a fly monster, but with NO jump scares.",
        expected_targets=["The Fly"],
        expected_genres=["Horror"],
    ),

    # Category 5: Negation Exclusions
    BenchmarkPromptCase(
        id="Q25",
        category="Negation Exclusions",
        prompt="I want a sci-fi military survival movie with alien creatures on an exoplanet, but NOT set on Earth and NOT involving time travel.",
        expected_targets=["Aliens"],
        expected_genres=["Science Fiction"],
        required_exclusions=["Earth", "Interstellar"],
    ),
    BenchmarkPromptCase(
        id="Q26",
        category="Negation Exclusions",
        prompt="Recommend a gritty crime movie about a heist gone wrong, but WITHOUT any romantic interest or slapstick comedy.",
        expected_targets=["Reservoir Dogs"],
        expected_genres=["Crime"],
        required_exclusions=["La La Land", "When Harry Met Sally..."],
    ),
    BenchmarkPromptCase(
        id="Q27",
        category="Negation Exclusions",
        prompt="Looking for an iconic 1978 slasher horror movie set in suburbia, but NOT involving supernatural dream demons or Freddy Krueger.",
        expected_targets=["Halloween"],
        expected_genres=["Horror"],
        required_exclusions=["A Nightmare on Elm Street"],
    ),
    BenchmarkPromptCase(
        id="Q28",
        category="Negation Exclusions",
        prompt="Give me a post-apocalyptic desert survival action film, but NOT featuring CGI superhero suits or interstellar space travel.",
        expected_targets=["Mad Max 2: The Road Warrior"],
        expected_genres=["Action"],
        required_exclusions=["The Dark Knight", "Spider-Man: Into the Spider-Verse", "Interstellar"],
    ),
    BenchmarkPromptCase(
        id="Q29",
        category="Negation Exclusions",
        prompt="Show me a surreal psychological romance about erasing memories, but NOT a horror movie with monsters or gory violence.",
        expected_targets=["Eternal Sunshine of the Spotless Mind"],
        expected_genres=["Romance"],
        required_exclusions=["The Thing", "The Fly", "Hereditary", "Psycho"],
    ),
    BenchmarkPromptCase(
        id="Q30",
        category="Negation Exclusions",
        prompt="Looking for an animated superhero movie set in New York City, but WITHOUT romantic drama or graphic R-rated gore.",
        expected_targets=["Spider-Man: Into the Spider-Verse"],
        expected_genres=["Animation"],
        required_exclusions=["The Fly", "Hereditary"],
    ),

    # Category 6: Multi-Anchor Cross-Genre Hybrid Fusion
    BenchmarkPromptCase(
        id="Q31",
        category="Multi-Anchor Fusion",
        prompt="A high-concept fusion of sci-fi action and tech-noir, like Terminator meets Blade Runner, featuring a shape-shifting cyborg assassin in Los Angeles.",
        expected_targets=["Terminator 2: Judgment Day"],
        expected_genres=["Action", "Science Fiction"],
        required_exclusions=["Blade Runner"],
    ),
    BenchmarkPromptCase(
        id="Q32",
        category="Multi-Anchor Fusion",
        prompt="A cross-genre hybrid of dark fairytale fantasy and historical war drama, blending 1940s Spanish fascists with labyrinthine monsters.",
        expected_targets=["Pan's Labyrinth"],
        expected_genres=["Fantasy", "War", "Historical"],
    ),
    BenchmarkPromptCase(
        id="Q33",
        category="Multi-Anchor Fusion",
        prompt="A fusion of Spaghetti Western bounty hunter showdowns and Civil War gold treasure hunting, blending ruthless outlaws and cemetery duels.",
        expected_targets=["The Good, the Bad and the Ugly"],
        expected_genres=["Western", "War"],
    ),
    BenchmarkPromptCase(
        id="Q34",
        category="Multi-Anchor Fusion",
        prompt="A cross-genre fusion of superhero action and gritty crime heist procedural, like Batman meets Heat in a sprawling city under terrorist threat.",
        expected_targets=["The Dark Knight"],
        expected_genres=["Action", "Crime"],
        required_exclusions=["Heat"],
    ),
    BenchmarkPromptCase(
        id="Q35",
        category="Multi-Anchor Fusion",
        prompt="A hybrid of sci-fi space exploration and claustrophobic creature horror, like 2001: A Space Odyssey meets violent monster survival in deep space.",
        expected_targets=["Alien"],
        expected_genres=["Science Fiction", "Horror"],
        required_exclusions=["2001: A Space Odyssey"],
    ),
    BenchmarkPromptCase(
        id="Q36",
        category="Multi-Anchor Fusion",
        prompt="A non-linear cross-genre fusion of mobster crime, hitman humor, and pop culture dialogue, like Goodfellas meets dark comedy vignettes in 90s LA.",
        expected_targets=["Pulp Fiction"],
        expected_genres=["Crime", "Comedy"],
    ),
]

def load_benchmark_suite(filepath: str = "data/benchmark_test_cases.json") -> List[BenchmarkPromptCase]:
    """Load benchmark prompt cases dynamically from JSON file or fallback to default suite."""
    import os, json
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                cases = []
                for idx, item in enumerate(data, start=1):
                    cases.append(
                        BenchmarkPromptCase(
                            id=f"Q{idx:02d}",
                            category=item.get("category", "General"),
                            prompt=item.get("prompt", ""),
                            expected_targets=item.get("expected", []),
                            expected_genres=item.get("expected_genres", item.get("expected", [])),
                            required_exclusions=item.get("required_exclusions", [])
                        )
                    )
                if cases:
                    return cases
        except Exception:
            pass
    return BENCHMARK_SUITE


def check_title_match(target: str, candidate_title: str) -> bool:
    """Check if target movie title matches candidate title exactly.
    
    Args:
        target: Ground truth target title.
        candidate_title: Retrieved candidate title.
        
    Returns:
        True if target matches candidate title exactly, False otherwise.
    """
    t_norm = target.strip().lower()
    c_norm = candidate_title.strip().lower()
    return t_norm == c_norm


def run_benchmark() -> bool:
    """Execute evaluation loop across all 36 benchmark prompt cases.
    
    Returns:
        True if Overall Top-5 Success Rate >= 85.0%, False otherwise.
    """
    print("=" * 110, flush=True)
    print("      MOGRA RECOMMENDER AGENT — PROMPT SIMULATION BENCHMARK RUNNER", flush=True)
    print("=" * 110, flush=True)

    # 1. Database Integrity Verification
    try:
        neo4j_res = neo4j_client.execute_query("MATCH (m:Movie) RETURN count(m) AS total")
        neo4j_count = neo4j_res[0]["total"] if neo4j_res else 0
        print(f"🟢 Neo4j Database Status  : Connected ({neo4j_count} Movies)", flush=True)
    except Exception as e:
        print(f"🔴 Neo4j Database Error   : {e}", flush=True)
        neo4j_count = 0

    try:
        q_client = qdrant_wrapper.get_client()
        col_info = q_client.get_collection("movies_multi_vector")
        qdrant_count = col_info.points_count
        print(f"🟢 Qdrant Vector Store Status: Connected ({qdrant_count} Points)", flush=True)
    except Exception as e:
        print(f"🔴 Qdrant Store Error     : {e}", flush=True)
        qdrant_count = 0

    print("-" * 110, flush=True)
    print("Clearing intent LRU cache for clean timing measurement...", flush=True)
    print("-" * 110, flush=True)
    intent_cache.clear()

    results: List[EvaluationResult] = []

    # Print Table Header
    hdr = f"{'ID':<4} | {'Category':<20} | {'Target Seed Movie':<32} | {'Rank':<5} | {'Top-5':<5} | {'Excl':<5} | {'Prec%':<6} | {'Parse':<6} | {'Retr':<6} | {'Total':<6} | {'Status':<6}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for case in load_benchmark_suite():
        t0 = time.perf_counter()

        # Step 1: Query Intent Parsing
        intent = intent_parser.parse_query(case.prompt)
        t_parse = (time.perf_counter() - t0) * 1000.0

        # Step 2: Hybrid Recommendation Retrieval
        t1 = time.perf_counter()
        candidates = hybrid_retriever.retrieve_recommendations(intent, top_k=10, min_similarity_threshold=0.30)
        t_retrieval = (time.perf_counter() - t1) * 1000.0
        t_total = (time.perf_counter() - t0) * 1000.0

        retrieved_titles = [c.get("title", "") for c in candidates]
        retrieved_genres = [c.get("primary_genre", "") for c in candidates if c.get("primary_genre")]

        # Step 3: Target Match Rank Determination
        rank_found: Optional[int] = None
        for idx, r_title in enumerate(retrieved_titles, start=1):
            if any(check_title_match(target, r_title) for target in case.expected_targets):
                rank_found = idx
                break

        top_1_hit = rank_found == 1
        top_3_hit = rank_found is not None and rank_found <= 3
        top_5_hit = rank_found is not None and rank_found <= 5
        top_10_hit = rank_found is not None and rank_found <= 10

        # Step 4: Exclusions Verification
        exclusions_passed = True
        for exc in case.required_exclusions:
            if any(check_title_match(exc, r_title) for r_title in retrieved_titles):
                exclusions_passed = False
                break

        # Step 5: Genre Precision % Calculation (Top-5 candidates)
        top5_genres = retrieved_genres[:5]
        if top5_genres and case.expected_genres:
            genre_matches = sum(
                1 for g in top5_genres
                if any(eg.lower() in g.lower() or g.lower() in eg.lower() for eg in case.expected_genres)
            )
            genre_precision_pct = (genre_matches / len(top5_genres)) * 100.0
        else:
            genre_precision_pct = 0.0

        # Step 6: Status Determination
        status = "PASS" if (top_5_hit and exclusions_passed) else "FAIL"

        res = EvaluationResult(
            case_id=case.id,
            category=case.category,
            prompt=case.prompt,
            expected_targets=case.expected_targets,
            retrieved_titles=retrieved_titles,
            retrieved_genres=retrieved_genres,
            top_1_hit=top_1_hit,
            top_3_hit=top_3_hit,
            top_5_hit=top_5_hit,
            top_10_hit=top_10_hit,
            rank_found=rank_found,
            genre_precision_pct=genre_precision_pct,
            parse_latency_ms=t_parse,
            retrieval_latency_ms=t_retrieval,
            total_latency_ms=t_total,
            exclusions_passed=exclusions_passed,
            status=status,
        )
        results.append(res)

        target_disp = case.expected_targets[0]
        if len(target_disp) > 30:
            target_disp = target_disp[:27] + "..."

        rank_disp = f"#{rank_found}" if rank_found else "N/A"
        top5_disp = "YES" if top_5_hit else "NO"
        excl_disp = "OK" if exclusions_passed else "FAIL"

        row = (
            f"{res.case_id:<4} | {res.category:<20} | {target_disp:<32} | "
            f"{rank_disp:<5} | {top5_disp:<5} | {excl_disp:<5} | {res.genre_precision_pct:5.1f}% | "
            f"{res.parse_latency_ms:5.0f}m | {res.retrieval_latency_ms:5.0f}m | {res.total_latency_ms:5.0f}m | {res.status:<6}"
        )
        print(row, flush=True)

    print("-" * len(hdr), flush=True)

    # Metrics Aggregation
    total_prompts = len(results)
    top1_hits = sum(1 for r in results if r.top_1_hit)
    top3_hits = sum(1 for r in results if r.top_3_hit)
    top5_hits = sum(1 for r in results if r.top_5_hit and r.exclusions_passed)
    top10_hits = sum(1 for r in results if r.top_10_hit)

    top1_rate = (top1_hits / total_prompts) * 100.0
    top3_rate = (top3_hits / total_prompts) * 100.0
    top5_success_rate = (top5_hits / total_prompts) * 100.0
    top10_rate = (top10_hits / total_prompts) * 100.0

    avg_parse_latency = sum(r.parse_latency_ms for r in results) / total_prompts
    avg_retrieval_latency = sum(r.retrieval_latency_ms for r in results) / total_prompts
    avg_total_latency = sum(r.total_latency_ms for r in results) / total_prompts
    avg_genre_precision = sum(r.genre_precision_pct for r in results) / total_prompts

    print("\n" + "=" * 110)
    print("                              GLOBAL BENCHMARK METRICS SUMMARY")
    print("=" * 110)
    print(f"  Total Prompts Evaluated          : {total_prompts}")
    print(f"  Top-1 Retrieval Accuracy         : {top1_hits}/{total_prompts} ({top1_rate:.1f}%)")
    print(f"  Top-3 Retrieval Accuracy         : {top3_hits}/{total_prompts} ({top3_rate:.1f}%)")
    print(f"  Top-5 Success Rate % (Target >=85%): {top5_hits}/{total_prompts} ({top5_success_rate:.1f}%)")
    print(f"  Top-10 Retrieval Accuracy        : {top10_hits}/{total_prompts} ({top10_rate:.1f}%)")
    print("-" * 110)
    print(f"  Average Parse Latency            : {avg_parse_latency:.2f} ms")
    print(f"  Average Retrieval Latency        : {avg_retrieval_latency:.2f} ms")
    print(f"  Average Total Latency            : {avg_total_latency:.2f} ms")
    print(f"  Average Genre Precision          : {avg_genre_precision:.1f}%")
    print("=" * 110)

    # Category Level Metrics Breakdown & Visual Bar Charts
    categories = sorted(list(set(r.category for r in results)))
    print("\n" + "=" * 110)
    print("                              CATEGORY-LEVEL TELEMETRY BAR CHARTS")
    print("=" * 110)
    print(f"{'Category':<24} | {'Prompts':<8} | {'Top-5 Hits':<10} | {'Visual Accuracy Bar Chart':<30} | {'Avg Lat (ms)':<12}")
    print("-" * 110)

    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cat_total = len(cat_results)
        cat_top5 = sum(1 for r in cat_results if r.top_5_hit and r.exclusions_passed)
        cat_success = (cat_top5 / cat_total) * 100.0 if cat_total > 0 else 0.0
        cat_lat = sum(r.total_latency_ms for r in cat_results) / cat_total if cat_total > 0 else 0.0

        # Build unbreakable ASCII bar
        width = 15
        filled = int(round((cat_success / 100.0) * width))
        empty = width - filled
        bar_chart = f"[{'█' * filled}{'░' * empty}] {cat_success:5.1f}%"

        print(f"{cat:<24} | {cat_total:<8} | {cat_top5:<10} | {bar_chart:<30} | {cat_lat:12.2f}")
    print("-" * 110 + "\n")

    passed = top5_success_rate >= 85.0
    if passed:
        print(f"✅ BENCHMARK SUCCEEDED: Overall Top-5 Success Rate {top5_success_rate:.1f}% >= Target 85.0%")
    else:
        print(f"❌ BENCHMARK FAILED: Overall Top-5 Success Rate {top5_success_rate:.1f}% < Target 85.0%")

    return passed


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
