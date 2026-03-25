"""Hybrid retrieval orchestrator for Lineage Hybrid GraphRAG.

Combines vector search (Weaviate) with graph refinement (Neo4j) and
canonical Gramps DB lookups.  Implements graceful fallback when
subsystems are unavailable.

Architecture (three data layers):
  1. Weaviate  — semantic vector search over NoteChunk, MediaChunk,
                 SourceChunk collections.
  2. Neo4j    — graph-based relationship traversal (CHILD_OF,
                SPOUSE_OF edges) for kinship reasoning.
  3. Canonical Gramps DB — the authoritative source for full person
                records (births, deaths, families, notes, media).
                Graph-found Person handles are enriched here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from gramps.gen.db.base import DbReadBase

from .clients import Neo4jClient, WeaviateClient, get_neo4j_client, get_weaviate_client
from .intent_parser import IntentResult, parse_intent_from_args, build_cypher_refinement

logger = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass
class EvidenceItem:
    """A single piece of evidence from hybrid retrieval."""

    handle: str
    gramps_id: str
    object_type: str
    text: str
    score: float = 0.0
    source: str = ""  # "weaviate", "neo4j", "canonical"
    visibility_scope: str = "public"
    needs_enrichment: bool = False  # True for graph-only results awaiting canonical lookup


@dataclass
class EvidenceBundle:
    """Collection of evidence from hybrid retrieval."""

    results: list[EvidenceItem] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    partial: bool = False  # True if a subsystem was unavailable
    query: str = ""

    def to_text(self, max_length: int = 4000) -> str:
        """Format evidence bundle as text for LLM context window."""
        if not self.results:
            return "No evidence found."

        parts = []
        parts.append(
            f"Found {len(self.results)} evidence items "
            f"(sources: {', '.join(self.sources_used)}):"
        )
        if self.partial:
            parts.append(
                "(Note: Some subsystems were unavailable — results may be incomplete.)"
            )
        parts.append("")

        per_item_max = 2000  # Enough for full canonical person records
        total_len = sum(len(p) for p in parts)
        for i, item in enumerate(self.results, 1):
            text = item.text[:per_item_max] if len(item.text) > per_item_max else item.text
            entry = (
                f"{i}. [{item.object_type}] {item.gramps_id} "
                f"(source: {item.source})\n"
                f"   {text}"
            )
            if total_len + len(entry) > max_length:
                parts.append(f"... ({len(self.results) - i + 1} more items truncated)")
                break
            parts.append(entry)
            total_len += len(entry)

        return "\n".join(parts)


# ── Hybrid Retrieval ──────────────────────────────────────────────────────────


def hybrid_retrieve(
    query: str,
    tree: str,
    user_id: str,
    include_private: bool,
    filters: dict[str, Any] | None = None,
    intent: IntentResult | None = None,
) -> EvidenceBundle:
    """Run hybrid retrieval: vector search → graph refinement → canonical enrichment.

    Three-layer pipeline:
      1. Weaviate vector search (notes, media, sources)
      2. Neo4j graph refinement (relationship traversal)
      3. Canonical Gramps DB enrichment (full person records)

    Falls back gracefully if any subsystem is unavailable.
    """
    bundle = EvidenceBundle(query=query)

    # Parse intent if not provided
    if intent is None:
        intent = parse_intent_from_args(query, **(filters or {}))

    # Get clients
    weaviate = get_weaviate_client()
    neo4j = get_neo4j_client()

    weaviate_available = weaviate is not None and weaviate.is_available()
    neo4j_available = neo4j is not None and neo4j.is_available()

    # ── Step 1: Vector retrieval (Weaviate) ──
    vector_results: list[dict[str, Any]] = []
    if weaviate_available and intent.semantic:
        try:
            vector_results = _vector_search(
                weaviate, intent, include_private
            )
            bundle.sources_used.append("weaviate")
        except Exception as exc:
            logger.warning("Weaviate vector search failed: %s", exc)
            bundle.partial = True
    elif not weaviate_available:
        bundle.partial = True
        logger.info("Weaviate unavailable — skipping vector search")

    # ── Step 2: Graph refinement (Neo4j) ──
    if neo4j_available and (vector_results or intent.filters):
        try:
            candidate_handles = [r.get("source_handle", "") for r in vector_results]
            graph_results = _graph_refine(
                neo4j, intent, candidate_handles, include_private
            )
            bundle.sources_used.append("neo4j")

            # Merge graph results with vector results
            _merge_results(bundle, vector_results, graph_results)
        except Exception as exc:
            logger.warning("Neo4j graph refinement failed: %s", exc)
            bundle.partial = True
            # Use vector results only
            _add_vector_results(bundle, vector_results)
    elif not neo4j_available:
        if not weaviate_available:
            bundle.partial = True
            logger.info("Both Neo4j and Weaviate unavailable — no hybrid results")
        else:
            bundle.partial = True
            logger.info("Neo4j unavailable — using vector results only")
            _add_vector_results(bundle, vector_results)
    else:
        _add_vector_results(bundle, vector_results)

    # ── Step 3: Canonical enrichment (Gramps DB) ──
    # Graph-found Person results only have a name.  Enrich them with
    # full records (birth, death, relationships, note text) from the
    # canonical Gramps database so the LLM gets complete data.
    _enrich_from_canonical_db(
        bundle, tree=tree, user_id=user_id, include_private=include_private
    )

    # Sort by score descending and cap at limit
    bundle.results.sort(key=lambda x: x.score, reverse=True)
    bundle.results = bundle.results[: intent.limit]

    return bundle


# ── Vector Search ─────────────────────────────────────────────────────────────


def _vector_search(
    client: WeaviateClient,
    intent: IntentResult,
    include_private: bool,
) -> list[dict[str, Any]]:
    """Search Weaviate for semantically similar chunks."""
    results: list[dict[str, Any]] = []
    embed_fn = _get_embed_fn()
    if embed_fn is None:
        logger.warning("No embedding function available — skipping vector search")
        return results

    # Generate query embedding
    query_vector = embed_fn(intent.semantic)
    if hasattr(query_vector, "tolist"):
        query_vector = query_vector.tolist()

    # Search across all chunk collections
    collections_to_search = ["NoteChunk", "MediaChunk", "SourceChunk"]

    for collection_name in collections_to_search:
        try:
            import weaviate.classes.query as wvq

            collection = client.get_collection(collection_name)

            # Build visibility filter
            if not include_private:
                vis_filter = wvq.Filter.by_property("visibility_scope").equal("public")
            else:
                vis_filter = None

            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=intent.limit,
                filters=vis_filter,
                return_metadata=wvq.MetadataQuery(distance=True),
            )

            for obj in response.objects:
                results.append(
                    {
                        "source_handle": obj.properties.get("source_handle", ""),
                        "gramps_id": obj.properties.get("gramps_id", ""),
                        "object_type": obj.properties.get("object_type", ""),
                        "chunk_text": obj.properties.get("chunk_text", ""),
                        "visibility_scope": obj.properties.get(
                            "visibility_scope", "public"
                        ),
                        "score": 1.0 - (obj.metadata.distance or 0.0),
                        "collection": collection_name,
                    }
                )
        except Exception as exc:
            logger.warning("Vector search on %s failed: %s", collection_name, exc)

    # Sort by score
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results[: intent.limit]


# ── Graph Refinement ──────────────────────────────────────────────────────────


def _graph_refine(
    client: Neo4jClient,
    intent: IntentResult,
    candidate_handles: list[str],
    include_private: bool,
) -> list[dict[str, Any]]:
    """Refine vector search results using graph queries."""
    results: list[dict[str, Any]] = []

    # Build filter conditions
    cypher_filter, params = build_cypher_refinement(intent)

    # Visibility filter
    if not include_private:
        visibility_clause = "AND p.visibility_scope = 'public'"
    else:
        visibility_clause = ""

    # If we have relationship constraints, use graph traversal
    if intent.filters.get("relationship_type") and intent.filters.get(
        "relationship_target_id"
    ):
        max_hops = intent.filters.get("max_hops", 6)
        rel_type = intent.filters["relationship_type"]
        target_id = intent.filters["relationship_target_id"]

        if rel_type == "ancestor":
            # Follow PARENT_OF edges INBOUND from target to find its ancestors.
            # PARENT_OF goes from parent→child, so ancestors are found by
            # traversing it backwards.  This covers BOTH maternal and paternal
            # lines because PARENT_OF edges are created for every father/mother
            # in the family node, not just one side.
            cypher = f"""
            MATCH (target:Person {{gramps_id: $target_id}})
            MATCH path = (ancestor:Person)-[:PARENT_OF*1..{max_hops}]->(target)
            WHERE ancestor <> target AND {cypher_filter} {visibility_clause}
            RETURN DISTINCT ancestor.handle AS handle,
                   ancestor.gramps_id AS gramps_id,
                   ancestor.first_name + ' ' + ancestor.surname AS name,
                   length(path) AS distance
            ORDER BY distance
            LIMIT $limit
            """
        elif rel_type == "descendant":
            # Follow PARENT_OF edges OUTBOUND from target to find descendants.
            cypher = f"""
            MATCH (target:Person {{gramps_id: $target_id}})
            MATCH path = (target)-[:PARENT_OF*1..{max_hops}]->(descendant:Person)
            WHERE descendant <> target AND {cypher_filter} {visibility_clause}
            RETURN DISTINCT descendant.handle AS handle,
                   descendant.gramps_id AS gramps_id,
                   descendant.first_name + ' ' + descendant.surname AS name,
                   length(path) AS distance
            ORDER BY distance
            LIMIT $limit
            """
        else:
            # General "related" — undirected PARENT_OF traversal covers both
            # ancestors and descendants; SPOUSE_OF for spousal connections.
            cypher = f"""
            MATCH (target:Person {{gramps_id: $target_id}})
            MATCH path = (target)-[:PARENT_OF|SPOUSE_OF*1..{max_hops}]-(related:Person)
            WHERE related <> target AND {cypher_filter} {visibility_clause}
            RETURN DISTINCT related.handle AS handle,
                   related.gramps_id AS gramps_id,
                   related.first_name + ' ' + related.surname AS name,
                   length(path) AS distance
            ORDER BY distance
            LIMIT $limit
            """

        params["target_id"] = target_id
        params["limit"] = intent.limit

        rows = client.execute_read(cypher, params)
        for row in rows:
            results.append(
                {
                    "handle": row.get("handle", ""),
                    "gramps_id": row.get("gramps_id", ""),
                    "name": row.get("name", ""),
                    "distance": row.get("distance", 0),
                    "object_type": "Person",
                    "source": "neo4j",
                }
            )

    elif candidate_handles:
        # Filter candidates by graph properties
        cypher = f"""
        MATCH (p:Person)
        WHERE p.handle IN $handles AND {cypher_filter} {visibility_clause}
        RETURN p.handle AS handle, p.gramps_id AS gramps_id,
               p.first_name + ' ' + p.surname AS name
        LIMIT $limit
        """
        params["handles"] = [h for h in candidate_handles if h]
        params["limit"] = intent.limit

        rows = client.execute_read(cypher, params)
        for row in rows:
            results.append(
                {
                    "handle": row.get("handle", ""),
                    "gramps_id": row.get("gramps_id", ""),
                    "name": row.get("name", ""),
                    "object_type": "Person",
                    "source": "neo4j",
                }
            )

    return results


# ── Merge & Format ────────────────────────────────────────────────────────────


def _add_vector_results(
    bundle: EvidenceBundle, vector_results: list[dict[str, Any]]
) -> None:
    """Add vector search results to the evidence bundle."""
    for vr in vector_results:
        bundle.results.append(
            EvidenceItem(
                handle=vr.get("source_handle", ""),
                gramps_id=vr.get("gramps_id", ""),
                object_type=vr.get("object_type", ""),
                text=vr.get("chunk_text", ""),
                score=vr.get("score", 0.0),
                source="weaviate",
                visibility_scope=vr.get("visibility_scope", "public"),
            )
        )


def _merge_results(
    bundle: EvidenceBundle,
    vector_results: list[dict[str, Any]],
    graph_results: list[dict[str, Any]],
) -> None:
    """Merge vector and graph results into the evidence bundle.

    Graph results boost scores of matching vector results.
    """
    graph_handles = {r.get("handle", "") for r in graph_results}

    # Add vector results, boosting those confirmed by graph
    for vr in vector_results:
        score = vr.get("score", 0.0)
        if vr.get("source_handle", "") in graph_handles:
            score *= 1.5  # Boost score for graph-confirmed results

        bundle.results.append(
            EvidenceItem(
                handle=vr.get("source_handle", ""),
                gramps_id=vr.get("gramps_id", ""),
                object_type=vr.get("object_type", ""),
                text=vr.get("chunk_text", ""),
                score=score,
                source="weaviate+neo4j" if vr.get("source_handle", "") in graph_handles else "weaviate",
                visibility_scope=vr.get("visibility_scope", "public"),
            )
        )

    # Add graph-only results (not already in vector results)
    # These get a placeholder text that will be enriched in Step 3
    # (canonical DB lookup) with full person records.
    vector_handles = {vr.get("source_handle", "") for vr in vector_results}
    for gr in graph_results:
        if gr.get("handle", "") not in vector_handles:
            bundle.results.append(
                EvidenceItem(
                    handle=gr.get("handle", ""),
                    gramps_id=gr.get("gramps_id", ""),
                    object_type=gr.get("object_type", "Person"),
                    text=gr.get("name", ""),
                    score=0.5,
                    source="neo4j",
                    needs_enrichment=True,
                )
            )


# ── Canonical DB Enrichment ───────────────────────────────────────────────────


def _enrich_from_canonical_db(
    bundle: EvidenceBundle,
    tree: str,
    user_id: str,
    include_private: bool,
) -> None:
    """Enrich graph-found results with full records from the canonical Gramps DB.

    Graph-found Person nodes only carry a name and handle.  This step
    looks them up in the canonical database and replaces the placeholder
    text with the same rich, formatted record that filter_people returns
    (births, deaths, relationships, notes, links).
    """
    items_to_enrich = [r for r in bundle.results if r.needs_enrichment]
    if not items_to_enrich:
        return

    db_handle = None
    try:
        from ..util import get_db_outside_request
        from ..search.text import obj_strings_from_object

        db_handle = get_db_outside_request(
            tree=tree,
            view_private=include_private,
            readonly=True,
            user_id=user_id,
        )

        for item in items_to_enrich:
            try:
                obj_type = item.object_type or "Person"
                get_method_name = f"get_{obj_type.lower()}_from_handle"
                get_method = getattr(db_handle, get_method_name, None)
                if get_method is None:
                    continue

                obj = get_method(item.handle)
                if obj is None:
                    continue

                # Privacy check
                if not include_private and hasattr(obj, "private") and obj.private:
                    continue

                obj_dict = obj_strings_from_object(
                    db_handle=db_handle,
                    class_name=obj_type,
                    obj=obj,
                    semantic=True,
                )
                if obj_dict:
                    content = (
                        obj_dict["string_all"]
                        if include_private
                        else obj_dict["string_public"]
                    )
                    if content:
                        item.text = content
                        item.source = f"{item.source}+canonical"
                        item.needs_enrichment = False
                        # Boost score — canonical-enriched results are high quality
                        item.score = max(item.score, 0.8)

            except Exception as exc:
                logger.debug(
                    "Could not enrich %s/%s from canonical DB: %s",
                    item.object_type,
                    item.handle,
                    exc,
                )

        if "canonical" not in bundle.sources_used:
            bundle.sources_used.append("canonical")

    except ImportError:
        logger.debug("Canonical enrichment skipped — dependencies not available")
    except Exception as exc:
        logger.warning("Canonical DB enrichment failed: %s", exc)
        bundle.partial = True
    finally:
        if db_handle is not None:
            try:
                db_handle.close()
            except Exception:
                pass


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_embed_fn() -> Optional[Callable]:
    """Get the embedding function from the Flask app config."""
    try:
        from flask import current_app

        model = current_app.config.get("_INITIALIZED_VECTOR_EMBEDDING_MODEL")
        if model:
            return model.encode
    except RuntimeError:
        pass
    return None
