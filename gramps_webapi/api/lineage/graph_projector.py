"""Neo4j graph projection for Lineage Hybrid GraphRAG.

Creates and maintains derived graph nodes and relationships from
canonical Gramps data. All Cypher is parameterized — no model-generated
queries. The canonical Gramps handle is the merge key for every node.
"""

from __future__ import annotations

import logging
from typing import Any

from gramps.gen.db.base import DbReadBase

from .clients import Neo4jClient, get_neo4j_client
from .normalizer import (
    normalize_person,
    normalize_family,
    normalize_event,
    normalize_place,
    normalize_source,
    normalize_citation,
    normalize_note,
    normalize_media,
)

logger = logging.getLogger(__name__)


# ── Projection Functions ─────────────────────────────────────────────────────


def project_person(client: Neo4jClient, person_dict: dict[str, Any]) -> None:
    """MERGE a Person node and its relationships."""
    # Merge the Person node
    client.execute_write(
        """
        MERGE (p:Person {handle: $handle})
        SET p.gramps_id = $gramps_id,
            p.first_name = $first_name,
            p.surname = $surname,
            p.gender = $gender,
            p.birth_date = $birth_date,
            p.birth_place = $birth_place,
            p.death_date = $death_date,
            p.death_place = $death_place,
            p.visibility_scope = $visibility_scope
        """,
        person_dict,
    )

    # CHILD_OF relationships (person → parent families)
    for family_handle in person_dict.get("parent_family_list", []):
        client.execute_write(
            """
            MATCH (p:Person {handle: $person_handle})
            MERGE (f:Family {handle: $family_handle})
            MERGE (p)-[:CHILD_OF]->(f)
            """,
            {"person_handle": person_dict["handle"], "family_handle": family_handle},
        )

    # SPOUSE_OF relationships (person → own families)
    for family_handle in person_dict.get("family_list", []):
        client.execute_write(
            """
            MATCH (p:Person {handle: $person_handle})
            MERGE (f:Family {handle: $family_handle})
            MERGE (p)-[:SPOUSE_OF]->(f)
            """,
            {"person_handle": person_dict["handle"], "family_handle": family_handle},
        )

    # PARTICIPATED_IN relationships (person → events)
    for event_ref in person_dict.get("event_ref_list", []):
        client.execute_write(
            """
            MATCH (p:Person {handle: $person_handle})
            MERGE (e:Event {handle: $event_handle})
            MERGE (p)-[:PARTICIPATED_IN {role: $role}]->(e)
            """,
            {
                "person_handle": person_dict["handle"],
                "event_handle": event_ref["handle"],
                "role": event_ref["role"],
            },
        )

    # HAS_NOTE relationships
    for note_handle in person_dict.get("note_list", []):
        client.execute_write(
            """
            MATCH (p:Person {handle: $person_handle})
            MERGE (n:Note {handle: $note_handle})
            MERGE (p)-[:HAS_NOTE]->(n)
            """,
            {"person_handle": person_dict["handle"], "note_handle": note_handle},
        )

    # ATTACHED_TO relationships (person → media)
    for media_handle in person_dict.get("media_list", []):
        client.execute_write(
            """
            MATCH (p:Person {handle: $person_handle})
            MERGE (m:Media {handle: $media_handle})
            MERGE (p)-[:ATTACHED_TO]->(m)
            """,
            {"person_handle": person_dict["handle"], "media_handle": media_handle},
        )


def project_family(client: Neo4jClient, family_dict: dict[str, Any]) -> None:
    """MERGE a Family node and its relationships."""
    client.execute_write(
        """
        MERGE (f:Family {handle: $handle})
        SET f.gramps_id = $gramps_id,
            f.type = $type,
            f.visibility_scope = $visibility_scope
        """,
        family_dict,
    )

    # Link father
    if family_dict.get("father_handle"):
        client.execute_write(
            """
            MATCH (f:Family {handle: $family_handle})
            MERGE (p:Person {handle: $father_handle})
            MERGE (p)-[:SPOUSE_OF]->(f)
            """,
            {
                "family_handle": family_dict["handle"],
                "father_handle": family_dict["father_handle"],
            },
        )

    # Link mother
    if family_dict.get("mother_handle"):
        client.execute_write(
            """
            MATCH (f:Family {handle: $family_handle})
            MERGE (p:Person {handle: $mother_handle})
            MERGE (p)-[:SPOUSE_OF]->(f)
            """,
            {
                "family_handle": family_dict["handle"],
                "mother_handle": family_dict["mother_handle"],
            },
        )

    # Link children — CHILD_OF (child→family) + PARENT_OF (parent→child) shortcuts
    for child_handle in family_dict.get("child_ref_list", []):
        client.execute_write(
            """
            MATCH (f:Family {handle: $family_handle})
            MERGE (p:Person {handle: $child_handle})
            MERGE (p)-[:CHILD_OF]->(f)
            """,
            {
                "family_handle": family_dict["handle"],
                "child_handle": child_handle,
            },
        )
        # Direct person-to-person PARENT_OF edges for efficient ancestor traversal.
        # These avoid the 2-hop CHILD_OF→SPOUSE_OF alternation in Cypher.
        for parent_key in ("father_handle", "mother_handle"):
            parent_handle = family_dict.get(parent_key)
            if parent_handle:
                client.execute_write(
                    """
                    MERGE (parent:Person {handle: $parent_handle})
                    MERGE (child:Person {handle: $child_handle})
                    MERGE (parent)-[:PARENT_OF]->(child)
                    """,
                    {
                        "parent_handle": parent_handle,
                        "child_handle": child_handle,
                    },
                )


def project_event(client: Neo4jClient, event_dict: dict[str, Any]) -> None:
    """MERGE an Event node and its relationships."""
    client.execute_write(
        """
        MERGE (e:Event {handle: $handle})
        SET e.gramps_id = $gramps_id,
            e.type = $type,
            e.date = $date,
            e.place_name = $place_name,
            e.description = $description,
            e.visibility_scope = $visibility_scope
        """,
        event_dict,
    )

    # TOOK_PLACE_IN
    if event_dict.get("place_handle"):
        client.execute_write(
            """
            MATCH (e:Event {handle: $event_handle})
            MERGE (p:Place {handle: $place_handle})
            MERGE (e)-[:TOOK_PLACE_IN]->(p)
            """,
            {
                "event_handle": event_dict["handle"],
                "place_handle": event_dict["place_handle"],
            },
        )

    # HAS_NOTE
    for note_handle in event_dict.get("note_list", []):
        client.execute_write(
            """
            MATCH (e:Event {handle: $event_handle})
            MERGE (n:Note {handle: $note_handle})
            MERGE (e)-[:HAS_NOTE]->(n)
            """,
            {"event_handle": event_dict["handle"], "note_handle": note_handle},
        )


def project_place(client: Neo4jClient, place_dict: dict[str, Any]) -> None:
    """MERGE a Place node."""
    client.execute_write(
        """
        MERGE (p:Place {handle: $handle})
        SET p.gramps_id = $gramps_id,
            p.title = $title,
            p.name = $name,
            p.visibility_scope = $visibility_scope
        """,
        place_dict,
    )


def project_source(client: Neo4jClient, source_dict: dict[str, Any]) -> None:
    """MERGE a Source node and its note relationships."""
    client.execute_write(
        """
        MERGE (s:Source {handle: $handle})
        SET s.gramps_id = $gramps_id,
            s.title = $title,
            s.author = $author,
            s.pubinfo = $pubinfo,
            s.visibility_scope = $visibility_scope
        """,
        source_dict,
    )

    for note_handle in source_dict.get("note_list", []):
        client.execute_write(
            """
            MATCH (s:Source {handle: $source_handle})
            MERGE (n:Note {handle: $note_handle})
            MERGE (s)-[:HAS_NOTE]->(n)
            """,
            {"source_handle": source_dict["handle"], "note_handle": note_handle},
        )


def project_citation(client: Neo4jClient, citation_dict: dict[str, Any]) -> None:
    """MERGE a Citation node and its REFERENCES relationship to Source."""
    client.execute_write(
        """
        MERGE (c:Citation {handle: $handle})
        SET c.gramps_id = $gramps_id,
            c.page = $page,
            c.confidence = $confidence,
            c.visibility_scope = $visibility_scope
        """,
        citation_dict,
    )

    # REFERENCES → Source
    if citation_dict.get("source_handle"):
        client.execute_write(
            """
            MATCH (c:Citation {handle: $citation_handle})
            MERGE (s:Source {handle: $source_handle})
            MERGE (c)-[:REFERENCES]->(s)
            """,
            {
                "citation_handle": citation_dict["handle"],
                "source_handle": citation_dict["source_handle"],
            },
        )


def project_note(client: Neo4jClient, note_dict: dict[str, Any]) -> None:
    """MERGE a Note node."""
    client.execute_write(
        """
        MERGE (n:Note {handle: $handle})
        SET n.gramps_id = $gramps_id,
            n.type = $type,
            n.visibility_scope = $visibility_scope
        """,
        {
            "handle": note_dict["handle"],
            "gramps_id": note_dict["gramps_id"],
            "type": note_dict["type"],
            "visibility_scope": note_dict["visibility_scope"],
        },
    )


def project_media(client: Neo4jClient, media_dict: dict[str, Any]) -> None:
    """MERGE a Media node."""
    client.execute_write(
        """
        MERGE (m:Media {handle: $handle})
        SET m.gramps_id = $gramps_id,
            m.path = $path,
            m.description = $description,
            m.mime = $mime,
            m.visibility_scope = $visibility_scope
        """,
        media_dict,
    )


# ── Delete ────────────────────────────────────────────────────────────────────


def delete_projection(client: Neo4jClient, handle: str, class_name: str) -> None:
    """Remove a node and all its relationships from Neo4j."""
    label = class_name.capitalize()
    # DETACH DELETE removes the node and all relationships
    client.execute_write(
        f"MATCH (n:{label} {{handle: $handle}}) DETACH DELETE n",
        {"handle": handle},
    )


# ── Bulk / Incremental ───────────────────────────────────────────────────────

# Map Gramps class names to normalizer + projector functions
_PROJECTORS = {
    "Person": (normalize_person, project_person),
    "Family": (normalize_family, project_family),
    "Event": (normalize_event, project_event),
    "Place": (normalize_place, project_place),
    "Source": (normalize_source, project_source),
    "Citation": (normalize_citation, project_citation),
    "Note": (normalize_note, project_note),
    "Media": (normalize_media, project_media),
}


def bootstrap_parent_of_edges(client: Neo4jClient) -> int:
    """Create PARENT_OF edges from the existing CHILD_OF / SPOUSE_OF graph.

    Safe to run multiple times (uses MERGE). Useful after upgrading from a
    schema version that did not have PARENT_OF edges.

    Returns:
        Number of PARENT_OF relationships created or confirmed.
    """
    client.execute_write(
        """
        MATCH (parent:Person)-[:SPOUSE_OF]->(family:Family)<-[:CHILD_OF]-(child:Person)
        MERGE (parent)-[:PARENT_OF]->(child)
        """,
        {},
    )
    # Count edges created
    result = client.execute_read(
        "MATCH ()-[:PARENT_OF]->() RETURN count(*) AS cnt",
        {},
    )
    count = result[0]["cnt"] if result else 0
    logger.info("bootstrap_parent_of_edges: %d PARENT_OF edges ensured", count)
    return count


def project_incremental(
    client: Neo4jClient, db_handle: DbReadBase, handle: str, class_name: str
) -> None:
    """Project a single object from canonical DB to Neo4j."""
    if class_name not in _PROJECTORS:
        logger.debug("No graph projector for class %s — skipping", class_name)
        return

    normalizer_func, projector_func = _PROJECTORS[class_name]

    # Fetch the object from canonical DB (read-only)
    query_method = db_handle.method("get_%s_from_handle", class_name)
    if query_method is None:
        return
    try:
        obj = query_method(handle)
    except Exception:
        logger.warning("Object %s/%s not found in canonical DB", class_name, handle)
        return

    obj_dict = normalizer_func(db_handle, obj)
    projector_func(client, obj_dict)


def project_full(
    client: Neo4jClient,
    db_handle: DbReadBase,
    progress_cb=None,
) -> int:
    """Full projection of all objects from canonical DB to Neo4j.

    Returns the total number of objects projected.
    """
    from ...const import GRAMPS_OBJECT_PLURAL

    # Clear existing graph data
    client.execute_write("MATCH (n) DETACH DELETE n")
    client.bootstrap_schema()

    total = 0
    count = 0

    for class_name, (normalizer_func, projector_func) in _PROJECTORS.items():
        plural_name = GRAMPS_OBJECT_PLURAL.get(class_name)
        if not plural_name:
            continue
        iter_method = db_handle.method("iter_%s", plural_name)
        if iter_method is None:
            continue
        for obj in iter_method():
            obj_dict = normalizer_func(db_handle, obj)
            projector_func(client, obj_dict)
            count += 1
            if progress_cb and count % 50 == 0:
                progress_cb(current=count, total=total)

    logger.info("Neo4j full projection complete: %d objects projected", count)
    return count
