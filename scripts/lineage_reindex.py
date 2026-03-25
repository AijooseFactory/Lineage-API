#!/usr/bin/env python3
"""Lineage full reindex script.

Rebuilds all three search layers from the canonical Gramps database:
  1. Neo4j graph projection   — relationship traversal
  2. Weaviate vector chunks   — semantic search over notes/sources/media
  3. SQLite semantic index    — search_genealogy_database tool

Run inside the API container:

    docker exec dev-lineage-api python3 /app/scripts/lineage_reindex.py

Or via docker cp + exec:

    docker cp scripts/lineage_reindex.py dev-lineage-api:/tmp/
    docker exec dev-lineage-api python3 /tmp/lineage_reindex.py

Steps that are already populated are re-synced (idempotent). Weaviate
always does a full delete + re-project to stay consistent.
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, '/app/src')

# ── Set Gramps database path BEFORE any Gramps imports ───────────────────────
# The GRAMPS_DATABASE_PATH env var is set in docker-compose but Gramps'
# internal config may not pick it up in a fresh subprocess.  We sync them.
_db_path_env = os.environ.get('GRAMPS_DATABASE_PATH') or '/root/.gramps/grampsdb'
from gramps.gen.config import config as gramps_config  # noqa: E402
gramps_config.set('database.path', _db_path_env)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('lineage-reindex')


def _find_tree_dirname(app) -> str:
    """Resolve the single-tree dirname from the Gramps database directory."""
    import uuid
    db_dir = gramps_config.get('database.path')
    if not db_dir or not os.path.isdir(db_dir):
        raise RuntimeError(f'Gramps database directory not found: {db_dir!r}')

    # Walk subdirectories looking for one that contains a name.txt
    for entry in os.scandir(db_dir):
        if not entry.is_dir():
            continue
        name_file = os.path.join(entry.path, 'name.txt')
        if os.path.isfile(name_file):
            with open(name_file, encoding='utf-8') as f:
                stored_name = f.readline().strip()
            if stored_name == app.config['TREE']:
                logger.info('Resolved tree %r → dirname %s', app.config['TREE'], entry.name)
                return entry.name

    raise RuntimeError(
        f"No Gramps database found for tree {app.config['TREE']!r} "
        f"in {db_dir!r}"
    )


def main() -> None:
    from gramps_webapi.app import create_app
    from gramps_webapi.dbmanager import WebDbManager
    from gramps_webapi.api.lineage import clients, graph_projector, vector_projector
    from gramps_webapi.api.search import get_semantic_search_indexer
    from gramps_webapi.api.search.embeddings import load_model

    app = create_app(config={})
    tree = app.config['TREE']
    logger.info('Tree: %s', tree)

    with app.app_context():
        tree_dirname = _find_tree_dirname(app)

        neo4j = clients.get_neo4j_client()
        weaviate_client = clients.get_weaviate_client()
        logger.info('Neo4j:    %s', 'ok' if neo4j and neo4j.is_available() else 'unavailable')
        logger.info('Weaviate: %s', 'ok' if weaviate_client and weaviate_client.is_available() else 'unavailable')

        mgr = WebDbManager(dirname=tree_dirname, ignore_lock=True)
        db_handle = mgr.get_db().db
        logger.info(
            'DB: %d persons, %d notes, %d sources, %d media',
            db_handle.get_number_of_people(),
            db_handle.get_number_of_notes(),
            db_handle.get_number_of_sources(),
            db_handle.get_number_of_media(),
        )

        embed_model = app.config.get('_INITIALIZED_VECTOR_EMBEDDING_MODEL')
        if embed_model is None:
            logger.info('Loading embedding model...')
            embed_model = load_model(
                model_name=app.config.get('VECTOR_EMBEDDING_MODEL', ''),
                base_url=app.config.get('LLM_BASE_URL'),
                api_key=(
                    app.config.get('OLLAMA_API_KEY')
                    or app.config.get('OPENAI_API_KEY')
                ),
            )
        embed_fn = embed_model.encode

        try:
            # ── Step 1: Neo4j ──────────────────────────────────────────────
            if neo4j and neo4j.is_available():
                logger.info('=== Step 1/3: Neo4j graph projection ===')
                count = graph_projector.project_full(neo4j, db_handle)
                logger.info('Neo4j: %d objects projected', count)
            else:
                logger.warning('Neo4j unavailable — skipping')

            # ── Step 2: Weaviate ───────────────────────────────────────────
            if weaviate_client and weaviate_client.is_available():
                logger.info('=== Step 2/3: Weaviate vector chunks (Ollama — slow) ===')
                chunks = vector_projector.project_full(weaviate_client, db_handle, tree, embed_fn)
                logger.info('Weaviate: %d chunks projected', chunks)
            else:
                logger.warning('Weaviate unavailable — skipping')

            # ── Step 3: Semantic search SQLite index ───────────────────────
            logger.info('=== Step 3/3: Semantic search index rebuild ===')
            indexer = get_semantic_search_indexer(tree)
            indexer.reindex_full(db_handle)
            logger.info('Semantic search index complete')

        finally:
            db_handle.close()

    logger.info('=== ALL DONE ===')


if __name__ == '__main__':
    main()
