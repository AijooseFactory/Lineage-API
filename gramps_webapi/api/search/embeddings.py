"""Functions to compute vector embeddings."""

from __future__ import annotations

from typing import Sequence

from ..util import get_logger


class OllamaEmbeddingModel:
    """Embedding model backed by an Ollama-compatible OpenAI /v1/embeddings endpoint.

    Uses the OpenAI-compatible REST API exposed by Ollama at
    ``{base_url}/embeddings``.  Ollama Cloud ``:cloud`` models require
    ``api_key`` to be the Ollama Cloud API key; local Ollama works
    with any non-empty string (e.g. ``"ollama"``).

    The ``.encode()`` method is API-compatible with
    ``sentence_transformers.SentenceTransformer.encode`` so that this class
    is a drop-in replacement in the semantic-search pipeline.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._client = OpenAI(
            base_url=base_url,
            # Ollama ignores the key for local models; a real key is used for
            # Ollama Cloud :cloud models.  Fall back to the sentinel "ollama"
            # that the Ollama OpenAI-compatibility layer accepts.
            api_key=api_key or "ollama",
        )
        self._logger = get_logger()
        self._logger.info(
            "OllamaEmbeddingModel ready: model=%r base_url=%r", model_name, base_url
        )

    def encode(
        self,
        texts: Sequence[str] | str,
        batch_size: int = 32,
        **kwargs,
    ) -> list[list[float]]:
        """Encode texts into embedding vectors.

        Args:
            texts: A single string or a sequence of strings to embed.
            batch_size: Maximum number of texts per API request.  Smaller
                batches reduce peak memory on the Ollama host.
            **kwargs: Accepted but ignored so the signature is compatible
                with ``sentence_transformers.SentenceTransformer.encode``.

        Returns:
            A list of embedding vectors (each a ``list[float]``).

        Raises:
            openai.OpenAIError: Propagated from the underlying HTTP call
                after logging the error.
        """
        if isinstance(texts, str):
            texts = [texts]

        texts = list(texts)
        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            self._logger.debug(
                "Requesting embeddings: model=%r batch=[%d..%d] total=%d",
                self.model_name,
                start,
                start + len(batch) - 1,
                len(texts),
            )
            try:
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as exc:
                self._logger.error(
                    "Ollama embedding request failed: model=%r batch=[%d..%d]: %s",
                    self.model_name,
                    start,
                    start + len(batch) - 1,
                    exc,
                )
                raise

        return all_embeddings


def _is_ollama_model(model_name: str) -> bool:
    """Return True when *model_name* looks like an Ollama tag (contains ``:``)."""
    return ":" in model_name


def load_model(
    model_name: str,
    base_url: str | None = None,
    api_key: str | None = None,
):
    """Load the embedding model and return an object with an ``.encode()`` method.

    **Ollama path** — selected when *model_name* contains ``:`` (the Ollama
    tag separator, e.g. ``qwen3-embedding:8b``) *and* *base_url* is provided.
    Returns an :class:`OllamaEmbeddingModel` that delegates to the Ollama
    OpenAI-compatible ``/v1/embeddings`` endpoint.  No local model download
    occurs; Ollama serves the model.

    **SentenceTransformer path** — used for all other model names (e.g.
    HuggingFace ``sentence-transformers/all-MiniLM-L6-v2``).  The model is
    downloaded and cached locally via the ``sentence_transformers`` library.

    The returned object is cached in
    ``app.config["_INITIALIZED_VECTOR_EMBEDDING_MODEL"]`` at app start-up and
    reused for every embedding request.

    Args:
        model_name: Ollama model tag or HuggingFace model identifier.
        base_url: Ollama OpenAI-compatible base URL
            (e.g. ``http://host.docker.internal:11434/v1``).  Required for the
            Ollama path; ignored otherwise.
        api_key: Ollama Cloud API key for ``:cloud`` models.  Falls back to
            the sentinel ``"ollama"`` if *None*.
    """
    logger = get_logger()
    logger.info("Initializing embedding model %r.", model_name)

    if _is_ollama_model(model_name):
        if not base_url:
            logger.warning(
                "Model %r looks like an Ollama model (contains ':') but "
                "LLM_BASE_URL is not configured.  Falling back to "
                "SentenceTransformer — this will almost certainly fail for "
                "Ollama-style tags.",
                model_name,
            )
        else:
            model = OllamaEmbeddingModel(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
            )
            logger.info("Embedding model %r initialised via Ollama.", model_name)
            return model

    # SentenceTransformer fallback (HuggingFace models)
    logger.info("Loading SentenceTransformer model %r.", model_name)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    logger.info("SentenceTransformer model %r loaded.", model_name)
    return model
