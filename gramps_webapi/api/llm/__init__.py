"""Functions for working with large language models (LLMs)."""

from __future__ import annotations

import re
from typing import Any

from flask import current_app
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ..util import abort_with_message, get_logger
from ...auth import get_tree_permissions
from .agent import create_agent
from .deps import AgentDeps


def strip_native_thinking(text: str) -> str:
    """Remove model-native thinking blocks from a response.

    Thinking models (Qwen3, DeepSeek-R1, etc.) emit their internal
    chain-of-thought inside ``<think>…</think>`` tags.  These tokens are
    part of the model's reasoning process and must **never** be shown to
    the user.  Strip them completely before any further sanitization.

    Handles:
    - Nested or malformed tags (greedy strip of anything between first
      ``<think>`` and last ``</think>``).
    - Models that open but never close the tag (strip from ``<think>`` to
      end of string so the actual answer, if any, is preserved via the
      ``re.DOTALL`` fallback).
    """
    # Remove complete <think>…</think> blocks (non-greedy, handles multi-line)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # If a <think> tag was opened but never closed, strip from it to end
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def extract_thought_block(text: str) -> tuple[str, str]:
    """Split a response into (thought, answer).

    The system prompt instructs the agent to begin with a ``<thought>…</thought>``
    block containing its visible reasoning chain, followed by the final prose
    answer.  This function separates the two so the frontend can render them
    distinctly (collapsible reasoning panel vs. answer body).

    Returns:
        (thought, answer) — both strings.  If no ``<thought>`` block is
        present the thought is empty and the entire text is returned as the
        answer.
    """
    match = re.search(
        r"<thought>(.*?)</thought>(.*)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        thought = match.group(1).strip()
        answer = match.group(2).strip()
        return thought, answer

    # No <thought> block — return as-is
    return "", text.strip()


def sanitize_answer(answer: str) -> str:
    """Sanitize the LLM answer.

    Strips model-native thinking tags (``<think>``) and cleans up any
    formatting that conflicts with the prose-only style rules.
    """
    # ── Strip native thinking-model tokens ──────────────────────────────────
    answer = strip_native_thinking(answer)

    # ── Placeholder URL cleanup ──────────────────────────────────────────────
    # Some models rewrite relative URLs to bogus absolute URLs.
    answer = answer.replace("https://www.example.com", "")
    answer = answer.replace("https://example.com", "")
    answer = answer.replace("http://example.com", "")

    # ── Markdown formatting the model should not produce ─────────────────────
    # Remove bold: **text** -> text
    answer = re.sub(r"\*\*(.*?)\*\*", r"\1", answer)
    # Remove headers: ### text -> text (at start of line)
    answer = re.sub(r"^#+\s+", "", answer, flags=re.MULTILINE)
    # Remove bullet points: - text or * text -> text (at start of line)
    answer = re.sub(r"^[-*]\s+", "", answer, flags=re.MULTILINE)
    # Remove horizontal rules: --- or *** or ___ (at start of line)
    answer = re.sub(r"^[-*_]{3,}\s*$", "", answer, flags=re.MULTILINE)

    return answer


def extract_metadata_from_result(result) -> dict[str, Any]:
    """Extract metadata from AgentRunResult.

    Args:
        result: AgentRunResult from Pydantic AI

    Returns:
        Dictionary containing run metadata including tool calls
    """
    tools_used: list[dict[str, Any]] = []
    tool_call_map = {}
    step = 0

    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    step += 1
                    tool_call_id = part.tool_call_id
                    tool_info = {
                        "step": step,
                        "name": part.tool_name,
                        "args": (
                            part.args_as_dict()
                            if hasattr(part, "args_as_dict")
                            else part.args
                        ),
                    }
                    tool_call_map[tool_call_id] = tool_info

        elif isinstance(msg, ModelRequest):
            # ModelRequest.parts can contain ToolReturnPart among other types
            for part in msg.parts:  # type: ignore[assignment]
                if isinstance(part, ToolReturnPart):
                    tool_call_id = part.tool_call_id
                    if tool_call_id in tool_call_map:
                        tools_used.append(tool_call_map[tool_call_id])

    usage = result.usage()
    metadata = {
        "run_id": result.run_id,
        "timestamp": result.timestamp().isoformat(),
        "usage": {
            "requests": usage.requests,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "tool_calls": usage.tool_calls,
        },
        "tools_used": tools_used,
    }

    return metadata


def extract_user_identity_from_history(history: list, db_handle) -> tuple[str, str]:
    """Extract user identity (handle, gramps_id) from chat history.

    Scans the conversation for identity assertions like "I am George" or "I am I0001".
    Returns (handle, gramps_id) if found, else ("", "").
    """
    logger = get_logger()
    if not history:
        return "", ""

    # Patterns to match:
    # 1. "I am I0123" or "I'm I0123"
    # 2. "I am [Name]" or "I'm [Name]" or "My name is [Name]"
    id_pattern = r"(?:i am|i'm|i\s+am)\s+(I\d+)"
    name_pattern = r"(?:i am|i'm|i\s+am|my name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"

    # Scan history from newest to oldest
    for message in reversed(history):
        if message.get("role", "").lower() != "user":
            continue
        
        content = message.get("message") or ""
        content_lower = content.lower()

        # Check for ID match first (most reliable)
        id_match = re.search(id_pattern, content_lower)
        if id_match:
            gramps_id = id_match.group(1).upper()
            try:
                person = db_handle.get_person_from_gramps_id(gramps_id)
                if person:
                    primary = person.get_primary_name()
                    name = f"{primary.get_first_name()} {primary.get_surname_list()[0].get_surname() if primary.get_surname_list() else ''}".strip()
                    logger.debug("Resolved user identity via ID: %s (%s)", gramps_id, name)
                    return person.handle, person.gramps_id, name
            except Exception: # pylint: disable=broad-except
                pass

        # Check for Name match
        name_match = re.search(name_pattern, content) # use original case for name matching
        if name_match:
            name_query = name_match.group(1).strip()
            try:
                # Simple case-insensitive name search in person handles
                for handle in db_handle.get_person_handles():
                    person = db_handle.get_person_from_handle(handle)
                    primary = person.get_primary_name()
                    if not primary:
                        continue
                    full_name = f"{primary.get_first_name()} {primary.get_surname_list()[0].get_surname() if primary.get_surname_list() else ''}".strip()
                    if name_query.lower() in full_name.lower():
                        logger.debug("Resolved user identity via Name: %s -> %s", name_query, person.gramps_id)
                        return person.handle, person.gramps_id, full_name
            except Exception: # pylint: disable=broad-except
                pass

    return "", "", ""


def answer_with_agent(
    prompt: str,
    tree: str,
    include_private: bool,
    user_id: str,
    history: list | None = None,
    frontend_home_person_gramps_id: str | None = None,
):
    """Answer a prompt using Pydantic AI agent.

    Args:
        prompt: The user's question/prompt
        tree: The tree identifier
        include_private: Whether to include private information
        user_id: The user identifier
        history: Optional chat history
        frontend_home_person_gramps_id: Optional Home Person Gramps ID from the frontend preferences

    Returns:
        AgentRunResult containing the response and metadata
    """
    logger = get_logger()

    # Get configuration
    config = current_app.config
    model_name = config.get("LLM_MODEL")
    base_url = config.get("LLM_BASE_URL")
    agent_name = config.get("LLM_AGENT_NAME", "Dottie")
    max_context_length = config.get("LLM_MAX_CONTEXT_LENGTH", 50000)

    # Admin prompt priority: per-tree DB setting → env var → none
    # This is AUGMENTED onto the base system prompt, never replaces it.
    tree_perms = get_tree_permissions(tree) or {}
    admin_prompt = (
        tree_perms.get("system_prompt_ai")
        or config.get("LLM_SYSTEM_PROMPT")
    )

    if not model_name:
        raise ValueError("No LLM model specified")

    # Fetch the Home Person from the canonical Gramps DB so the agent
    # knows who "I" and "my" refer to without asking.
    # If the frontend explicitly provides a Gramps ID for the Home Person,
    # use that to match the UI state. Otherwise, fall back to the DB default.
    home_person_handle = ""
    home_person_name = ""
    home_person_gramps_id = ""
    home_person_possessive = "their"
    try:
        from ..util import get_db_outside_request, close_db

        db_handle = get_db_outside_request(
            tree=tree,
            view_private=include_private,
            readonly=True,
            user_id=user_id,
        )
        try:
            hp = None
            if frontend_home_person_gramps_id:
                hp = db_handle.get_person_from_gramps_id(frontend_home_person_gramps_id)
            if not hp:
                hp_handle = db_handle.get_default_handle()
                if hp_handle:
                    hp = db_handle.get_person_from_handle(hp_handle)
            if hp:
                home_person_handle = hp.handle
                home_person_gramps_id = hp.gramps_id
                primary = hp.get_primary_name()
                first = primary.get_first_name() if primary else ""
                surnames = (
                    [s.get_surname() for s in primary.get_surname_list()]
                    if primary else []
                )
                surname = surnames[0] if surnames else ""
                # Title case if all caps (e.g. "FREENEY" -> "Freeney")
                if surname.isupper():
                    surname = surname.title()
                suffix = primary.get_suffix() if primary else ""
                home_person_name = f"{first} {surname}"
                if suffix:
                    home_person_name += f" {suffix}"
                home_person_name = home_person_name.strip()

                # Determine possessive pronoun
                from gramps.gen.lib import Gender
                hp_gender = hp.get_gender()
                if hp_gender == Gender.MALE:
                    home_person_possessive = "his"
                elif hp_gender == Gender.FEMALE:
                    home_person_possessive = "her"
                else:
                    home_person_possessive = "their"
        finally:
            close_db(db_handle)
    except Exception:
        logger.debug("Could not fetch Home Person — agent will ask for identity")

    # Extract user identity from history to anchor personal context
    user_person_handle = ""
    user_person_gramps_id = ""
    user_person_name = ""
    if history:
        try:
            from ..util import get_db_outside_request, close_db
            db_handle = get_db_outside_request(
                tree=tree,
                view_private=include_private,
                readonly=True,
                user_id=user_id,
            )
            try:
                user_person_handle, user_person_gramps_id, user_person_name = extract_user_identity_from_history(history, db_handle)
            finally:
                close_db(db_handle)
        except Exception:
            pass

    agent = create_agent(
        model_name=model_name,
        base_url=base_url,
        admin_prompt=admin_prompt,
        home_person_name=home_person_name,
        home_person_gramps_id=home_person_gramps_id,
        agent_name=agent_name,
        home_person_possessive=home_person_possessive,
        user_person_name=user_person_name,
        user_person_gramps_id=user_person_gramps_id,
    )

    deps = AgentDeps(
        tree=tree,
        include_private=include_private,
        max_context_length=max_context_length,
        user_id=user_id,
        home_person_handle=home_person_handle,
        home_person_name=home_person_name,
        home_person_gramps_id=home_person_gramps_id,
        home_person_possessive=home_person_possessive,
        user_person_handle=user_person_handle,
        user_person_gramps_id=user_person_gramps_id,
        agent_name=agent_name,
    )

    message_history: list[ModelRequest | ModelResponse] = []
    if history:
        for message in history:
            role = message.get("role", "").lower()
            if not role:
                raise ValueError(f"Invalid message format: {message}")
            
            # Extract content from 'message', 'answer', or 'thought'
            content = message.get("message")
            if content is None:
                # Fallback to answer or thought (Thinking UI compatibility)
                thought = message.get("thought", "")
                answer = message.get("answer", "")
                if thought and answer:
                    content = f"<thought>\n{thought}\n</thought>\n{answer}"
                elif thought:
                    content = f"<thought>\n{thought}\n</thought>"
                else:
                    content = answer

            if content is None:
                raise ValueError(f"Missing content in message: {message}")

            if role in ["ai", "system", "assistant"]:
                message_history.append(
                    ModelResponse(
                        parts=[TextPart(content=content)],
                    )
                )
            elif role != "error":  # skip error messages
                message_history.append(
                    ModelRequest(parts=[UserPromptPart(content=content)])
                )

    try:
        logger.debug("Running Pydantic AI agent with prompt: '%s'", prompt)
        result = agent.run_sync(prompt, deps=deps, message_history=message_history)
        logger.debug("Agent response: '%s'", result.response.text or "")
        return result
    except (UnexpectedModelBehavior, ModelRetry) as e:
        logger.error("Pydantic AI error: %s", e)
        abort_with_message(500, "Error communicating with the AI model")
    except Exception as e:
        logger.error("Unexpected error in agent: %s", e)
        abort_with_message(500, "Unexpected error.")
