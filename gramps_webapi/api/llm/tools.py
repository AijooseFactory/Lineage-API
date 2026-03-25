#
# Gramps Web API - A RESTful API for the Gramps genealogy program
#
# Copyright (C) 2025      David Straub
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

"""Pydantic AI tools for LLM interactions."""

from __future__ import annotations

import json
from datetime import datetime
from functools import wraps
from typing import Any

from pydantic_ai import RunContext

from ..resources.filters import apply_filter
from ..resources.util import (
    get_one_relationship,
    get_all_relationships_string,
    get_event_participants_for_handle,
)
from ..search import get_semantic_search_indexer
from ..search.text import obj_strings_from_object
from ..util import get_db_outside_request, get_logger
from .deps import AgentDeps


def _build_date_expression(before: str, after: str) -> str:
    """Build a date string from before/after parameters.

    Args:
        before: Year before which to filter (e.g., "1900")
        after: Year after which to filter (e.g., "1850")

    Returns:
        A date string for Gramps filters:
        - "between 1850 and 1900" for date ranges
        - "after 1850" for only after
        - "before 1900" for only before
    """
    if before and after:
        return f"between {after} and {before}"
    if after:
        return f"after {after}"
    if before:
        return f"before {before}"
    return ""


def _get_relationship_prefix(db_handle, anchor_person, result_person, logger) -> str:
    """Get a relationship string prefix for a result person.

    Args:
        db_handle: Database handle
        anchor_person: The Person object to calculate relationship from
        result_person: The Person object to calculate relationship to
        logger: Logger instance

    Returns:
        A formatted relationship prefix like "[grandfather] " or "[grandfather, uncle] " or empty string
    """
    try:
        # Use get_all_relationships_string to catch double-relationships (endogamy)
        rel_string = get_all_relationships_string(
            db_handle=db_handle,
            person1=anchor_person,
            person2=result_person,
            depth=12,  # Increased depth for broader relationship detection
        )
        if rel_string and rel_string.lower() not in ["", "self"]:
            return f"[{rel_string}] "
        
        # Fallback for self check if get_all_relationships_string is empty
        if anchor_person.handle == result_person.handle:
            return "[self] "
            
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(
            "Error calculating relationship between %s and %s: %s",
            anchor_person.gramps_id,
            result_person.gramps_id,
            e,
        )
    return ""


def _apply_gramps_filter(
    ctx: RunContext[AgentDeps],
    namespace: str,
    rules: list[dict[str, Any]],
    max_results: int,
    empty_message: str = "No results found matching the filter criteria.",
    show_relation_with: str = "",
) -> str:
    """Apply a Gramps filter and return formatted results.

    This is a common helper for filter tools that handles:
    - Database handle management
    - Filter application
    - Result iteration with privacy checking
    - Context length limiting
    - Truncation messages
    - Error handling
    - Optional relationship calculation

    Args:
        ctx: The Pydantic AI run context with dependencies
        namespace: Gramps object namespace ("Person", "Event", "Family", etc.)
        rules: List of filter rule dictionaries
        max_results: Maximum number of results to return (already validated)
        empty_message: Message to return when no results found
        show_relation_with: Gramps ID of anchor person for relationship calculation (Person namespace only)

    Returns:
        Formatted string with matching objects or error message
    """
    logger = get_logger()
    db_handle = None

    try:
        # Use get_db_outside_request to avoid Flask's g caching, since Pydantic AI's
        # run_sync() uses an event loop that can violate SQLite's thread-safety.
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )

        filter_dict: dict[str, Any] = {"rules": rules}
        if len(rules) > 1:
            filter_dict["function"] = "and"

        filter_rules = json.dumps(filter_dict)
        logger.debug("%s filter rules: %s", namespace, filter_rules)

        args = {"rules": filter_rules}
        matching_handles = apply_filter(
            db_handle=db_handle,
            args=args,
            namespace=namespace,
            handles=None,
        )

        if not matching_handles:
            db_handle.close()
            return empty_message

        total_matches = len(matching_handles)
        matching_handles = matching_handles[:max_results]

        context_parts: list[str] = []
        max_length = ctx.deps.max_context_length
        per_item_max = 10000  # Maximum chars per individual item
        current_length = 0

        # Get the anchor person for relationship calculation if requested
        anchor_person = None
        if show_relation_with and namespace == "Person":
            try:
                anchor_person = db_handle.get_person_from_gramps_id(show_relation_with)
                if not anchor_person:
                    logger.warning(
                        "Anchor person %s not found for relationship calculation",
                        show_relation_with,
                    )
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(
                    "Error fetching anchor person %s: %s", show_relation_with, e
                )

        # Get the appropriate method to fetch objects
        get_method_name = f"get_{namespace.lower()}_from_handle"
        get_method = getattr(db_handle, get_method_name)

        for handle in matching_handles:
            try:
                obj = get_method(handle)

                if not ctx.deps.include_private and obj.private:
                    continue

                obj_dict = obj_strings_from_object(
                    db_handle=db_handle,
                    class_name=namespace,
                    obj=obj,
                    semantic=True,
                )

                if not obj_dict:
                    continue

                # obj_strings_from_object always returns string_all/string_public
                content = (
                    obj_dict["string_all"]
                    if ctx.deps.include_private
                    else obj_dict["string_public"]
                )

                if not content:
                    continue

                # Add relationship prefix if anchor person is set
                if anchor_person and namespace == "Person":
                    rel_prefix = _get_relationship_prefix(
                        db_handle, anchor_person, obj, logger
                    )
                    content = rel_prefix + content

                # Truncate individual items if they're too long
                if len(content) > per_item_max:
                    content = (
                        content[:per_item_max]
                        + "\n\n[Content truncated due to length...]"
                    )
                    logger.debug(
                        "Truncated %s content from %d to %d chars",
                        namespace,
                        len(content) - per_item_max,
                        per_item_max,
                    )

                # Check if adding this item would exceed total limit
                if current_length + len(content) > max_length:
                    logger.debug(
                        "Reached max context length (%d chars), stopping at %d results",
                        max_length,
                        len(context_parts),
                    )
                    break

                context_parts.append(content)
                current_length += len(content) + 2

            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Error processing %s %s: %s", namespace, handle, e)
                continue

        if not context_parts:
            db_handle.close()
            return f"{empty_message} (or all results are private)."

        result = "\n\n".join(context_parts)

        # Add truncation messages
        returned_count = len(context_parts)
        if returned_count < total_matches:
            result += f"\n\n---\nShowing {returned_count} of {total_matches} matching {namespace.lower()}s. Use max_results parameter to see more."
        elif total_matches == max_results:
            result += f"\n\n---\nShowing {returned_count} {namespace.lower()}s (limit reached). There may be more matches."

        logger.debug(
            "Tool filter_%ss returned %d results (%d chars)",
            namespace.lower(),
            returned_count,
            len(result),
        )

        db_handle.close()
        return result

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error filtering %ss: %s", namespace.lower(), e)
        if db_handle is not None:
            try:
                db_handle.close()
            except Exception:  # pylint: disable=broad-except
                pass
        return f"Error filtering {namespace.lower()}s: {str(e)}"


def log_tool_call(func):
    """Decorator to log tool usage."""
    logger = get_logger()

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug("Tool called: %s", func.__name__)
        return func(*args, **kwargs)

    return wrapper


@log_tool_call
def get_current_date(_ctx: RunContext[AgentDeps]) -> str:
    """Returns today's date in ISO format (YYYY-MM-DD)."""
    logger = get_logger()

    result = datetime.now().date().isoformat()
    logger.debug("Tool get_current_date returned: %s", result)
    return result


@log_tool_call
def search_genealogy_database(
    ctx: RunContext[AgentDeps], query: str, max_results: int = 20
) -> str:
    """Searches the user's family tree using semantic similarity.

    Args:
        query: Search query for genealogical information
        max_results: Maximum results to return (default: 20, max: 50)

    Returns:
        Formatted genealogical data including people, families, events, places,
        sources, citations, repositories, notes, and media matching the query.
    """

    logger = get_logger()

    # Limit max_results to reasonable bounds
    max_results = min(max(1, max_results), 50)

    db_handle = None
    try:
        # Use get_db_outside_request to avoid Flask's g caching
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )

        searcher = get_semantic_search_indexer(ctx.deps.tree)
        _, hits = searcher.search(
            query=query,
            page=1,
            pagesize=max_results,
            include_private=ctx.deps.include_private,
            include_content=True,
        )

        if not hits:
            return "No results found in the genealogy database."

        context_parts: list[str] = []
        max_length = ctx.deps.max_context_length
        per_item_max = 10000  # Maximum chars per individual item
        current_length = 0

        # Get anchor person for relationship labeling
        # Prioritize User Person over Home Person for personal context
        anchor_person = None
        if ctx.deps.user_person_handle:
            try:
                anchor_person = db_handle.get_person_from_handle(
                    ctx.deps.user_person_handle
                )
            except Exception: # pylint: disable=broad-except
                pass
        elif ctx.deps.user_person_gramps_id:
            try:
                anchor_person = db_handle.get_person_from_gramps_id(
                    ctx.deps.user_person_gramps_id
                )
            except Exception: # pylint: disable=broad-except
                pass
        
        # Fall back to Home Person if user identity is not established
        if not anchor_person:
            if ctx.deps.home_person_handle:
                try:
                    anchor_person = db_handle.get_person_from_handle(
                        ctx.deps.home_person_handle
                    )
                except Exception: # pylint: disable=broad-except
                    pass
            elif ctx.deps.home_person_gramps_id:
                try:
                    anchor_person = db_handle.get_person_from_gramps_id(
                        ctx.deps.home_person_gramps_id
                    )
                except Exception: # pylint: disable=broad-except
                    pass

        for hit in hits:
            content = hit.get("content", "")
            obj_type = hit.get("object_type", "")
            handle = hit.get("handle")
            
            # Resolve the "Subject" person for the hit to provide relationship context
            subject_person = None
            if anchor_person:
                try:
                    if obj_type == "person":
                        subject_person = db_handle.get_person_from_handle(handle)
                    elif obj_type == "event":
                        participants = get_event_participants_for_handle(db_handle, handle)
                        if participants["people"]:
                            # Prioritize primary participants
                            subject_person = next(
                                (p for r, p in participants["people"] if r.is_primary()),
                                participants["people"][0][1]
                            )
                    elif obj_type == "note":
                        # Find first person referencing this note
                        backlinks = db_handle.find_backlink_handles(
                            handle, include_classes=["Person"]
                        )
                        if backlinks:
                            _, p_handle = backlinks[0]
                            subject_person = db_handle.get_person_from_handle(p_handle)
                except Exception: # pylint: disable=broad-except
                    pass

            if subject_person and anchor_person:
                try:
                    rel_prefix = _get_relationship_prefix(
                        db_handle, anchor_person, subject_person, logger
                    )
                    if obj_type == "person":
                        content = rel_prefix + content
                    else:
                        person_name = str(subject_person.get_primary_name().get_name())
                        content = f"{rel_prefix}{person_name}'s {obj_type.capitalize()}: {content}"
                except Exception: # pylint: disable=broad-except
                    pass

            # Truncate individual items if they're too long
            if len(content) > per_item_max:
                content = (
                    content[:per_item_max] + "\n\n[Content truncated due to length...]"
                )
                logger.debug(
                    "Truncated search result from %d to %d chars",
                    len(content) - per_item_max,
                    per_item_max,
                )

            if current_length + len(content) > max_length:
                logger.debug(
                    "Reached max context length (%d chars), stopping at %d results",
                    max_length,
                    len(context_parts),
                )
                break
            context_parts.append(content)
            current_length += len(content) + 2

        db_handle.close()
        result = "\n\n".join(context_parts)
        logger.debug(
            "Tool search_genealogy_database returned %d results (%d chars) for query: %r",
            len(context_parts),
            len(result),
            query,
        )
        return result

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error searching genealogy database: %s", e)
        if db_handle:
            db_handle.close()
        return f"Error searching the database: {str(e)}"


@log_tool_call
def filter_people(
    ctx: RunContext[AgentDeps],
    given_name: str = "",
    surname: str = "",
    birth_year_before: str = "",
    birth_year_after: str = "",
    birth_place: str = "",
    death_year_before: str = "",
    death_year_after: str = "",
    death_place: str = "",
    ancestor_of: str = "",
    ancestor_generations: int = 10,
    descendant_of: str = "",
    descendant_generations: int = 10,
    is_male: bool = False,
    is_female: bool = False,
    probably_alive_on_date: str = "",
    has_common_ancestor_with: str = "",
    degrees_of_separation_from: str = "",
    degrees_of_separation: int = 2,
    combine_filters: str = "and",
    max_results: int = 50,
    show_relation_with: str = "",
) -> str:
    """Filters people in the family tree based on simple criteria.

    IMPORTANT: When filtering by relationships (ancestor_of, descendant_of, degrees_of_separation_from),
    ALWAYS set show_relation_with to the same Gramps ID to get relationship labels in results.
    Without it, you cannot determine specific relationships like "grandfather" vs "father".

    Args:
        given_name: Given/first name to search for (partial match)
        surname: Surname/last name to search for (partial match)
        birth_year_before: Year before which people were born (e.g., "1900"). Use only the year.
        birth_year_after: Year after which people were born (e.g., "1850"). Use only the year.
        birth_place: Place name where person was born (partial match)
        death_year_before: Year before which people died (e.g., "1950"). Use only the year.
        death_year_after: Year after which people died (e.g., "1800"). Use only the year.
        death_place: Place name where person died (partial match)
        ancestor_of: Gramps ID of person to find ancestors of (e.g., "I0044")
        ancestor_generations: Maximum generations to search for ancestors (default: 10)
        descendant_of: Gramps ID of person to find descendants of (e.g., "I0044")
        descendant_generations: Maximum generations to search for descendants (default: 10)
        is_male: Filter to only males (True/False)
        is_female: Filter to only females (True/False)
        probably_alive_on_date: Date to check if person was likely alive (YYYY-MM-DD)
        has_common_ancestor_with: Gramps ID to find people sharing an ancestor (e.g., "I0044")
        degrees_of_separation_from: Gramps ID of person to find relatives connected to (e.g., "I0044")
        degrees_of_separation: Maximum relationship path length (default: 2). Each parent-child
            or spousal connection counts as 1. Examples: sibling=2, grandparent=2, uncle=3,
            first cousin=4, brother-in-law=2
        combine_filters: How to combine multiple filters: "and" (default) or "or"
        max_results: Maximum results to return (default: 50, max: 100)
        show_relation_with: Gramps ID of person to show relationships relative to (e.g., "I0044").
            When set, each result will include the relationship to this anchor person.

    Returns:
        Formatted list of people matching the filter criteria.

    Examples:
        - Find people with surname Smith: surname="Smith"
        - Find people born before 1900: birth_year_before="1900"
        - Find people born between 1850-1900: birth_year_after="1850", birth_year_before="1900"
        - Find who was alive in 1880: probably_alive_on_date="1880-01-01"
        - Find cousins: has_common_ancestor_with="I0044"
        - Find someone's parents (with labels): ancestor_of="I0044", ancestor_generations=1, show_relation_with="I0044"
        - Find someone's grandfathers (with labels): ancestor_of="I0044", ancestor_generations=2, is_male=True, show_relation_with="I0044"
        - Find siblings (with labels): degrees_of_separation_from="I0044", degrees_of_separation=2, show_relation_with="I0044"
        - Find extended family (uncles, aunts): degrees_of_separation_from="I0044", degrees_of_separation=3
    """
    logger = get_logger()

    # Default show_relation_with to User Person or Home Person if not provided
    if not show_relation_with:
        if ctx.deps.user_person_gramps_id:
            show_relation_with = ctx.deps.user_person_gramps_id
        elif ctx.deps.home_person_gramps_id:
            show_relation_with = ctx.deps.home_person_gramps_id

    max_results = min(max(1, max_results), 100)

    rules: list[dict[str, Any]] = []

    if given_name or surname:
        rules.append(
            {
                "name": "HasNameOf",
                "values": [given_name, surname, "", "", "", "", "", "", "", "", ""],
            }
        )

    if birth_year_before or birth_year_after or birth_place:
        date_expr = _build_date_expression(birth_year_before, birth_year_after)
        rules.append({"name": "HasBirth", "values": [date_expr, birth_place, ""]})

    if death_year_before or death_year_after or death_place:
        date_expr = _build_date_expression(death_year_before, death_year_after)
        rules.append({"name": "HasDeath", "values": [date_expr, death_place, ""]})

    if ancestor_of:
        rules.append(
            {
                "name": "IsLessThanNthGenerationAncestorOf",
                "values": [ancestor_of, str(ancestor_generations + 1)],
            }
        )

    if descendant_of:
        rules.append(
            {
                "name": "IsLessThanNthGenerationDescendantOf",
                "values": [descendant_of, str(descendant_generations + 1)],
            }
        )

    if has_common_ancestor_with:
        rules.append(
            {"name": "HasCommonAncestorWith", "values": [has_common_ancestor_with]}
        )

    if degrees_of_separation_from:
        # Check if DegreesOfSeparation filter is available (from FilterRules addon)
        from ..resources.filters import get_rule_list

        available_rules = [rule.__name__ for rule in get_rule_list("Person")]  # type: ignore
        if "DegreesOfSeparation" in available_rules:
            rules.append(
                {
                    "name": "DegreesOfSeparation",
                    "values": [degrees_of_separation_from, str(degrees_of_separation)],
                }
            )
        else:
            logger.warning(
                "DegreesOfSeparation filter not available. "
                "Install FilterRules addon to use this feature."
            )
            return (
                "DegreesOfSeparation filter is not available. "
                "The FilterRules addon must be installed to use this feature."
            )

    if is_male:
        rules.append({"name": "IsMale", "values": []})

    if is_female:
        rules.append({"name": "IsFemale", "values": []})

    if probably_alive_on_date:
        rules.append({"name": "ProbablyAlive", "values": [probably_alive_on_date]})

    if not rules:
        return (
            "No filter criteria provided. Please specify at least one filter parameter."
        )

    if combine_filters.lower() == "or":
        # For OR logic, we need to update the filter_dict in _apply_gramps_filter
        # Pass it as part of the rules structure
        filter_dict: dict[str, Any] = {"rules": rules, "function": "or"}
        filter_rules = json.dumps(filter_dict)
        logger.debug("Built filter rules: %s", filter_rules)

        db_handle = None
        try:
            db_handle = get_db_outside_request(
                tree=ctx.deps.tree,
                view_private=ctx.deps.include_private,
                readonly=True,
                user_id=ctx.deps.user_id,
            )

            args = {"rules": filter_rules}
            try:
                matching_handles = apply_filter(
                    db_handle=db_handle,
                    args=args,
                    namespace="Person",
                    handles=None,
                )
            except Exception as filter_error:
                logger.error(
                    "Filter validation failed: %s. Filter rules: %r",
                    filter_error,
                    filter_rules,
                )
                db_handle.close()
                raise

            if not matching_handles:
                db_handle.close()
                return "No people found matching the filter criteria."

            matching_handles = matching_handles[:max_results]

            context_parts: list[str] = []
            max_length = ctx.deps.max_context_length
            per_item_max = 10000  # Maximum chars per individual item
            current_length = 0

            # Get the anchor person for relationship calculation if requested
            anchor_person = None
            if show_relation_with:
                try:
                    anchor_person = db_handle.get_person_from_gramps_id(
                        show_relation_with
                    )
                    if not anchor_person:
                        logger.warning(
                            "Anchor person %s not found for relationship calculation",
                            show_relation_with,
                        )
                except Exception as e:  # pylint: disable=broad-except
                    logger.warning(
                        "Error fetching anchor person %s: %s", show_relation_with, e
                    )

            for handle in matching_handles:
                try:
                    person = db_handle.get_person_from_handle(handle)

                    if not ctx.deps.include_private and person.private:
                        continue

                    obj_dict = obj_strings_from_object(
                        db_handle=db_handle,
                        class_name="Person",
                        obj=person,
                        semantic=True,
                    )

                    if obj_dict:
                        content = (
                            obj_dict["string_all"]
                            if ctx.deps.include_private
                            else obj_dict["string_public"]
                        )

                        # Add relationship prefix if anchor person is set
                        if anchor_person:
                            rel_prefix = _get_relationship_prefix(
                                db_handle, anchor_person, person, logger
                            )
                            content = rel_prefix + content

                        # Truncate individual items if they're too long
                        if len(content) > per_item_max:
                            content = (
                                content[:per_item_max]
                                + "\n\n[Content truncated due to length...]"
                            )
                            logger.debug(
                                "Truncated Person content from %d to %d chars",
                                len(content) - per_item_max,
                                per_item_max,
                            )

                        if current_length + len(content) > max_length:
                            logger.debug(
                                "Reached max context length (%d chars), stopping at %d results",
                                max_length,
                                len(context_parts),
                            )
                            break

                        context_parts.append(content)
                        current_length += len(content) + 2

                except Exception as e:  # pylint: disable=broad-except
                    logger.warning("Error processing person %s: %s", handle, e)
                    continue

            if not context_parts:
                db_handle.close()
                return "No people found matching the filter criteria (or all results are private)."

            result = "\n\n".join(context_parts)

            total_matches = len(matching_handles)
            returned_count = len(context_parts)

            if returned_count < total_matches:
                result += f"\n\n---\nShowing {returned_count} of {total_matches} matching people. Use max_results parameter to see more."
            elif total_matches == max_results:
                result += f"\n\n---\nShowing {returned_count} people (limit reached). There may be more matches."

            logger.debug(
                "Tool filter_people returned %d results (%d chars)",
                len(context_parts),
                len(result),
            )

            db_handle.close()

            return result

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error filtering people: %s", e)
            if db_handle is not None:
                try:
                    db_handle.close()
                except Exception:  # pylint: disable=broad-except
                    pass
            return f"Error filtering people: {str(e)}"

    # Use the common filter helper for AND logic
    return _apply_gramps_filter(
        ctx=ctx,
        namespace="Person",
        rules=rules,
        max_results=max_results,
        empty_message="No people found matching the filter criteria.",
        show_relation_with=show_relation_with,
    )


@log_tool_call
def filter_events(
    ctx: RunContext[AgentDeps],
    event_type: str = "",
    date_before: str = "",
    date_after: str = "",
    place: str = "",
    description_contains: str = "",
    participant_id: str = "",
    participant_role: str = "",
    max_results: int = 50,
) -> str:
    """Filter events in the genealogy database.

    Use this tool to find events matching specific criteria. Events are occurrences in
    people's lives (births, deaths, marriages, etc.) or general historical events.

    Args:
        event_type: Type of event (e.g., "Birth", "Death", "Marriage", "Baptism",
            "Census", "Emigration", "Burial", "Occupation", "Residence")
        date_before: Latest year to include (inclusive). For "between 1892 and 1900", use "1900".
            Use only the year as a string.
        date_after: Earliest year to include (inclusive). For "between 1892 and 1900", use "1892".
            Use only the year as a string.
        place: Location name to search for (e.g., "Boston", "Massachusetts")
        description_contains: Text that should appear in the event description
        participant_id: Gramps ID of a person who participated in the event (e.g., "I0001")
        participant_role: Role of the participant if participant_id is provided
            (e.g., "Primary", "Family")
        max_results: Maximum number of results to return (1-100, default 50)

    Returns:
        A formatted string containing matching events with their details, or an error message.

    Examples:
        - "births in 1850": filter_events(event_type="Birth", date_after="1850", date_before="1850")
        - "marriages in Boston": filter_events(event_type="Marriage", place="Boston")
        - "events between 1892 and 1900": filter_events(date_after="1892", date_before="1900")
        - "events after 1850": filter_events(date_after="1850")
        - "events before 1900": filter_events(date_before="1900")
        - "events for person I0044": filter_events(participant_id="I0044")
    """
    max_results = min(max(1, max_results), 100)

    rules: list[dict[str, Any]] = []

    if event_type or date_before or date_after or place or description_contains:
        date_expr = _build_date_expression(before=date_before, after=date_after)
        rules.append(
            {
                "name": "HasData",
                "values": [
                    event_type or "",
                    date_expr,
                    place or "",
                    description_contains or "",
                ],
            }
        )

    if participant_id:
        person_filter_rules = [{"name": "HasIdOf", "values": [participant_id]}]
        person_filter_json = json.dumps({"rules": person_filter_rules})

        rules.append(
            {
                "name": "MatchesPersonFilter",
                "values": [person_filter_json, "1" if participant_role else "0"],
            }
        )

    if not rules:
        return (
            "No filter criteria provided. Please specify at least one filter parameter "
            "(event_type, date_before, date_after, place, description_contains, or participant_id)."
        )

    # Use the common filter helper
    return _apply_gramps_filter(
        ctx=ctx,
        namespace="Event",
        rules=rules,
        max_results=max_results,
        empty_message="No events found matching the filter criteria.",
    )


# ── Historical Enslavement Research Tools ────────────────────────────────────

# Keywords suggesting an enslaved person (search events, notes, attributes)
_ENSLAVED_KEYWORDS: frozenset = frozenset([
    "enslaved", "slave", "freedman", "freedwoman", "free person of color",
    "fpc", "manumit", "manumission", "emancipat", "bondage", "bondsman",
    "bondswoman", "chattel", "sold to", "purchased by", "listed in inventory",
    "property of", "estate of", "tax list", "formerly enslaved",
    "formerly a slave", "free black", "freedom papers", "liberation",
])

# Keywords suggesting an enslaver (search notes of enslaved ancestors)
_ENSLAVER_KEYWORDS: frozenset = frozenset([
    "enslaver", "slaveholder", "slave owner", "slaveowner", "planter",
    "plantation owner", "held enslaved", "owned enslaved", "owned slaves",
    "owned slave", "probate inventory", "estate inventory", "slave schedule",
    "overseer", "listed enslaved", "his slaves", "her slaves", "their slaves",
    "negroes", "list of negroes", "master was", "mistress was",
    "belong to", "belonged to",
])


def _check_person_for_enslavement_evidence(
    person,
    db_handle,
    include_private: bool,
) -> list[str]:
    """Return evidence strings if a person has enslaved-status indicators.

    Checks events (type and description), notes, and attributes for
    keywords associated with enslaved status.
    """
    evidence: list[str] = []

    # ── Events ──
    for event_ref in person.get_event_ref_list():
        try:
            event = db_handle.get_event_from_handle(event_ref.ref)
            if not event:
                continue
            if not include_private and event.private:
                continue

            type_str = str(event.get_type()).lower()
            desc_str = event.get_description().lower()

            for kw in _ENSLAVED_KEYWORDS:
                if kw in type_str:
                    yr = event.get_date_object().get_year()
                    year_str = f" ({yr})" if yr else ""
                    evidence.append(f"Event[{event.get_type()}{year_str}]")
                    break

            # Only check description if type didn't already match
            if not any("Event[" in e for e in evidence):
                for kw in _ENSLAVED_KEYWORDS:
                    if kw in desc_str:
                        evidence.append(
                            f"Event desc: '{event.get_description()[:100]}'"
                        )
                        break
        except Exception:  # pylint: disable=broad-except
            continue

    # ── Notes ──
    for note_handle in person.get_note_list():
        try:
            note = db_handle.get_note_from_handle(note_handle)
            if not note:
                continue
            if not include_private and note.private:
                continue
            note_text = note.get().lower()
            for kw in _ENSLAVED_KEYWORDS:
                if kw in note_text:
                    evidence.append(f"Note mentions '{kw}'")
                    break
        except Exception:  # pylint: disable=broad-except
            continue

    # ── Attributes ──
    for attr in person.get_attribute_list():
        try:
            if not include_private and attr.private:
                continue
            combined = (str(attr.get_type()) + " " + attr.get_value()).lower()
            for kw in _ENSLAVED_KEYWORDS:
                if kw in combined:
                    evidence.append(
                        f"Attribute: {attr.get_type()} = {attr.get_value()}"
                    )
                    break
        except Exception:  # pylint: disable=broad-except
            continue

    return evidence


def _extract_enslaver_references_from_person(
    person,
    db_handle,
    include_private: bool,
) -> list[str]:
    """Return enslaver-reference snippets from a person's notes and events.

    Scans notes and event descriptions for mentions of enslavers, planters,
    slave owners, etc.  Returns short contextual excerpts (≤200 chars).
    """
    refs: list[str] = []

    # ── Notes ──
    for note_handle in person.get_note_list():
        try:
            note = db_handle.get_note_from_handle(note_handle)
            if not note:
                continue
            if not include_private and note.private:
                continue
            raw = note.get()
            lower = raw.lower()
            for kw in _ENSLAVER_KEYWORDS:
                idx = lower.find(kw)
                if idx != -1:
                    start = max(0, idx - 60)
                    end = min(len(raw), idx + 140)
                    snippet = raw[start:end].strip().replace("\n", " ")
                    refs.append(f"Note: '…{snippet}…'")
                    break  # one snippet per note
        except Exception:  # pylint: disable=broad-except
            continue

    # ── Event descriptions ──
    for event_ref in person.get_event_ref_list():
        try:
            event = db_handle.get_event_from_handle(event_ref.ref)
            if not event:
                continue
            if not include_private and event.private:
                continue
            desc = event.get_description()
            for kw in _ENSLAVER_KEYWORDS:
                if kw in desc.lower():
                    refs.append(f"Event desc: '{desc[:120]}'")
                    break
        except Exception:  # pylint: disable=broad-except
            continue

    return refs


@log_tool_call
def get_enslaved_ancestors(
    ctx: RunContext[AgentDeps],
    person_id: str = "",
    max_generations: int = 8,
    max_results: int = 50,
) -> str:
    """Find ancestors with documentary evidence of being enslaved.

    Walks the ancestor chain of the specified person (or the Home Person by
    default) and checks each ancestor's events, notes, and attributes for
    enslaved-status indicators: event types such as "Slave" / "Freedman" /
    "Manumission", event descriptions, notes referencing enslavement, and
    attributes marking enslaved status.

    Use this tool when the user asks about enslaved ancestors, which ancestors
    were enslaved, ancestry connected to plantation slavery, freedmen/freedwomen
    lineage, or similar slavery-era history questions.

    Args:
        person_id: Gramps ID of the anchor person whose ancestors to search
                   (e.g., "I0001"). Defaults to the Home Person.
        max_generations: Generations back to search (default: 8, max: 15).
        max_results: Maximum results to return (default: 50, max: 100).

    Returns:
        Formatted text listing ancestors with enslaved-status evidence,
        including relationship labels and the evidence type found.
    """
    logger = get_logger()

    # Resolve anchor person
    if not person_id:
        person_id = (
            ctx.deps.home_person_gramps_id or ctx.deps.user_person_gramps_id
        )
    if not person_id:
        return (
            "No anchor person specified and no Home Person is configured. "
            "Please provide a Gramps ID (e.g., person_id='I0001')."
        )

    max_generations = min(max(1, max_generations), 15)
    max_results = min(max(1, max_results), 100)

    db_handle = None
    try:
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )

        # Verify anchor person exists
        anchor = db_handle.get_person_from_gramps_id(person_id)
        if not anchor:
            db_handle.close()
            return f"Person {person_id} not found in the database."

        # Get all ancestors using Gramps filter
        ancestor_rules = json.dumps({
            "rules": [{
                "name": "IsLessThanNthGenerationAncestorOf",
                "values": [person_id, str(max_generations + 1)],
            }]
        })
        ancestor_handles = apply_filter(
            db_handle=db_handle,
            args={"rules": ancestor_rules},
            namespace="Person",
            handles=None,
        )

        if not ancestor_handles:
            db_handle.close()
            return (
                f"No ancestors found for {person_id} within "
                f"{max_generations} generations."
            )

        logger.debug(
            "get_enslaved_ancestors: checking %d ancestors of %s",
            len(ancestor_handles),
            person_id,
        )

        results: list[str] = []

        for handle in ancestor_handles:
            if len(results) >= max_results:
                break
            try:
                person = db_handle.get_person_from_handle(handle)
                if not person:
                    continue
                if not ctx.deps.include_private and person.private:
                    continue

                evidence = _check_person_for_enslavement_evidence(
                    person, db_handle, ctx.deps.include_private
                )
                if not evidence:
                    continue

                # Get full person record text
                obj_dict = obj_strings_from_object(
                    db_handle=db_handle,
                    class_name="Person",
                    obj=person,
                    semantic=True,
                )
                content = ""
                if obj_dict:
                    content = (
                        obj_dict["string_all"]
                        if ctx.deps.include_private
                        else obj_dict["string_public"]
                    )

                # Get relationship label relative to anchor
                rel_prefix = ""
                try:
                    rel_prefix = _get_relationship_prefix(
                        db_handle, anchor, person, logger
                    )
                except Exception:  # pylint: disable=broad-except
                    pass

                evidence_str = " | ".join(evidence[:5])
                entry = (
                    f"{rel_prefix}EVIDENCE: {evidence_str}\n"
                    f"{content[:2000] if content else person.gramps_id}"
                )
                results.append(entry)

            except Exception as exc:  # pylint: disable=broad-except
                logger.debug(
                    "Error checking ancestor %s for enslavement: %s", handle, exc
                )
                continue

        db_handle.close()

        if not results:
            return (
                f"No ancestors of {person_id} within {max_generations} generations "
                "have documentary evidence of being enslaved in the current database. "
                "This does not mean they were not enslaved — records may be missing. "
                "Use hybrid_search or search_genealogy_database to search notes "
                "for plantation, enslavement, or slavery keywords."
            )

        header = (
            f"Found {len(results)} ancestor(s) of {person_id} with "
            f"enslaved-status evidence (searched {len(ancestor_handles)} ancestors "
            f"across {max_generations} generations):\n\n"
        )
        return header + "\n\n---\n\n".join(results)

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("get_enslaved_ancestors error: %s", exc)
        if db_handle is not None:
            try:
                db_handle.close()
            except Exception:  # pylint: disable=broad-except
                pass
        return f"Error finding enslaved ancestors: {str(exc)}"


@log_tool_call
def get_enslavers_of_ancestors(
    ctx: RunContext[AgentDeps],
    person_id: str = "",
    max_generations: int = 8,
    max_results: int = 30,
) -> str:
    """Find enslavers mentioned in the records of enslaved ancestors.

    Scans the notes, events, and attributes of enslaved ancestors of the
    specified person for references to enslavers, slave owners, plantation
    owners, probate records, estate inventories, and similar documents.
    Returns contextual excerpts naming or describing the enslaving parties.

    Use this tool when the user asks who enslaved their ancestors, which
    plantation their ancestors lived on, who the slave owner was, or asks
    for enslavers by name.

    Args:
        person_id: Gramps ID of the anchor person whose enslaved ancestors
                   to investigate (e.g., "I0001"). Defaults to the Home Person.
        max_generations: Generations back to search (default: 8, max: 15).
        max_results: Maximum enslaved ancestors to examine (default: 30, max: 100).

    Returns:
        Formatted text with enslaver references found in the records, grouped
        by the enslaved ancestor they relate to.
    """
    logger = get_logger()

    # Resolve anchor person
    if not person_id:
        person_id = (
            ctx.deps.home_person_gramps_id or ctx.deps.user_person_gramps_id
        )
    if not person_id:
        return (
            "No anchor person specified and no Home Person is configured. "
            "Please provide a Gramps ID (e.g., person_id='I0001')."
        )

    max_generations = min(max(1, max_generations), 15)
    max_results = min(max(1, max_results), 100)

    db_handle = None
    try:
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )

        anchor = db_handle.get_person_from_gramps_id(person_id)
        if not anchor:
            db_handle.close()
            return f"Person {person_id} not found in the database."

        # Get all ancestors
        ancestor_rules = json.dumps({
            "rules": [{
                "name": "IsLessThanNthGenerationAncestorOf",
                "values": [person_id, str(max_generations + 1)],
            }]
        })
        ancestor_handles = apply_filter(
            db_handle=db_handle,
            args={"rules": ancestor_rules},
            namespace="Person",
            handles=None,
        )

        if not ancestor_handles:
            db_handle.close()
            return f"No ancestors found for {person_id} within {max_generations} generations."

        results: list[str] = []
        enslaved_checked = 0

        for handle in ancestor_handles:
            if enslaved_checked >= max_results:
                break
            try:
                person = db_handle.get_person_from_handle(handle)
                if not person:
                    continue
                if not ctx.deps.include_private and person.private:
                    continue

                # Only scan people who have enslaved evidence
                enslavement_evidence = _check_person_for_enslavement_evidence(
                    person, db_handle, ctx.deps.include_private
                )
                if not enslavement_evidence:
                    continue

                enslaved_checked += 1
                enslaver_refs = _extract_enslaver_references_from_person(
                    person, db_handle, ctx.deps.include_private
                )
                if not enslaver_refs:
                    continue

                # Get person name for display
                primary = person.get_primary_name()
                name = primary.get_name() if primary else person.gramps_id

                # Get relationship label
                rel_prefix = ""
                try:
                    rel_prefix = _get_relationship_prefix(
                        db_handle, anchor, person, logger
                    )
                except Exception:  # pylint: disable=broad-except
                    pass

                enslaver_str = "\n  ".join(enslaver_refs[:5])
                results.append(
                    f"{rel_prefix}ENSLAVED ANCESTOR: {name} [{person.gramps_id}]\n"
                    f"  ENSLAVER REFERENCES:\n  {enslaver_str}"
                )

            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("Error examining ancestor %s: %s", handle, exc)
                continue

        db_handle.close()

        if not results:
            if enslaved_checked == 0:
                return (
                    f"No ancestors of {person_id} within {max_generations} generations "
                    "have documentary evidence of being enslaved. "
                    "Use get_enslaved_ancestors first to confirm enslaved status, "
                    "or use hybrid_search to look for plantation/enslavement records."
                )
            return (
                f"Found {enslaved_checked} ancestor(s) with enslaved-status evidence, "
                "but none of their records contain explicit enslaver references "
                "(planter names, estate owners, probate records, etc.). "
                "The enslaver's name may be in unlinked sources — try "
                "hybrid_search with queries like 'plantation owner', 'slave schedule', "
                "or 'probate estate'."
            )

        header = (
            f"Found enslaver references in the records of {len(results)} "
            f"enslaved ancestor(s) of {person_id} "
            f"(searched {len(ancestor_handles)} ancestors across "
            f"{max_generations} generations):\n\n"
        )
        return header + "\n\n---\n\n".join(results)

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("get_enslavers_of_ancestors error: %s", exc)
        if db_handle is not None:
            try:
                db_handle.close()
            except Exception:  # pylint: disable=broad-except
                pass
        return f"Error finding enslaver references: {str(exc)}"


# ── Lineage Hybrid GraphRAG Tool ────────────────────────────────────────────


@log_tool_call
def hybrid_search(
    ctx: RunContext[AgentDeps],
    query: str,
    gender: str = "",
    birth_year_before: str = "",
    birth_year_after: str = "",
    place_contains: str = "",
    relationship_type: str = "",
    relationship_target_id: str = "",
    max_hops: int = 6,
    max_results: int = 20,
) -> str:
    """Search using Lineage Hybrid GraphRAG — combines semantic search
    over notes, media, and sources with graph-based relationship reasoning.

    Use this tool for complex questions that combine free-text evidence
    with relationship or kinship constraints. For simple lookups of
    specific people, events, or places, use the filter tools instead.

    Args:
        query: Natural language search query
        gender: Filter by gender ("male", "female", "unknown")
        birth_year_before: Born before this year (e.g., "1900")
        birth_year_after: Born after this year (e.g., "1800")
        place_contains: Place name filter (e.g., "Virginia")
        relationship_type: Relationship constraint ("ancestor", "descendant", "related")
        relationship_target_id: Gramps ID for relationship anchor (e.g., "I0001")
        max_hops: Maximum relationship hops (default: 6)
        max_results: Maximum results to return (default: 20)
    """
    logger = get_logger()

    try:
        from ..lineage.intent_parser import parse_intent_from_args
        from ..lineage.retriever import hybrid_retrieve

        # Default relationship target to User Person or Home Person if not provided
        if relationship_type and not relationship_target_id:
            if ctx.deps.user_person_gramps_id:
                relationship_target_id = ctx.deps.user_person_gramps_id
            elif ctx.deps.home_person_gramps_id:
                relationship_target_id = ctx.deps.home_person_gramps_id

        intent = parse_intent_from_args(
            query=query,
            gender=gender,
            birth_year_before=birth_year_before,
            birth_year_after=birth_year_after,
            place_contains=place_contains,
            relationship_type=relationship_type,
            relationship_target_id=relationship_target_id,
            max_hops=max_hops,
            max_results=min(max_results, 50),
        )

        bundle = hybrid_retrieve(
            query=query,
            tree=ctx.deps.tree,
            user_id=ctx.deps.user_id,
            include_private=ctx.deps.include_private,
            intent=intent,
        )

        result = bundle.to_text(max_length=ctx.deps.max_context_length)
        logger.debug(
            "hybrid_search returned %d results (sources: %s, partial: %s)",
            len(bundle.results),
            bundle.sources_used,
            bundle.partial,
        )
        return result

    except ImportError:
        logger.warning(
            "Hybrid search not available — graphrag dependencies not installed"
        )
        return (
            "Hybrid search is not available. Please use the "
            "search_genealogy_database tool or filter tools instead."
        )
    except Exception as exc:
        logger.error("hybrid_search failed: %s", exc, exc_info=True)
        return (
            "Hybrid search encountered an error. Please try using "
            "search_genealogy_database or filter tools as a fallback."
        )

