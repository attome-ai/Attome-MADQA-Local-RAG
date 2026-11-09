"""Agent loop: split the question, search each part, read, and search again if evidence is missing.

The model and search are passed in, so the loop can be tested without a GPU. A question that
needs one search and one read behaves like the plain pipeline, so simple questions stay cheap.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .profiling import profiled, timed
from .read import MAX_IMAGES, final_json, render_jpeg

Page = tuple[str, int]

PLAN_PROMPT = """Split this document question into the separate searches needed to answer it.
Use one search if a single page could answer it. Use several when it compares things or combines
facts about different people, organizations, documents or years, with one search for each.
Keep each search specific and keep names, dates and identifiers exactly as written.
Reply with JSON only: {{"searches": ["..."]}}

Question: {question}"""

READ_PROMPT = """{rules}

You are given document pages, each labelled with its file and page number. Answer using only them.
{memory}Keep names apart when they look alike, and keep the direction of each relationship straight.
Record each fact you have confirmed from the pages, with its page labels, and state exactly which
connections are still missing. Don't treat an assumption as a fact.
If something is missing, name the specific person, document or fact to search for next, and don't
repeat an earlier search. List only the page labels your answer actually relies on.
Reply with JSON only:
{{"answer": ["..."], "pages": [1], "confirmed": ["..."], "missing": ["..."], "sufficient": true, "next_search": ""}}

Question: {question}"""


def memory_text(confirmed: list[str], searches: list[str]) -> str:
    """Carry what earlier rounds learned, because each round sees a new set of pages."""
    if not confirmed and len(searches) <= 1:
        return ""
    lines = ["Already confirmed in earlier rounds, which you may rely on:"]
    lines += [f"- {fact}" for fact in confirmed] or ["- nothing yet"]
    lines.append(f"Searches already tried: {json.dumps(searches, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


RETRY_PROMPT = """Searches for this question have not found the answer. Earlier searches:
{searches}

Write one new search that takes a different approach, not a rewording. Use the most distinctive
name or identifier alone, or describe the kind of document that would hold the answer, such as a
form, table, resume or financial statement. Reply with JSON only: {{"search": "..."}}

Question: {question}"""

NOT_FOUND = re.compile(
    r"not (found|stated|provided|mentioned|available|specified|listed)|does not (provide|state|contain|mention|specify)"
    r"|cannot be determined|no (information|answer)|unable to",
    re.IGNORECASE,
)


def looks_unanswered(answer: list[str]) -> bool:
    """True when the reader found nothing usable, which usually means search missed the page."""
    joined = " ".join(answer).strip()
    return not joined or bool(NOT_FOUND.search(joined))


def different_search(question: str, searches: list[str], text_model: Callable[[str], str]) -> str:
    """Ask for a new angle, returning nothing if the reply is unusable or a repeat."""
    prompt = RETRY_PROMPT.format(question=question, searches=json.dumps(searches, ensure_ascii=False))
    try:
        reply = text_model(prompt)
    except OSError:
        return ""
    body = reply.rsplit("</think>", 1)[-1]
    start, end = body.find("{"), body.rfind("}")
    try:
        data = json.loads(body[start : end + 1]) if start >= 0 else {}
    except json.JSONDecodeError:
        return ""
    query = data.get("search") if isinstance(data, dict) else None
    if not isinstance(query, str) or not query.strip() or query.strip() in searches:
        return ""
    return query.strip()


@dataclass
class AgentSettings:
    max_rounds: int = 3
    pages_per_search: int = 3
    window: int = 1
    max_images: int = MAX_IMAGES
    rules: str = ""
    max_side: int | None = None


@dataclass
class AgentResult:
    answer: list[str]
    citations: list[dict]
    iterations: int
    searches: list[str] = field(default_factory=list)


def plan_searches(question: str, text_model: Callable[[str], str]) -> list[str]:
    """Return the planned searches, falling back to the question itself if the plan is unusable."""
    try:
        reply = text_model(PLAN_PROMPT.format(question=question))
    except OSError:
        return [question]
    rest = reply.rsplit("</think>", 1)[-1]
    start = rest.find("{")
    try:
        data = json.loads(rest[start : rest.rfind("}") + 1]) if start >= 0 else {}
    except json.JSONDecodeError:
        data = {}
    searches = data.get("searches") if isinstance(data, dict) else None
    if not isinstance(searches, list):
        return [question]
    cleaned = list(dict.fromkeys(s.strip() for s in searches if isinstance(s, str) and s.strip()))
    return cleaned[:4] or [question]


def select_pages(
    ranked_lists: list[list[Page]], settings: AgentSettings, known: set[Page]
) -> list[Page]:
    """Take the top pages from every search in turn, so no search is crowded out, then neighbours."""
    chosen: list[Page] = []
    for rank in range(settings.pages_per_search):
        for ranked in ranked_lists:
            if rank < len(ranked) and ranked[rank] not in chosen:
                chosen.append(ranked[rank])
    for distance in range(1, settings.window + 1):
        for document, page in list(chosen):
            for near in ((document, page + distance), (document, page - distance)):
                if near in known and near not in chosen:
                    chosen.append(near)
    return chosen[: settings.max_images]


def build_read_content(
    question: str, pages: list[Page], data: Path, settings: AgentSettings, memory: str = ""
) -> list[dict]:
    content: list[dict] = []
    for number, (document, page) in enumerate(pages, 1):
        image = render_jpeg(data / "pdfs" / document, page, settings.max_side)
        content.append({"type": "text", "text": f"Page {number} (file {document}, its page {page}):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
    prompt = READ_PROMPT.format(rules=settings.rules, question=question, memory=memory).strip()
    content.append({"type": "text", "text": prompt})
    return content


def parse_read(text: str, pages: list[Page]) -> tuple[list[str], list[dict], bool, str, list[str]]:
    data, rest = final_json(text)
    if data is None:
        # An unstructured reply is still an answer, but there's no basis to search again.
        return ([rest] if rest else []), [], True, "", []
    answer = data["answer"]
    answer = [str(a) for a in answer] if isinstance(answer, list) else [str(answer)]
    labels = data.get("pages", [])
    labels = labels if isinstance(labels, list) else [labels]
    citations, seen = [], set()
    for label in labels:
        if isinstance(label, int) and 1 <= label <= len(pages) and label not in seen:
            seen.add(label)
            citations.append({"file": pages[label - 1][0], "page": pages[label - 1][1]})
    next_search = data.get("next_search")
    sufficient = data.get("sufficient") is not False
    confirmed = data.get("confirmed", [])
    confirmed = [str(c) for c in confirmed] if isinstance(confirmed, list) else []
    return answer, citations, sufficient, next_search if isinstance(next_search, str) else "", confirmed


@profiled
def run_agent(
    question: str,
    search: Callable[[str], list[Page]],
    text_model: Callable[[str], str],
    vision_model: Callable[[list[dict]], str],
    data: Path,
    known: set[Page],
    settings: AgentSettings,
) -> AgentResult:
    with timed("agent_plan"):
        searches = plan_searches(question, text_model)
    ranked_lists = [search(query) for query in searches]
    best = AgentResult([], [], 0, list(searches))
    confirmed: list[str] = []
    for round_number in range(1, settings.max_rounds + 1):
        pages = select_pages(ranked_lists, settings, known)
        if not pages:
            best.iterations = round_number
            return best
        with timed("agent_read"):
            reply = vision_model(
                build_read_content(question, pages, data, settings, memory_text(confirmed, best.searches))
            )
        answer, citations, sufficient, next_search, facts = parse_read(reply, pages)
        confirmed = list(dict.fromkeys(confirmed + facts))
        best.iterations = round_number
        unanswered = looks_unanswered(answer)
        # A real answer always replaces a "not found", but "not found" never replaces a real answer.
        if answer and (not unanswered or not best.answer):
            best.answer, best.citations = answer, citations
        if round_number == settings.max_rounds:
            return best
        query = next_search.strip()
        if unanswered and (not query or query in best.searches):
            query = different_search(question, best.searches, text_model)
        elif sufficient and not unanswered:
            return best
        if not query or query in best.searches:
            return best
        best.searches.append(query)
        # The newest search goes first, because earlier results already failed to answer this.
        ranked_lists.insert(0, search(query))
    return best
