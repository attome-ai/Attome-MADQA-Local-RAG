"""Specialist instructions chosen by question type, on top of the shared rules.

Routing uses simple patterns, so it's instant and repeatable. Anything unclear is a lookup, which
only adds care about copying values, so a wrong route does limited harm.
"""

from __future__ import annotations

import re

YES_NO = re.compile(r"^(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)\b", re.IGNORECASE)
COMPARISON = re.compile(
    r"\b(older|younger|earlier|later|higher|lower|larger|smaller|greater|fewer|longer|shorter|deeper|"
    r"more|less|most|least|highest|lowest|largest|smallest|earliest|latest|first|last)\b.*(:| or |\bbetween\b)"
    r"|\b(compared? (to|with)|whose .* (higher|lower|more|less))\b",
    re.IGNORECASE,
)
COUNT_TOTAL = re.compile(
    r"\b(how many|how much|number of|total|sum of|combined|in total|altogether|average|difference)\b",
    re.IGNORECASE,
)
LIST = re.compile(
    r"^(what|which|who|where) (are|were)\b|\b(names of|list of|titles of|categories|all the|each of|which items|"
    r"in which (years|states|countries)|what (items|schools|states|years))\b|^which \w+s\b",
    re.IGNORECASE,
)

FOCUS = {
    "yes_no": """Focus for this question: it asks yes or no. List each condition it contains and check every one
against the pages. "Are both" is Yes only if all of them hold. Answer only "Yes" or "No".""",
    "comparison": """Focus for this question: it compares items. Find the value for every item named, from the right
page and period, and compare them with care for dates, ages and units. Answer with the name of the winner
exactly as the question gives it, not the values.""",
    "count_total": """Focus for this question: it needs a count or total. Go through the relevant rows and pages one by
one and include every qualifying item, leaving out subtotals and totals that would double count. Check for
a printed total matching the request first. Give one final number with its unit.""",
    "list": """Focus for this question: it asks for a list. Scan every page shown from top to bottom, and don't stop at
the first few. Include every qualifying item once, each in its own entry, in the wording printed.""",
    "lookup": """Focus for this question: it asks for one fact. Find the exact label, row and column, confirm the person,
organization and period match the question, and copy the value character by character.""",
}


def classify_question(question: str) -> str:
    text = question.strip()
    if COMPARISON.search(text):
        return "comparison"
    if YES_NO.search(text):
        return "yes_no"
    if COUNT_TOTAL.search(text):
        return "count_total"
    if LIST.search(text):
        return "list"
    return "lookup"


def specialist_rules(question: str, base: str) -> str:
    return f"{base}\n\n{FOCUS[classify_question(question)]}"
