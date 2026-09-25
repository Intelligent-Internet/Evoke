#!/usr/bin/env python3
"""Utilities for feeding natural-language text into Evoke raw query APIs."""

from __future__ import annotations

import re


_PLAIN_TERM_RE = re.compile(r"[^\W_]+", re.UNICODE)


def plain_text_to_raw_terms(text: str) -> str:
    """Convert natural-language query text to safe Evoke raw query terms.

    Evoke raw queries also support boolean syntax, phrase syntax, prefixes,
    and unary operators. BEIR query text is plain text, so punctuation such as
    ``(+)-`` or uppercase words such as ``AND`` should not reach the parser as
    syntax. A lower-cased term list keeps BM25 and hybrid retrieval on the same
    conservative plain-text surface.
    """

    terms = _PLAIN_TERM_RE.findall(text.lower())
    if not terms:
        raise ValueError(f'plain query has no searchable terms: {text!r}')
    return ' '.join(terms)
