"""Title / artist normalisation for deduplication and matching."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_PAREN_RE = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")
_DASH_SUFFIX_RE = re.compile(
    r"\s+-\s+(.*\b(remaster|remastered|live|version|edit|mix|mono|stereo|demo|acoustic|instrumental|"
    r"karaoke|radio|single|album|extended|original|anniversary|deluxe|bonus|from)\b.*)$", re.IGNORECASE)
_FEAT_RE = re.compile(r"\s+(feat\.?|featuring|ft\.?)\s+.*$", re.IGNORECASE)
_NONALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_SPACES_RE = re.compile(r"\s+")
_LEADING_THE_RE = re.compile(r"^(the|les|le|la|los|las|die|der|das)\s+")


def fold(s: str | None) -> str:
    """Lowercase, no accents, no punctuation, collapsed whitespace."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("&", " and ").replace("'", "")
    s = _NONALNUM_RE.sub(" ", s)
    return _SPACES_RE.sub(" ", s).strip()


def norm_title(title: str | None) -> str:
    """Strips version mentions: "(Remastered 2011)", "- Live", "feat. X"."""
    if not title:
        return ""
    t = _DASH_SUFFIX_RE.sub("", title)
    t = _PAREN_RE.sub("", t)
    t = _FEAT_RE.sub("", t)
    return fold(t) or fold(title)


def norm_artist(artist: str | None) -> str:
    if not artist:
        return ""
    a = _FEAT_RE.sub("", artist)
    a = _PAREN_RE.sub("", a)
    a = fold(a)
    return _LEADING_THE_RE.sub("", a).strip() or a


def dedup_key(title: str | None, artist: str | None) -> str:
    return f"{norm_title(title)}|{norm_artist(artist)}"


def similarity(a: str | None, b: str | None) -> float:
    """Similarity in [0, 1] between two folded strings (SequenceMatcher)."""
    fa, fb = fold(a), fold(b)
    if not fa or not fb:
        return 0.0
    if fa == fb:
        return 1.0
    return SequenceMatcher(None, fa, fb).ratio()


def is_version_of(title: str | None) -> bool:
    """True if the title announces an alternative version (live, remix, karaoke…)."""
    if not title:
        return False
    return bool(re.search(r"\b(live|remix|karaoke|instrumental|acoustic|acoustique|demo|cover|reprise|piano|unplugged)\b",
                          title, re.IGNORECASE))
