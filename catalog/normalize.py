"""Normalisation titre / artiste pour le dédoublonnage et l'appariement."""

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
    """Minuscules, sans accents, sans ponctuation, espaces réduits."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("&", " and ").replace("'", "")
    s = _NONALNUM_RE.sub(" ", s)
    return _SPACES_RE.sub(" ", s).strip()


def norm_title(title: str | None) -> str:
    """Retire les mentions de version : « (Remastered 2011) », « - Live », « feat. X »."""
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
    """Similarité [0, 1] entre deux chaînes repliées (SequenceMatcher)."""
    fa, fb = fold(a), fold(b)
    if not fa or not fb:
        return 0.0
    if fa == fb:
        return 1.0
    return SequenceMatcher(None, fa, fb).ratio()


def is_version_of(title: str | None) -> bool:
    """Vrai si le titre annonce une version alternative (live, remix, karaoke…)."""
    if not title:
        return False
    return bool(re.search(r"\b(live|remix|karaoke|instrumental|acoustic|acoustique|demo|cover|reprise|piano|unplugged)\b",
                          title, re.IGNORECASE))
