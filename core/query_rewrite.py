"""🧪 Réécriture de requête par Llama (PLAN.md phase 5).

À partir d'une description floue (FR ou EN), Llama produit en JSON :
- `captions` : 3 à 5 légendes audio en anglais (instruments, tempo, voix, ambiance,
  production) pour les index audio (CLAP ne comprend que l'anglais) ;
- `hints` : indices structurés — décennie, langue des paroles, support (pub, série,
  film, jeu, radio…), genre, fragments de paroles cités par l'utilisateur ;
- `query` : requête texte nettoyée (anglais) pour BM25 et les fiches ;
- `query_fr` : la même en français si la description est en français (fiches FR).
Interdiction d'inventer des titres ou des artistes : seuls ceux cités par
l'utilisateur peuvent apparaître (contrôle programmatique : un nom propre absent
de la description est retiré de `query`). Cache par hash de la description.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from core import llm
from core.cache import DiskCache
from core.config import get_settings

logger = logging.getLogger(__name__)

SCHEMA = {
    "title": "query_rewrite",
    "type": "object",
    "required": ["captions", "hints", "query"],
    "properties": {
        "captions": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 5},
        "hints": {
            "type": "object",
            "properties": {
                "decade": {"type": ["string", "null"], "description": "e.g. '1980s' or null"},
                "language": {"type": ["string", "null"], "description": "language of the lyrics, english name, or 'instrumental', or null"},
                "medium": {"type": ["string", "null"], "description": "ad, tv series, film, video game, radio, club, or null"},
                "genre": {"type": ["string", "null"]},
                "vocals": {"type": ["string", "null"], "description": "male, female, mixed, instrumental or null"},
                "lyrics_fragments": {"type": "array", "items": {"type": "string"}},
                "sound_terms": {"type": "array", "items": {"type": "string"}, "description": "words the user uses about the SOUND itself (whistling, acoustic guitar, saxophone solo, slow tempo…); empty if the description is only about context, lyrics or era"},
                "named_entities": {"type": "array", "items": {"type": "string"}, "description": "artists, titles, brands, shows explicitly mentioned by the user"},
            },
        },
        "query": {"type": "string"},
        "query_fr": {"type": ["string", "null"]},
    },
}

_SYSTEM = (
    "You help a music search engine. The user describes a song they half-remember, in French or English. "
    "Produce JSON only.\n"
    "- captions: 3 to 5 short English captions describing how the song probably SOUNDS (instruments, tempo, "
    "vocals, mood, production era). Write them as audio captions, not as a story.\n"
    "- hints: decade (like '1980s') if the user gives any clue (era, 'when I was a kid in 1995'…) else null; "
    "language of the lyrics if stated or obvious else null; medium (ad, tv series, film, video game, radio…) if "
    "mentioned; genre; vocals; lyrics_fragments = words the user quotes as lyrics, verbatim; "
    "named_entities = artists, titles, brands or shows the user explicitly names; sound_terms = words about the sound itself "
    "(instruments, voice type, tempo, whistling…), empty when the user only talks about context, story or lyrics.\n"
    "- query: a clean English keyword query (genre, era, instruments, themes, quoted lyrics, named entities). "
    "- query_fr: same in French if the description was in French, else null.\n"
    "NEVER invent a song title or an artist name. If you are tempted to guess the song, do not: describe it instead.\n"
    "Hints and captions must stay within what the user wrote: do NOT infer a genre, a decade, a tempo or the singer's "
    "gender from an instrument or a mood. 'guitar' alone says nothing about rock, male vocals or tempo. A one-word "
    "description gets captions that only paraphrase that word, and every hint null except sound_terms."
)

# ---- Ancrage des indices dans le texte de l'utilisateur (indépendant du modèle) -------------------------
# Llama 8B « complète » volontiers : « guitare » → genre rock, voix masculine, tempo rapide. Un indice
# n'est gardé que si le texte en porte la trace (mot du lexique ou valeur citée).
_LEX = {
    "genre": ("rock", "pop", "rap", "hip", "hop", "jazz", "metal", "électro", "electro", "electronic", "techno", "house", "disco",
              "funk", "soul", "reggae", "blues", "country", "folk", "punk", "classique", "classical", "r&b", "rnb", "variété",
              "chanson", "dance", "trance", "indie", "grunge", "ska", "latino", "salsa", "gospel", "opéra", "opera", "swing",
              "k-pop", "kpop", "afro", "zouk", "raï", "reggaeton", "trap", "drill", "ambient", "lo-fi", "lofi", "synthwave",
              "new wave", "hardcore", "emo", "boléro", "tango", "bossa", "samba", "musette", "yéyé", "yeye"),
    "vocals": ("voix", "voice", "vocal", "chanteur", "chanteuse", "singer", "femme", "woman", "female", "homme", "man ",
               "male", "garçon", "fille", "girl", "boy", "chœur", "choir", "chorus", "duo", "duet", "instrumental", "sans parole",
               "no lyrics", "rappeur", "rapper", "crooner", "falsetto", "grave", "aiguë", "aigue", "enfant", "child", "kid"),
    "decade": ("19", "20", "années", "annees", "year", "decade", "décennie", "'s", "enfant", "kid", "child", "jeune", "young",
               "ado", "teen", "vieux", "old", "récent", "recent", "nouveau", "new", "époque", "era", "ancien", "lycée", "collège",
               "école", "school", "université", "college", "mariage", "wedding"),
    "language": ("français", "francais", "french", "anglais", "english", "espagnol", "spanish", "italien", "italian", "allemand",
                 "german", "portugais", "portuguese", "brésil", "brazil", "arabe", "arabic", "japonais", "japanese", "coréen",
                 "korean", "latin", "langue", "language", "paroles", "lyrics", "instrumental", "sans parole", "no words", "chante en",
                 "sings in", "accent"),
    "medium": ("pub", "publicité", "ad", "advert", "commercial", "spot", "série", "series", "show", "film", "movie", "cinéma", "cinema",
               "jeu", "game", "radio", "tv", "télé", "tele", "générique", "theme", "club", "boîte", "disco", "soirée", "party",
               "youtube", "tiktok", "instagram", "stade", "stadium", "match", "mariage", "wedding", "école", "school", "trailer",
               "bande-annonce", "dessin animé", "cartoon", "anime", "jingle", "spectacle", "concert", "festival"),
}


def _grounded(kind: str, value, description: str) -> bool:
    """Vrai si la valeur de l'indice est citée telle quelle, ou si un mot du lexique de ce type d'indice apparaît."""
    if value is None or not str(value).strip():
        return False
    d = " " + description.lower() + " "
    v = str(value).lower()
    if any(w in d for w in re.findall(r"[a-zà-ÿ0-9&'-]{3,}", v)):
        return True
    return any(w in d for w in _LEX.get(kind, ()))


def ground_hints(hints: dict[str, Any], description: str) -> dict[str, Any]:
    """Annule décennie, langue, support, genre et voix qui n'ont aucune trace dans la description."""
    out = dict(hints)
    for k in ("decade", "language", "medium", "genre", "vocals"):
        if out.get(k) and not _grounded(k, out[k], description):
            logger.info("réécriture : indice %s=%r non cité par l'utilisateur, retiré", k, out[k])
            out[k] = None
    return out


_CAPTION_TRAPS = {
    "genre": _LEX["genre"],
    "vocals": ("male", "female", "woman", "man's", "men's", "girl", "boy", "singer", "vocals", "vocal", "chœur", "choir"),
    "tempo": ("upbeat", "fast", "slow", "mid-tempo", "midtempo", "tempo", "bpm", "energetic", "ballad", "driving", "groovy"),
}


def ground_captions(captions: list[str], description: str) -> list[str]:
    """Écarte les légendes qui ajoutent un genre, un type de voix ou un tempo que le texte ne mentionne pas.
    Si tout est écarté, on garde la description telle quelle (le modèle texte→audio la comprend)."""
    d = " " + description.lower() + " "
    kept = []
    for c in captions:
        cl = " " + c.lower() + " "
        bad = False
        for kind, words in _CAPTION_TRAPS.items():
            lex = _LEX.get(kind, words)
            if any((" " + w) in cl for w in words) and not any(w in d for w in lex):
                bad = True
                break
        if bad:
            logger.info("réécriture : légende non ancrée écartée : %r", c)
        else:
            kept.append(c)
    return kept or [description.strip()]


@dataclass
class Rewrite:
    captions: list[str]
    hints: dict[str, Any]
    query: str
    query_fr: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def decade(self) -> str | None:
        d = self.hints.get("decade")
        m = re.search(r"(19|20)\d0", str(d or ""))
        return f"{m.group(0)}s" if m else None

    @property
    def sound_terms(self) -> list[str]:
        return [str(x) for x in (self.hints.get("sound_terms") or []) if str(x).strip()]

    @property
    def describes_sound(self) -> bool:
        """Vrai si l'utilisateur décrit le son lui-même : seule situation où texte→audio a un sens."""
        return len(self.sound_terms) > 0

    @property
    def language(self) -> str | None:
        v = self.hints.get("language")
        return str(v).lower() if v else None


_CAP_WORD = re.compile(r"\b[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)*")


def _strip_unsupported_names(query: str, description: str, allowed: list[str]) -> str:
    """Retire de la requête les groupes de mots capitalisés absents de la description
    (titres/artistes devinés), sauf en début de phrase ou s'ils sont dans `allowed`."""
    desc_l = description.lower()
    allowed_l = [a.lower() for a in allowed]
    out = query
    for m in list(_CAP_WORD.finditer(query)):
        phrase = m.group(0)
        if m.start() == 0:
            continue
        pl = phrase.lower()
        if pl in desc_l or any(pl in a or a in pl for a in allowed_l):
            continue
        if len(phrase) <= 3:
            continue
        logger.info("réécriture : nom propre non cité retiré de la requête : %r", phrase)
        out = out.replace(phrase, "")
    return re.sub(r"\s{2,}", " ", out).strip(" ,;")


def _is_named_in(entity: str, description: str) -> bool:
    """Une entité n'est gardée que si l'utilisateur l'a écrite, avec une majuscule ou un chiffre
    (nom propre, marque, année) : « voiture » n'est pas une entité, « Citroën » ou « 2008 » oui."""
    e = entity.strip()
    if len(e) < 2:
        return False
    m = re.search(re.escape(e), description, flags=re.IGNORECASE)
    if not m:
        return False
    found = m.group(0)
    return any(c.isupper() for c in found) or any(c.isdigit() for c in found)


def rewrite(description: str, model: str | None = None, use_cache: bool = True) -> Rewrite:
    cache = DiskCache("query_rewrite")
    key = cache.key(description, model=model or get_settings().llm_model, v=2)
    hit = cache.get(key) if use_cache else None
    if hit is None:
        r = llm.chat([{"role": "system", "content": _SYSTEM}, {"role": "user", "content": description.strip()}],
                     json_schema=SCHEMA, temperature=0.1, max_tokens=500, model=model)
        hit = r.json
        cache.put(key, hit)
    hints = ground_hints(dict(hit.get("hints") or {}), description)
    allowed = [str(x) for x in (hints.get("named_entities") or []) if _is_named_in(str(x), description)]
    hints["named_entities"] = allowed
    query = _strip_unsupported_names(str(hit.get("query") or ""), description, allowed)
    captions = ground_captions([str(c) for c in (hit.get("captions") or []) if str(c).strip()][:5], description)
    return Rewrite(captions=captions, hints=hints, query=query, query_fr=hit.get("query_fr") or None, raw=hit)


def translate_to_english(text: str, model: str | None = None) -> str:
    """Traduction brute (pour CLAP) ; identité si le texte est déjà en anglais selon Llama."""
    cache = DiskCache("translate_en")
    key = cache.key(text, model=model or get_settings().llm_model, v=1)
    hit = cache.get(key)
    if hit is not None:
        return hit["v"]
    r = llm.chat([{"role": "system", "content": "Translate the user's text to English. If it is already English, return it unchanged. "
                                                 "Return JSON: {\"text\": \"...\"}. Do not add anything."},
                  {"role": "user", "content": text}], json_schema={"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}},
                 temperature=0.0, max_tokens=400, model=model)
    out = str(r.json.get("text") or text)
    cache.put(key, {"v": out})
    return out
