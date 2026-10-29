"""Client LLM OpenAI-compatible vers Llama (Ollama en local, ou API hébergée).

Point d'entrée : `chat(messages, tools=None, json_schema=None)`.

- `tools`       : liste d'outils au format OpenAI (`{"type": "function", "function": {...}}`).
                  Le résultat expose `tool_calls` (nom + arguments déjà parsés).
- `json_schema` : schéma JSON attendu. On demande une sortie structurée au serveur
                  (`response_format`), on parse de façon tolérante (blocs ```json,
                  texte autour, virgules finales…) et on **relance une fois** avec
                  le message d'erreur si le JSON est invalide ou ne respecte pas les
                  clés requises du schéma. Le résultat expose `json`.

Le modèle Llama 8B est faible en JSON strict : le parseur tolérant et la relance
sont la parade prévue dans PLAN.md.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from openai import OpenAI, APIConnectionError, APIStatusError, BadRequestError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.config import get_settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """Le serveur LLM n'est pas joignable (Ollama éteint, mauvaise URL)."""


class InvalidJSONOutput(LLMError):
    """Le modèle n'a pas produit de JSON exploitable après relance."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    json: Any = None
    raw: Any = None
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def message(self) -> dict[str, Any]:
        """Le message assistant à réinjecter dans l'historique (format OpenAI)."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                }
                for tc in self.tool_calls
            ]
        return msg


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    s = get_settings()
    return OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key or "none", timeout=120.0, max_retries=0)


def is_available(timeout: float = 2.0) -> bool:
    """Vrai si le serveur LLM répond (utilisé pour ignorer les tests `llm`)."""
    try:
        get_client().with_options(timeout=timeout).models.list()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- JSON tolérant

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _balanced_slice(text: str, open_ch: str, close_ch: str) -> str | None:
    """Extrait la première sous-chaîne équilibrée commençant par `open_ch`."""
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json_lenient(text: str) -> Any:
    """Parse un JSON « sale » : blocs ```json, texte avant/après, virgules finales,
    guillemets typographiques. Lève `ValueError` si rien n'est exploitable."""
    if text is None:
        raise ValueError("texte vide")
    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1).strip())
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        sl = _balanced_slice(text, open_ch, close_ch)
        if sl:
            candidates.append(sl)

    last_err: Exception | None = None
    for cand in candidates:
        if not cand:
            continue
        for variant in (cand, _TRAILING_COMMA_RE.sub(r"\1", cand).replace("“", '"').replace("”", '"')):
            try:
                return json.loads(variant)
            except json.JSONDecodeError as e:
                last_err = e
    raise ValueError(f"JSON invalide : {last_err}")


def _missing_required(obj: Any, schema: dict[str, Any] | None) -> list[str]:
    """Vérification minimale : type objet + clés `required` du premier niveau."""
    if not schema:
        return []
    if schema.get("type") == "object" or "properties" in schema:
        if not isinstance(obj, dict):
            return ["<objet attendu>"]
        return [k for k in schema.get("required", []) if k not in obj]
    if schema.get("type") == "array" and not isinstance(obj, list):
        return ["<liste attendue>"]
    return []


# --------------------------------------------------------------------------- appel

def _to_tool_calls(message: Any) -> list[ToolCall]:
    out: list[ToolCall] = []
    for tc in getattr(message, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        raw_args = fn.arguments
        if isinstance(raw_args, str):
            try:
                args = parse_json_lenient(raw_args) if raw_args.strip() else {}
            except ValueError:
                logger.warning("Arguments d'outil non parsables pour %s : %r", fn.name, raw_args[:200])
                args = {"_raw": raw_args}
        else:
            args = dict(raw_args or {})
        if not isinstance(args, dict):
            args = {"_value": args}
        out.append(ToolCall(id=getattr(tc, "id", "") or f"call_{len(out)}", name=fn.name, arguments=args))
    return out


@retry(
    retry=retry_if_exception_type(APIConnectionError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)
def _create(**kwargs: Any) -> Any:
    return get_client().chat.completions.create(**kwargs)


def _call(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
          response_format: dict[str, Any] | None, temperature: float, max_tokens: int | None,
          model: str | None) -> Any:
    s = get_settings()
    kwargs: dict[str, Any] = {
        "model": model or s.llm_model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if tools:
        kwargs["tools"] = tools
    if response_format:
        kwargs["response_format"] = response_format
    try:
        return _create(**kwargs)
    except BadRequestError as e:
        # Certains serveurs n'acceptent pas `json_schema` : repli sur `json_object`.
        if response_format and response_format.get("type") == "json_schema":
            logger.info("response_format json_schema refusé (%s), repli sur json_object", e)
            kwargs["response_format"] = {"type": "json_object"}
            return _create(**kwargs)
        raise LLMError(str(e)) from e
    except APIConnectionError as e:
        raise LLMUnavailable(f"LLM injoignable sur {s.llm_base_url} : {e}") from e
    except APIStatusError as e:
        raise LLMError(f"Erreur LLM {e.status_code} : {e}") from e


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    json_schema: dict[str, Any] | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    model: str | None = None,
    json_retries: int = 1,
) -> ChatResult:
    """Un tour de conversation.

    - Avec `json_schema`, `result.json` contient l'objet parsé ; une relance est
      faite si le JSON est invalide (`json_retries`, 1 par défaut).
    - Avec `tools`, `result.tool_calls` contient les appels demandés.
    """
    messages = [dict(m) for m in messages]
    response_format: dict[str, Any] | None = None
    if json_schema:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": json_schema.get("title", "output"), "schema": json_schema},
        }
        # Rappel explicite dans le prompt : les petits modèles en ont besoin même avec response_format.
        hint = ("Réponds uniquement avec un objet JSON valide, sans texte autour, respectant ce schéma :\n"
                + json.dumps(json_schema, ensure_ascii=False))
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0]['content']}\n\n{hint}"
        else:
            messages.insert(0, {"role": "system", "content": hint})

    attempt = 0
    while True:
        resp = _call(messages, tools, response_format, temperature, max_tokens, model)
        choice = resp.choices[0]
        msg = choice.message
        content = msg.content or ""
        usage = {}
        if getattr(resp, "usage", None):
            usage = {k: getattr(resp.usage, k, 0) or 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        result = ChatResult(content=content, tool_calls=_to_tool_calls(msg), raw=resp, usage=usage)

        if not json_schema:
            return result

        try:
            parsed = parse_json_lenient(content)
            missing = _missing_required(parsed, json_schema)
            if missing:
                raise ValueError(f"clés manquantes : {missing}")
            result.json = parsed
            return result
        except ValueError as e:
            if attempt >= json_retries:
                raise InvalidJSONOutput(f"{e} ; sortie : {content[:500]!r}") from e
            attempt += 1
            logger.warning("JSON invalide (%s), relance %d/%d", e, attempt, json_retries)
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (f"Ta réponse n'est pas un JSON valide ({e}). Renvoie uniquement l'objet JSON "
                            "corrigé, sans commentaire ni bloc de code."),
            })
