import pytest

from core import llm
from core.llm import parse_json_lenient


# ---------------------------------------------------------------- lenient parser (no LLM)

@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('Voici :\n```json\n{"a": [1, 2]}\n```\nmerci', {"a": [1, 2]}),
    ('{"a": 1, "b": [1,2,],}', {"a": 1, "b": [1, 2]}),
    ('bla {"titre": "x {y}", "n": 2} bla', {"titre": "x {y}", "n": 2}),
    ('[{"a":1},{"a":2}]', [{"a": 1}, {"a": 2}]),
    ('{“a”: “b”}', {"a": "b"}),
])
def test_parse_json_lenient(text, expected):
    assert parse_json_lenient(text) == expected


def test_parse_json_lenient_fails():
    with pytest.raises(ValueError):
        parse_json_lenient("pas de json ici")


def test_json_retry_then_success(monkeypatch):
    """First answer invalid, second valid: a single retry."""
    calls = []

    class _Msg:
        def __init__(self, c):
            self.content, self.tool_calls = c, None

    class _Resp:
        def __init__(self, c):
            self.choices = [type("C", (), {"message": _Msg(c)})()]
            self.usage = None

    def fake_call(messages, *a, **k):
        calls.append(messages)
        return _Resp("je ne sais pas" if len(calls) == 1 else '{"decennie": "1980s"}')

    monkeypatch.setattr(llm, "_call", fake_call)
    r = llm.chat([{"role": "user", "content": "x"}],
                 json_schema={"type": "object", "required": ["decennie"], "properties": {"decennie": {"type": "string"}}})
    assert r.json == {"decennie": "1980s"}
    assert len(calls) == 2
    assert calls[1][-1]["role"] == "user" and "JSON" in calls[1][-1]["content"]


def test_json_gives_up_after_retry(monkeypatch):
    class _Msg:
        content, tool_calls = "rien", None

    class _Resp:
        choices = [type("C", (), {"message": _Msg()})()]
        usage = None

    monkeypatch.setattr(llm, "_call", lambda *a, **k: _Resp())
    with pytest.raises(llm.InvalidJSONOutput):
        llm.chat([{"role": "user", "content": "x"}], json_schema={"type": "object"})


# ---------------------------------------------------------------- with a real Llama (Ollama)

@pytest.mark.llm
def test_chat_answers():
    r = llm.chat([{"role": "user", "content": "Réponds par un seul mot : quelle est la capitale de la France ?"}])
    assert "paris" in r.content.lower()


@pytest.mark.llm
def test_json_output():
    schema = {
        "title": "indices",
        "type": "object",
        "required": ["decennie", "langue"],
        "properties": {"decennie": {"type": "string"}, "langue": {"type": "string"}},
    }
    r = llm.chat(
        [{"role": "user", "content": "Un morceau disco chanté en anglais, sorti vers 1978. "
                                      "Donne la décennie (ex. '1970s') et la langue des paroles."}],
        json_schema=schema,
    )
    assert isinstance(r.json, dict)
    assert "197" in r.json["decennie"]


@pytest.mark.llm
def test_fake_tool_call():
    tools = [{
        "type": "function",
        "function": {
            "name": "catalog_search",
            "description": "Cherche un morceau dans le catalogue à partir d'une description textuelle.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "description du morceau"}},
                "required": ["text"],
            },
        },
    }]
    r = llm.chat(
        [{"role": "system", "content": "Tu es un agent qui retrouve des chansons. Utilise l'outil catalog_search "
                                        "pour chercher avant de répondre."},
         {"role": "user", "content": "Je cherche une chanson des années 80 avec un gros solo de saxophone."}],
        tools=tools,
    )
    assert r.tool_calls, f"no tool call, content: {r.content!r}"
    assert r.tool_calls[0].name == "catalog_search"
    assert "text" in r.tool_calls[0].arguments
