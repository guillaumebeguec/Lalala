"""Web search (Tavily or Brave) with disk cache — for usages (ads, series)
and lyrics fragments at query time (no massive scraping).

`SEARCH_PROVIDER=tavily|brave`, `SEARCH_API_KEY`. Without a key: `WebSearchDisabled`.
Result: list of {title, url, snippet}.
"""

from __future__ import annotations

import logging

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.cache import DiskCache
from core.config import get_settings

logger = logging.getLogger(__name__)


class WebSearchDisabled(RuntimeError):
    pass


class WebSearch:
    def __init__(self):
        s = get_settings()
        self.provider = s.search_provider.lower()
        self.key = s.search_api_key
        if not self.key:
            raise WebSearchDisabled("SEARCH_API_KEY missing")
        self._http = httpx.Client(timeout=20)
        self._cache = DiskCache("web_search")
        self.n_calls = 0

    @retry(retry=retry_if_exception_type(httpx.TransportError), stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def _call(self, query: str, k: int) -> list[dict]:
        if self.provider == "tavily":
            r = self._http.post("https://api.tavily.com/search", json={"api_key": self.key, "query": query, "max_results": k, "search_depth": "basic"})
            r.raise_for_status()
            return [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")} for x in r.json().get("results", [])]
        if self.provider == "brave":
            r = self._http.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": k},
                               headers={"X-Subscription-Token": self.key, "Accept": "application/json"})
            r.raise_for_status()
            return [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("description", "")}
                    for x in r.json().get("web", {}).get("results", [])]
        raise WebSearchDisabled(f"unknown provider: {self.provider}")

    def search(self, query: str, k: int = 8) -> list[dict]:
        key = self._cache.key(query, provider=self.provider, k=k)
        hit = self._cache.get(key)
        if hit is not None:
            return hit["v"]
        res = self._call(query, k)
        self.n_calls += 1
        self._cache.put(key, {"v": res})
        return res
