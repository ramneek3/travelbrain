"""Free web search via DuckDuckGo (no API key)."""
import asyncio

from ddgs import DDGS


def _search(query: str, n: int) -> list[dict]:
    try:
        return [{"title": r["title"], "url": r["href"], "snippet": r["body"]}
                for r in DDGS().text(query, max_results=n)]
    except Exception:  # rate limits / network hiccups: carry on with fewer results
        return []


async def web_search(queries: list[str], per_query: int = 5) -> list[dict]:
    """Run several searches concurrently, de-duplicated by URL."""
    batches = await asyncio.gather(*(asyncio.to_thread(_search, q, per_query) for q in queries))
    seen, results = set(), []
    for batch in batches:
        for r in batch:
            if r["url"] not in seen:
                seen.add(r["url"])
                results.append(r)
    return results


def format_results(results: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {r['title']}\n{r['url']}\n{r['snippet']}"
                       for i, r in enumerate(results, 1)) or "(no search results)"
