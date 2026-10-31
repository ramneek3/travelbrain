"""The TravelBrain specialist team.

1. intake    - turns the free-text request into structured trip details
2. flights   - researches routes, airlines, durations and fares (DuckDuckGo search)
3. hotels    - researches places to stay per city (DuckDuckGo search)
4. weather   - reads live Open-Meteo data and gives travel advice
5. planner   - merges every report into a day-by-day itinerary and budget
"""
import asyncio
import json
import os
import re
from datetime import date
from typing import AsyncIterator, Awaitable, Callable

from openai import AsyncOpenAI

import search
import weather as wx

# Free OpenRouter models. Free models come and go: override with OPENROUTER_MODEL,
# and see https://openrouter.ai/models?max_price=0 for the current list.
MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-super-vl:free")
FALLBACK_MODELS = ["google/gemma-4-31b-it:free", "nvidia/nemotron-3-ultra-550b-a55b:free"]

_client: AsyncOpenAI | None = None


class AgentError(RuntimeError):
    pass


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise AgentError("No OpenRouter API key. Set OPENROUTER_API_KEY and restart the server.")
        _client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, max_retries=3,
                              default_headers={"X-Title": "TravelBrain"})
    return _client


def _today() -> str:
    return date.today().isoformat()


def _clean(text: str) -> str:
    """Drop reasoning tags some open models leave in their output."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def _parse_json(text: str) -> dict:
    text = _clean(text)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


async def _ask(system: str, prompt: str, *, schema=None, max_tokens=4000) -> str | dict:
    """One specialist call. With a schema, returns parsed JSON (retrying once if malformed)."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    extra = {"extra_body": {"models": [MODEL, *FALLBACK_MODELS]}}
    if schema:
        messages[0]["content"] += ("\n\nReply with ONLY a JSON object matching this JSON schema, "
                                   "no prose, no code fences:\n" + json.dumps(schema))
        extra["response_format"] = {"type": "json_schema",
                                    "json_schema": {"name": "result", "strict": True, "schema": schema}}

    for attempt in range(2):
        resp = await client().chat.completions.create(
            model=MODEL, messages=messages, max_tokens=max_tokens, **extra)
        if not resp.choices:
            raise AgentError("The model returned no answer.")
        text = resp.choices[0].message.content or ""
        if not schema:
            return _clean(text)
        try:
            return _parse_json(text)
        except (ValueError, json.JSONDecodeError) as e:
            if attempt:
                raise AgentError(f"The model returned malformed JSON ({e}).")
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"That was not valid JSON ({e}). "
                                                     "Reply again with only the JSON object."}]


# ---------- 1. Intake ----------

TRIP_SCHEMA = {
    "type": "object",
    "properties": {
        "origin": {"type": "string", "description": "Departure city/country"},
        "destination": {"type": "string", "description": "Overall destination, e.g. 'Western Europe'"},
        "cities": {"type": "array", "items": {"type": "string"},
                   "description": "Concrete cities to visit, in a sensible travel order"},
        "start_date": {"type": "string", "description": "YYYY-MM-DD"},
        "end_date": {"type": "string", "description": "YYYY-MM-DD"},
        "days": {"type": "integer"},
        "travelers": {"type": "integer"},
        "budget_level": {"type": "string", "enum": ["budget", "mid-range", "luxury"]},
        "currency": {"type": "string", "description": "ISO code of the traveller's home currency"},
        "interests": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"},
                        "description": "Anything you had to assume because the request did not say"},
    },
    "required": ["origin", "destination", "cities", "start_date", "end_date", "days",
                 "travelers", "budget_level", "currency", "interests", "assumptions"],
    "additionalProperties": False,
}


async def intake(prompt: str) -> dict:
    system = (
        f"You are the intake specialist of a trip-planning team. Today is {_today()}. "
        "Turn the traveller's request into concrete trip details. Pick specific cities that fit "
        "the number of days (roughly 2-4 nights per city) and order them to minimise backtracking. "
        "Resolve vague dates to the next upcoming occurrence (e.g. 'April' -> the next April, "
        "starting mid-month if unspecified). Default to 1 traveller and mid-range if unstated. "
        "Record every such default in `assumptions`."
    )
    return await _ask(system, prompt, schema=TRIP_SCHEMA)


def _trip_brief(trip: dict) -> str:
    return (
        f"Trip: {trip['origin']} -> {trip['destination']} ({', '.join(trip['cities'])}), "
        f"{trip['start_date']} to {trip['end_date']} ({trip['days']} days), "
        f"{trip['travelers']} traveller(s), {trip['budget_level']} budget, "
        f"prices in {trip['currency']}. Interests: {', '.join(trip['interests']) or 'general sightseeing'}."
    )


# ---------- 2-4. Specialists ----------

REPORT_STYLE = (
    "Write a concise markdown report (under 250 words) with short headings and bullet points. "
    "Give price ranges, not single numbers, and say they are estimates. Do not ask questions."
)


def _sources(results: list[dict]) -> list[dict]:
    return [{"title": r["title"], "url": r["url"]} for r in results[:8]]


async def flights_agent(trip: dict) -> dict:
    month = trip["start_date"][:7]
    cities = trip["cities"]
    queries = [
        f"flights {trip['origin']} to {cities[0]} airlines duration price {month}",
        f"cheap flights {trip['origin']} {cities[0]} round trip fare",
    ]
    if len(cities) > 1:
        queries.append(f"train or flight {' to '.join(cities)} travel time cost")
    results = await search.web_search(queries)
    system = (
        f"You are the flights specialist of a trip-planning team. Today is {_today()}. "
        "Using the web search results provided (and general knowledge where they are thin), cover: "
        "the best departure and arrival airports, airlines that fly the route (direct and one-stop), "
        "typical flight duration, a realistic round-trip economy fare range for these dates, and "
        "intercity transport between the cities with rough costs. " + REPORT_STYLE
    )
    prompt = _trip_brief(trip) + "\n\nWEB SEARCH RESULTS\n" + search.format_results(results)
    return {"report": await _ask(system, prompt), "sources": _sources(results)}


async def hotels_agent(trip: dict) -> dict:
    queries = [f"best area to stay in {c} {trip['budget_level']} hotels price per night"
               for c in trip["cities"]]
    results = await search.web_search(queries, per_query=4)
    system = (
        f"You are the accommodation specialist of a trip-planning team. Today is {_today()}. "
        "Using the web search results provided (and general knowledge where they are thin), "
        "recommend for each city the best neighbourhood to stay in and 2 specific hotels that "
        "match the budget level, with a typical nightly price range. " + REPORT_STYLE
    )
    prompt = _trip_brief(trip) + "\n\nWEB SEARCH RESULTS\n" + search.format_results(results)
    return {"report": await _ask(system, prompt), "sources": _sources(results)}


async def weather_agent(trip: dict) -> dict:
    try:
        start = date.fromisoformat(trip["start_date"])
        end = date.fromisoformat(trip["end_date"])
    except ValueError:
        start = end = None
    data = await asyncio.gather(*(wx.city_weather(c, start, end) for c in trip["cities"]))
    system = (
        "You are the weather specialist of a trip-planning team. You are given live data from "
        "Open-Meteo: current conditions, and either a forecast or last year's observed weather for "
        "the trip dates. Summarise what to expect in each city, then give practical advice "
        "(what to pack, which days suit outdoor plans). " + REPORT_STYLE
    )
    prompt = _trip_brief(trip) + "\n\nWeather data:\n" + json.dumps(data, indent=1)
    return {"report": await _ask(system, prompt), "data": data,
            "sources": [{"title": "Open-Meteo", "url": "https://open-meteo.com"}]}


# ---------- 5. Planner ----------

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string", "description": "2-3 sentence overview"},
        "itinerary": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "day": {"type": "integer"},
                "date": {"type": "string"},
                "city": {"type": "string"},
                "title": {"type": "string"},
                "morning": {"type": "string"},
                "afternoon": {"type": "string"},
                "evening": {"type": "string"},
                "tip": {"type": "string"},
            },
            "required": ["day", "date", "city", "title", "morning", "afternoon", "evening", "tip"],
            "additionalProperties": False,
        }},
        "budget": {
            "type": "object",
            "properties": {
                "currency": {"type": "string"},
                "items": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "low": {"type": "number"},
                        "high": {"type": "number"},
                        "note": {"type": "string"},
                    },
                    "required": ["category", "low", "high", "note"],
                    "additionalProperties": False,
                }},
                "total_low": {"type": "number"},
                "total_high": {"type": "number"},
            },
            "required": ["currency", "items", "total_low", "total_high"],
            "additionalProperties": False,
        },
        "packing": {"type": "array", "items": {"type": "string"}},
        "booking_checklist": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "summary", "itinerary", "budget", "packing", "booking_checklist"],
    "additionalProperties": False,
}


async def planner(trip: dict, reports: dict) -> dict:
    system = (
        "You are the lead planner of a trip-planning team. Combine your specialists' findings "
        "into one plan the traveller can act on. The itinerary must have exactly one entry per "
        "day, account for travel days between cities, respect the weather advice, and feel "
        "realistic (no more than 2-3 major sights per day). The budget covers the whole group "
        "for the whole trip: flights, intercity transport, accommodation, food, local transport, "
        "activities, and a buffer. Keep budget figures consistent with the specialists' ranges."
    )
    prompt = _trip_brief(trip) + "\n\n" + "\n\n".join(
        f"## {name.title()} specialist report\n{r['report']}" for name, r in reports.items()
    )
    return await _ask(system, prompt, schema=PLAN_SCHEMA, max_tokens=12000)


# ---------- Orchestration ----------

SPECIALISTS: dict[str, Callable[[dict], Awaitable[dict]]] = {
    "flights": flights_agent,
    "hotels": hotels_agent,
    "weather": weather_agent,
}


async def plan_trip(prompt: str) -> AsyncIterator[dict]:
    """Yields progress events; the last one is {'type': 'plan', ...}."""
    yield {"type": "status", "agent": "intake", "state": "running"}
    trip = await intake(prompt)
    yield {"type": "trip", "trip": trip}
    yield {"type": "status", "agent": "intake", "state": "done"}

    queue: asyncio.Queue = asyncio.Queue()

    async def run(name, fn):
        await queue.put({"type": "status", "agent": name, "state": "running"})
        try:
            result = await fn(trip)
            await queue.put({"type": "report", "agent": name, **result})
        except Exception as e:  # one failed specialist shouldn't sink the plan
            await queue.put({"type": "report", "agent": name, "error": str(e),
                             "report": f"_The {name} specialist could not finish: {e}_", "sources": []})

    tasks = [asyncio.create_task(run(n, f)) for n, f in SPECIALISTS.items()]
    reports, done = {}, 0
    while done < len(tasks):
        event = await queue.get()
        if event["type"] == "report":
            reports[event["agent"]] = event
            done += 1
        yield event

    yield {"type": "status", "agent": "planner", "state": "running"}
    plan = await planner(trip, reports)
    yield {"type": "status", "agent": "planner", "state": "done"}
    yield {"type": "plan", "trip": trip, "reports": reports, "plan": plan}


async def follow_up(saved: dict, question: str) -> AsyncIterator[str]:
    """Stream an answer to a follow-up question about a saved trip."""
    system = (
        f"You are TravelBrain, a travel planner. Today is {_today()}. The traveller is asking a "
        "follow-up about the trip below. Answer briefly in markdown. If they ask to change the "
        "plan, give the revised days or budget lines in full.\n\n"
        f"TRIP DETAILS\n{json.dumps(saved['request'])}\n\n"
        f"CURRENT PLAN\n{json.dumps(saved['plan'])}\n\n"
        "SPECIALIST REPORTS\n" + "\n\n".join(
            f"## {k}\n{v['report']}" for k, v in saved["reports"].items())
    )
    messages = [{"role": "system", "content": system}]
    messages += [{"role": m["role"], "content": m["content"]} for m in saved["messages"]]
    messages.append({"role": "user", "content": question})
    stream = await client().chat.completions.create(
        model=MODEL, messages=messages, max_tokens=4000, stream=True,
        extra_body={"models": [MODEL, *FALLBACK_MODELS]},
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
