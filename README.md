# TravelBrain

Describe a trip in plain words; a team of AI specialists researches it and builds a plan. Runs on free OpenRouter models and free DuckDuckGo search.

| Agent | What it does | Data source |
|---|---|---|
| 🧭 Intake | Turns your sentence into cities, dates, travellers, budget | LLM (JSON output) |
| ✈️ Flights | Airports, airlines, duration, fare range, intercity transport | LLM + DuckDuckGo search |
| 🏨 Hotels | Neighbourhoods and hotels per city, nightly prices | LLM + DuckDuckGo search |
| 🌤️ Weather | Current conditions + forecast (or last year's same dates) | Open-Meteo (free, no key) |
| 🗺️ Planner | Day-by-day itinerary, budget breakdown, packing, checklist | LLM (JSON output) |

Flights, hotels and weather run in parallel. Trips are saved to `travelbrain.db` (SQLite), and you can reopen any trip and ask follow-up questions.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
.venv/bin/uvicorn app:app --reload
```

Open http://localhost:8000.

Get a free key at https://openrouter.ai/keys. The default model is `nvidia/nemotron-3.5-super-vl:free`, with free fallbacks if it's busy. Free models change often; pick another with:

```bash
export OPENROUTER_MODEL=google/gemma-4-31b-it:free
```

Free models have per-minute and daily request limits. Each plan uses about 5 requests.

Prices are web-researched estimates, not live booking quotes. To add real booking data later, swap a specialist's web search for a flights/hotels API (e.g. Amadeus, Duffel) in `agents.py`.
