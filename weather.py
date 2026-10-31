"""Live weather data from Open-Meteo (free, no API key)."""
from datetime import date, timedelta

import httpx

GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"


async def _geocode(http: httpx.AsyncClient, city: str) -> dict | None:
    r = await http.get(GEOCODE, params={"name": city, "count": 1})
    results = r.json().get("results") or []
    return results[0] if results else None


def _summarise_daily(daily: dict) -> dict:
    hi = [t for t in daily.get("temperature_2m_max", []) if t is not None]
    lo = [t for t in daily.get("temperature_2m_min", []) if t is not None]
    rain = [p for p in daily.get("precipitation_sum", []) if p is not None]
    if not hi:
        return {}
    return {
        "avg_high_c": round(sum(hi) / len(hi), 1),
        "avg_low_c": round(sum(lo) / len(lo), 1),
        "rainy_days": sum(1 for p in rain if p >= 1),
        "total_rain_mm": round(sum(rain), 1),
        "days": len(hi),
    }


async def city_weather(city: str, start: date | None, end: date | None) -> dict:
    """Current conditions plus either a real forecast (trip within 16 days)
    or last year's observed weather for the same dates as a climate guide."""
    async with httpx.AsyncClient(timeout=20) as http:
        place = await _geocode(http, city)
        if not place:
            return {"city": city, "error": "location not found"}
        lat, lon = place["latitude"], place["longitude"]
        out = {"city": place["name"], "country": place.get("country"), "lat": lat, "lon": lon}

        cur = await http.get(FORECAST, params={
            "latitude": lat, "longitude": lon, "timezone": "auto",
            "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
        })
        out["current"] = cur.json().get("current")

        today = date.today()
        start = start or today
        end = end or start + timedelta(days=6)
        if start <= today + timedelta(days=15):
            r = await http.get(FORECAST, params={
                "latitude": lat, "longitude": lon, "timezone": "auto", "daily": DAILY,
                "start_date": max(start, today).isoformat(),
                "end_date": min(end, today + timedelta(days=15)).isoformat(),
            })
            out["trip_period"] = {"kind": "forecast", **_summarise_daily(r.json().get("daily", {}))}
        else:
            ly_start, ly_end = start.replace(year=start.year - 1), end.replace(year=end.year - 1)
            if ly_end >= today:  # dates not yet observed last year; step back one more
                ly_start, ly_end = ly_start.replace(year=ly_start.year - 1), ly_end.replace(year=ly_end.year - 1)
            r = await http.get(ARCHIVE, params={
                "latitude": lat, "longitude": lon, "timezone": "auto", "daily": DAILY,
                "start_date": ly_start.isoformat(), "end_date": ly_end.isoformat(),
            })
            out["trip_period"] = {
                "kind": f"historical ({ly_start.year}, same dates) - use as a climate guide",
                **_summarise_daily(r.json().get("daily", {})),
            }
        return out
