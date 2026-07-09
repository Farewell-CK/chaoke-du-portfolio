import os
import httpx

API_URL = os.environ.get("PORTFOLIO_API_URL", "http://localhost:3000")
API_KEY = os.environ.get("PORTFOLIO_API_KEY", "")


def get_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    return headers


def get_client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=30.0, headers=get_headers())
