import os
import httpx

BASE_URL = os.environ.get("PORTFOLIO_URL", "http://localhost:3000")
API_KEY = os.environ.get("PORTFOLIO_API_KEY", "")


def get_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    return headers


def get_client() -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, headers=get_headers(), timeout=30.0)
