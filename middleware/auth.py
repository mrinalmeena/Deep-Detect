from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, HTTPException

API_KEY = "mysecretkey"  # you can change later

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip docs
        if request.url.path in ["/docs", "/openapi.json"]:
            return await call_next(request)

        api_key = request.headers.get("x-api-key")

        if api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

        response = await call_next(request)
        return response