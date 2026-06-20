from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.settings import get_settings

security = HTTPBearer(auto_error=False)


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> None:
    settings = get_settings()
    expected_key = settings.FRUGAL_API_KEY
    if not expected_key:
        return
    if credentials is None or credentials.credentials != expected_key:
        raise HTTPException(status_code=401, detail="Invalid FRUGAL API key")
