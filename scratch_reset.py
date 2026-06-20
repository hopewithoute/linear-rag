import asyncio
from app.storage.db import init_db
import app.storage.db as db
from app.storage.models import Base

async def reset_db():
    init_db()
    async with db.engine.begin() as conn:
        from app.settings import get_settings
        settings = get_settings()
        from sqlalchemy import text
        await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {settings.DB_SCHEMA}"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

asyncio.run(reset_db())
