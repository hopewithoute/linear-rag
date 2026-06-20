from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.settings import get_settings

engine = None
AsyncSessionLocal = None


def init_db():
    global engine, AsyncSessionLocal
    if engine is None:
        settings = get_settings()
        engine = create_async_engine(settings.FRUGAL_DATABASE_URL, echo=False)
        AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db_session() -> AsyncSession:
    if AsyncSessionLocal is None:
        init_db()
    async with AsyncSessionLocal() as session:
        yield session
