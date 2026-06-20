import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from app.settings import get_settings
from app.storage.models import LinearRagBase

settings = get_settings()

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = LinearRagBase.metadata


def get_url():
    settings = get_settings()
    url = settings.FRUGAL_DATABASE_URL or "postgresql+asyncpg://postgres:gas12kilo@localhost:5432/linear_rag_db"
    return url


def include_object(object, name, type_, reflected, compare_to):
    if type_ == "table" and object.schema != "linear_rag":
        return False
    return True


def run_migrations_offline() -> None:
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        include_object=include_object,
        version_table_schema=settings.DB_SCHEMA,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_schemas=True,
        include_object=include_object,
        version_table_schema=settings.DB_SCHEMA,
    )

    with context.begin_transaction():
        context.execute("CREATE SCHEMA IF NOT EXISTS linear_rag;")
        context.run_migrations()


async def run_async_migrations() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
