import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# DATABASE_URL comes from the environment, never from alembic.ini — the
# .ini file is committed to git, and a real Postgres connection string
# (with password) must never end up there. Same "fail loudly, no silent
# fallback" reasoning as backend/tenant_db/base.py.
_database_url = os.environ.get("DATABASE_URL", "")
if not _database_url:
    raise RuntimeError(
        "DATABASE_URL is not set — required to run Alembic migrations "
        "against the tenant/user Postgres database."
    )
if _database_url.startswith("postgres://"):
    _database_url = _database_url.replace("postgres://", "postgresql://", 1)
config.set_main_option("sqlalchemy.url", _database_url)

# Import Base.metadata for 'autogenerate' support — and, critically, import
# models too: SQLAlchemy's declarative registry only populates Base.metadata
# with tables from model classes that have actually been imported somewhere.
# Importing only Base (as PR 0 did, when no models existed yet) would make
# autogenerate see an empty schema even with real models defined elsewhere.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.tenant_db.base import Base  # noqa: E402
import backend.tenant_db.models  # noqa: E402,F401 -- import for side effect: registers tables onto Base.metadata

target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
