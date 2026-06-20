import re

file_path = "app/storage/migrations/versions/a1b2c3d4e5f6_initial_linear_rag_schema.py"
with open(file_path, "r") as f:
    content = f.read()

# Add import
import_stmt = "from app.settings import get_settings\nsettings = get_settings()\nschema_name = settings.DB_SCHEMA\n"
content = content.replace("import sqlalchemy as sa\n", "import sqlalchemy as sa\n" + import_stmt)

# Replace schema='linear_rag' with schema=schema_name
content = re.sub(r"schema='linear_rag'", "schema=schema_name", content)
content = re.sub(r'schema="linear_rag"', "schema=schema_name", content)

# Create schema dynamically
content = content.replace("op.execute('CREATE SCHEMA IF NOT EXISTS linear_rag')", "op.execute(f'CREATE SCHEMA IF NOT EXISTS {schema_name}')")

with open(file_path, "w") as f:
    f.write(content)
