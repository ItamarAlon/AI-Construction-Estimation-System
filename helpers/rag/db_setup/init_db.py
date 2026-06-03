from pathlib import Path
import os

import psycopg
from psycopg import sql
from dotenv import load_dotenv


def load_config() -> dict:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    return {
        "host": os.getenv("PG_HOST", "127.0.0.1"),
        "port": int(os.getenv("PG_PORT", "5435")),
        "user": os.getenv("PG_ADMIN_USER", os.getenv("PG_USER", "receipt_user")),
        "password": os.getenv(
            "PG_ADMIN_PASSWORD", os.getenv("PG_PASSWORD", "receipt_pass")
        ),
        "dbname": os.getenv("PG_DATABASE", "receipt_db"),
    }


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def create_or_update_role(
    cur: psycopg.Cursor,
    role_name: str,
    role_password: str,
) -> None:
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s;", (role_name,))
    role_exists = cur.fetchone() is not None
    statement = (
        sql.SQL("ALTER ROLE {} LOGIN PASSWORD {};")
        if role_exists
        else sql.SQL("CREATE ROLE {} LOGIN PASSWORD {};")
    ).format(sql.Identifier(role_name), sql.Literal(role_password))
    cur.execute(statement)


def apply_permissions(cur: psycopg.Cursor, dbname: str) -> None:
    read_role = os.getenv("PG_READ_USER", "receipt_read")
    read_password = os.getenv("PG_READ_PASSWORD", "receipt_read_pass")
    write_role = os.getenv("PG_WRITE_USER", "receipt_write")
    write_password = os.getenv("PG_WRITE_PASSWORD", "receipt_write_pass")
    embed_role = os.getenv("PG_EMBED_USER", "receipt_embed")
    embed_password = os.getenv("PG_EMBED_PASSWORD", "receipt_embed_pass")

    create_or_update_role(cur, read_role, read_password)
    create_or_update_role(cur, write_role, write_password)
    create_or_update_role(cur, embed_role, embed_password)

    db_ident = quote_ident(dbname)
    read_ident = quote_ident(read_role)
    write_ident = quote_ident(write_role)
    embed_ident = quote_ident(embed_role)

    cur.execute(f"GRANT CONNECT ON DATABASE {db_ident} TO {read_ident};")
    cur.execute(f"GRANT CONNECT ON DATABASE {db_ident} TO {write_ident};")
    cur.execute(f"GRANT CONNECT ON DATABASE {db_ident} TO {embed_ident};")

    cur.execute(f"GRANT USAGE ON SCHEMA public TO {read_ident};")
    cur.execute(f"GRANT USAGE ON SCHEMA public TO {write_ident};")
    cur.execute(f"GRANT USAGE ON SCHEMA public TO {embed_ident};")

    cur.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {read_ident};")
    cur.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {read_ident};"
    )

    cur.execute(
        f"""
        GRANT SELECT, INSERT, UPDATE, DELETE
        ON TABLE merchants, categories, receipts, receipt_items, tags, receipt_tags, receipt_chunks
        TO {write_ident};
        """
    )
    cur.execute(
        f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {write_ident};"
    )
    cur.execute(
        f"""
        ALTER DEFAULT PRIVILEGES IN SCHEMA public
        GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {write_ident};
        """
    )
    cur.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO {write_ident};"
    )

    cur.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {embed_ident};")
    cur.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE schema_embeddings TO {embed_ident};"
    )
    cur.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {embed_ident};"
    )


def main() -> None:
    config = load_config()
    dbname = config["dbname"]
    schema_path = Path(__file__).resolve().parent / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")

    print(f"Applying schema from: {schema_path}")
    with psycopg.connect(**config) as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
            apply_permissions(cur, dbname)
        conn.commit()
    print("Schema applied and permissions configured successfully.")


if __name__ == "__main__":
    main()
