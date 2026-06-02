from __future__ import annotations

import json
import uuid
from typing import Any


def build_receipt_write_query(parsed_receipt: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Build a single parameterized SQL write query from parser output.

    Returns:
        (sql_query, params)

    Usage:
        query, params = build_receipt_write_query(parsed_json)
        cursor.execute(query, params)
    """
    if not isinstance(parsed_receipt, dict):
        raise ValueError("parsed_receipt must be a dictionary.")

    merchant = parsed_receipt.get("merchant") or {}
    amounts = parsed_receipt.get("amounts") or {}
    items = parsed_receipt.get("items") or []
    tags = parsed_receipt.get("tags") or []

    receipt_date = parsed_receipt.get("receipt_date")
    total_amount = amounts.get("total_amount")
    if not receipt_date:
        raise ValueError("receipt_date is required to insert into receipts.")
    if total_amount is None:
        raise ValueError("amounts.total_amount is required to insert into receipts.")

    receipt_id = parsed_receipt.get("receipt_id")
    if not isinstance(receipt_id, str) or not receipt_id.strip():
        receipt_id = str(uuid.uuid4())

    # Persist the full parser JSON in raw_payload for traceability/debugging.
    full_payload = dict(parsed_receipt)
    full_payload["receipt_id"] = receipt_id

    params: dict[str, Any] = {
        "receipt_id": receipt_id,
        "merchant_name": merchant.get("name") or "Unknown",
        "merchant_normalized_name": merchant.get("normalized_name"),
        "merchant_tax_id": merchant.get("tax_id"),
        "merchant_country_code": merchant.get("country_code"),
        "category_code": parsed_receipt.get("category_code"),
        "receipt_date": receipt_date,
        "currency": parsed_receipt.get("currency") or "USD",
        "subtotal_amount": amounts.get("subtotal_amount"),
        "tax_amount": amounts.get("tax_amount"),
        "tip_amount": amounts.get("tip_amount"),
        "total_amount": total_amount,
        "payment_method": parsed_receipt.get("payment_method"),
        "source": parsed_receipt.get("source"),
        "notes": parsed_receipt.get("notes"),
        "raw_payload_json": json.dumps(full_payload, ensure_ascii=True, default=str),
        "items_json": json.dumps(items, ensure_ascii=True),
        "tags_json": json.dumps(tags, ensure_ascii=True),
    }

    query = """
WITH input AS (
    SELECT
        %(receipt_id)s::uuid AS receipt_id,
        %(merchant_name)s::text AS merchant_name,
        %(merchant_normalized_name)s::text AS merchant_normalized_name,
        %(merchant_tax_id)s::text AS merchant_tax_id,
        %(merchant_country_code)s::char(2) AS merchant_country_code,
        %(category_code)s::text AS category_code,
        %(receipt_date)s::date AS receipt_date,
        %(currency)s::char(3) AS currency,
        %(subtotal_amount)s::numeric AS subtotal_amount,
        %(tax_amount)s::numeric AS tax_amount,
        %(tip_amount)s::numeric AS tip_amount,
        %(total_amount)s::numeric AS total_amount,
        %(payment_method)s::text AS payment_method,
        %(source)s::text AS source,
        %(notes)s::text AS notes,
        %(raw_payload_json)s::jsonb AS raw_payload_json,
        %(items_json)s::jsonb AS items_json,
        %(tags_json)s::jsonb AS tags_json
),
resolved_merchant AS (
    SELECT m.id
    FROM merchants m
    JOIN input i ON (
        i.merchant_normalized_name IS NOT NULL AND m.normalized_name = i.merchant_normalized_name
    ) OR (
        i.merchant_normalized_name IS NULL AND m.name = i.merchant_name
    )
    ORDER BY m.created_at
    LIMIT 1
),
inserted_merchant AS (
    INSERT INTO merchants (name, normalized_name, tax_id, country_code)
    SELECT
        i.merchant_name,
        i.merchant_normalized_name,
        i.merchant_tax_id,
        i.merchant_country_code
    FROM input i
    WHERE NOT EXISTS (SELECT 1 FROM resolved_merchant)
    RETURNING id
),
selected_merchant AS (
    SELECT id FROM resolved_merchant
    UNION ALL
    SELECT id FROM inserted_merchant
    LIMIT 1
),
selected_category AS (
    SELECT c.id
    FROM categories c
    JOIN input i ON c.code = lower(i.category_code)
    LIMIT 1
),
inserted_receipt AS (
    INSERT INTO receipts (
        id,
        merchant_id,
        category_id,
        receipt_date,
        currency,
        subtotal_amount,
        tax_amount,
        tip_amount,
        total_amount,
        payment_method,
        source,
        notes,
        raw_payload
    )
    SELECT
        i.receipt_id,
        sm.id,
        sc.id,
        i.receipt_date,
        upper(i.currency),
        i.subtotal_amount,
        i.tax_amount,
        i.tip_amount,
        i.total_amount,
        i.payment_method,
        i.source,
        i.notes,
        i.raw_payload_json
    FROM input i
    LEFT JOIN selected_merchant sm ON TRUE
    LEFT JOIN selected_category sc ON TRUE
    RETURNING id
),
parsed_items AS (
    SELECT
        COALESCE((item->>'line_no')::int, row_number() OVER ()) AS line_no,
        COALESCE(NULLIF(item->>'description', ''), 'Unknown item') AS description,
        COALESCE((item->>'quantity')::numeric, 1) AS quantity,
        (item->>'unit_price')::numeric AS unit_price,
        (item->>'tax_rate_percent')::numeric AS tax_rate_percent,
        COALESCE(
            (item->>'line_total')::numeric,
            COALESCE((item->>'quantity')::numeric, 1) * COALESCE((item->>'unit_price')::numeric, 0)
        ) AS line_total,
        COALESCE(item->'metadata', '{}'::jsonb) AS metadata
    FROM input i,
         LATERAL jsonb_array_elements(i.items_json) item
),
inserted_items AS (
    INSERT INTO receipt_items (
        receipt_id,
        line_no,
        description,
        quantity,
        unit_price,
        tax_rate_percent,
        line_total,
        metadata
    )
    SELECT
        ir.id,
        pi.line_no,
        pi.description,
        pi.quantity,
        pi.unit_price,
        pi.tax_rate_percent,
        pi.line_total,
        pi.metadata
    FROM inserted_receipt ir
    JOIN parsed_items pi ON TRUE
    ON CONFLICT (receipt_id, line_no) DO UPDATE
    SET
        description = EXCLUDED.description,
        quantity = EXCLUDED.quantity,
        unit_price = EXCLUDED.unit_price,
        tax_rate_percent = EXCLUDED.tax_rate_percent,
        line_total = EXCLUDED.line_total,
        metadata = EXCLUDED.metadata
    RETURNING 1
),
parsed_tags AS (
    SELECT DISTINCT lower(trim(tag_name)) AS tag_name
    FROM input i,
         LATERAL jsonb_array_elements_text(i.tags_json) tag_name
    WHERE trim(tag_name) <> ''
),
upserted_tags AS (
    INSERT INTO tags (name)
    SELECT pt.tag_name
    FROM parsed_tags pt
    ON CONFLICT (name) DO UPDATE
    SET name = EXCLUDED.name
    RETURNING id
),
inserted_receipt_tags AS (
    INSERT INTO receipt_tags (receipt_id, tag_id)
    SELECT ir.id, ut.id
    FROM inserted_receipt ir
    JOIN upserted_tags ut ON TRUE
    ON CONFLICT DO NOTHING
    RETURNING 1
)
SELECT id AS receipt_id
FROM inserted_receipt;
"""
    return query.strip(), params
