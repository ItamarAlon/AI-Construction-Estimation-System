from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.db_scripts.db_query_runner import WriteQueryExecutionError, run_write_query
from scripts.receipt_write_query_builder import build_receipt_write_query


def _sample_receipts() -> list[dict[str, Any]]:
    return [
        {
            "receipt_id": str(uuid.uuid4()),
            "merchant": {
                "name": "fresh mart",
                "normalized_name": "fresh_mart",
                "tax_id": None,
                "country_code": "US",
            },
            "category_code": "food",
            "receipt_date": "2026-03-01",
            "currency": "USD",
            "amounts": {
                "subtotal_amount": 19.50,
                "tax_amount": 1.56,
                "tip_amount": 0.0,
                "total_amount": 21.06,
            },
            "payment_method": "credit_card",
            "source": "manual_seed",
            "notes": "Weekly groceries",
            "items": [
                {
                    "line_no": 1,
                    "description": "whole milk",
                    "quantity": 2,
                    "unit_price": 3.25,
                    "tax_rate_percent": 8.0,
                    "line_total": 6.50,
                    "metadata": {},
                },
                {
                    "line_no": 2,
                    "description": "bread loaf",
                    "quantity": 1,
                    "unit_price": 2.80,
                    "tax_rate_percent": 8.0,
                    "line_total": 2.80,
                    "metadata": {},
                },
            ],
            "tags": ["groceries", "home"],
            "raw_payload": {},
        },
        {
            "receipt_id": str(uuid.uuid4()),
            "merchant": {
                "name": "city taxi",
                "normalized_name": "city_taxi",
                "tax_id": None,
                "country_code": "US",
            },
            "category_code": "travel",
            "receipt_date": "2026-03-06",
            "currency": "USD",
            "amounts": {
                "subtotal_amount": 34.00,
                "tax_amount": 0.0,
                "tip_amount": 4.00,
                "total_amount": 38.00,
            },
            "payment_method": "mobile_wallet",
            "source": "manual_seed",
            "notes": "Airport trip",
            "items": [
                {
                    "line_no": 1,
                    "description": "taxi fare",
                    "quantity": 1,
                    "unit_price": 34.00,
                    "tax_rate_percent": 0.0,
                    "line_total": 34.00,
                    "metadata": {"route": "downtown-airport"},
                }
            ],
            "tags": ["transport", "business_trip"],
            "raw_payload": {},
        },
        {
            "receipt_id": str(uuid.uuid4()),
            "merchant": {
                "name": "office depot",
                "normalized_name": "office_depot",
                "tax_id": None,
                "country_code": "US",
            },
            "category_code": "office",
            "receipt_date": "2026-03-11",
            "currency": "USD",
            "amounts": {
                "subtotal_amount": 57.90,
                "tax_amount": 4.63,
                "tip_amount": 0.0,
                "total_amount": 62.53,
            },
            "payment_method": "debit_card",
            "source": "manual_seed",
            "notes": "Office supplies",
            "items": [
                {
                    "line_no": 1,
                    "description": "printer paper a4",
                    "quantity": 3,
                    "unit_price": 8.50,
                    "tax_rate_percent": 8.0,
                    "line_total": 25.50,
                    "metadata": {},
                },
                {
                    "line_no": 2,
                    "description": "pens pack",
                    "quantity": 2,
                    "unit_price": 6.20,
                    "tax_rate_percent": 8.0,
                    "line_total": 12.40,
                    "metadata": {},
                },
            ],
            "tags": ["office", "supplies"],
            "raw_payload": {},
        },
    ]


def main() -> None:
    receipts = _sample_receipts()
    print(f"Seeding {len(receipts)} receipts...")

    success_count = 0
    for index, receipt_json in enumerate(receipts, start=1):
        query, params = build_receipt_write_query(receipt_json)
        try:
            affected_rows = run_write_query(query, params)
            success_count += 1
            print(
                f"[{index}/{len(receipts)}] receipt_id={params['receipt_id']} saved "
                f"(affected_rows={affected_rows})."
            )
        except WriteQueryExecutionError as exc:
            print(
                f"[{index}/{len(receipts)}] receipt_id={params['receipt_id']} failed: {exc}"
            )

    print(f"Done. Inserted {success_count}/{len(receipts)} receipts.")


if __name__ == "__main__":
    main()
