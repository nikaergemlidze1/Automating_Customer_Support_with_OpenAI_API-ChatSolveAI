"""Lightweight regression checks for the canonical support knowledge base."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_anchor_support_answers_keep_expected_facts():
    responses = json.loads((ROOT / "predefined_responses.json").read_text())
    expected_substrings = {
        "password_reset": "Forgot Password",
        "refund_policy": "30 days",
        "order_tracking": "Orders",
        "payment_methods": "Visa",
        "first_time_discount": "WELCOME10",
        "customer_support": "email or live chat",
        "international_shipping": "select countries",
        "payment_security": "encrypted",
        "promo_code": "checkout",
        "return_terms": "original receipt",
    }

    for key, expected in expected_substrings.items():
        assert expected in responses[key]


def test_predefined_response_catalog_size_does_not_shrink():
    responses = json.loads((ROOT / "predefined_responses.json").read_text())
    assert len(responses) >= 19
