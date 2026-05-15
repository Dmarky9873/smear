from __future__ import annotations

import importlib
import os
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit


STRIPE_API_VERSION = "2026-04-22.dahlia"
DEFAULT_DONATION_CURRENCY = "cad"
DEFAULT_PUBLIC_SITE_URL = "http://127.0.0.1:5174"
LOCAL_DEV_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
}


class DonationConfigurationError(RuntimeError):
    pass


class DonationCheckoutError(RuntimeError):
    pass


def _normalize_site_url(site_url: str) -> str:
    parsed = urlsplit(site_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DonationConfigurationError(
            "SMEAR_PUBLIC_SITE_URL must be an absolute http or https URL."
        )

    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def resolve_public_site_url(origin: str | None = None) -> str:
    configured_site_url = os.getenv("SMEAR_PUBLIC_SITE_URL", "").strip()
    if configured_site_url:
        return _normalize_site_url(configured_site_url)

    normalized_origin = origin.strip() if origin else ""
    if normalized_origin in LOCAL_DEV_ORIGINS:
        return _normalize_site_url(normalized_origin)

    return DEFAULT_PUBLIC_SITE_URL


def build_donation_return_url(site_url: str, donation_status: str) -> str:
    return f"{site_url}/?{urlencode({'donation': donation_status})}#donate"


def get_donation_currency() -> str:
    currency = os.getenv("SMEAR_DONATION_CURRENCY", DEFAULT_DONATION_CURRENCY)
    currency = currency.strip().lower()
    if len(currency) != 3 or not currency.isalpha():
        raise DonationConfigurationError(
            "SMEAR_DONATION_CURRENCY must be a three-letter currency code."
        )
    return currency


def _load_stripe_module() -> Any:
    try:
        return importlib.import_module("stripe")
    except ModuleNotFoundError as exc:
        raise DonationConfigurationError(
            "Stripe checkout is not available until the backend requirements are installed."
        ) from exc


def create_donation_checkout_session(
    amount_cents: int,
    origin: str | None = None,
) -> str:
    secret_key = os.getenv("STRIPE_SECRET_KEY", "").strip()
    if not secret_key:
        raise DonationConfigurationError(
            "Stripe checkout is not configured. Set STRIPE_SECRET_KEY on the backend."
        )

    stripe = _load_stripe_module()
    stripe.api_key = secret_key
    stripe.api_version = STRIPE_API_VERSION

    site_url = resolve_public_site_url(origin)
    currency = get_donation_currency()

    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            submit_type="donate",
            line_items=[
                {
                    "price_data": {
                        "currency": currency,
                        "product_data": {
                            "name": "Smear site donation",
                        },
                        "unit_amount": amount_cents,
                    },
                    "quantity": 1,
                }
            ],
            success_url=build_donation_return_url(site_url, "success"),
            cancel_url=build_donation_return_url(site_url, "cancelled"),
            metadata={
                "source": "smear",
                "type": "donation",
            },
        )
    except Exception as exc:
        raise DonationCheckoutError("Could not start Stripe Checkout.") from exc

    checkout_url = getattr(session, "url", None)
    if not checkout_url:
        raise DonationCheckoutError("Stripe did not return a Checkout URL.")

    return str(checkout_url)
