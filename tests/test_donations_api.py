from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend import donations


def test_resolve_public_site_url_prefers_configured_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMEAR_PUBLIC_SITE_URL", "https://play-smear.com/")

    assert donations.resolve_public_site_url("http://127.0.0.1:5174") == "https://play-smear.com"


def test_resolve_public_site_url_allows_local_dev_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SMEAR_PUBLIC_SITE_URL", raising=False)

    assert donations.resolve_public_site_url("http://localhost:5174") == "http://localhost:5174"


def test_build_donation_return_url_keeps_donate_route() -> None:
    assert (
        donations.build_donation_return_url("https://play-smear.com", "success")
        == "https://play-smear.com/?donation=success#donate"
    )


def test_create_donation_checkout_session_uses_hosted_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class FakeSessionApi:
        @staticmethod
        def create(**kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(url="https://checkout.stripe.test/session")

    fake_stripe = SimpleNamespace(
        api_key=None,
        api_version=None,
        checkout=SimpleNamespace(Session=FakeSessionApi),
    )

    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_unit")
    monkeypatch.setenv("SMEAR_PUBLIC_SITE_URL", "https://play-smear.com")
    monkeypatch.setattr(donations, "_load_stripe_module", lambda: fake_stripe)

    checkout_url = donations.create_donation_checkout_session(500)

    assert checkout_url == "https://checkout.stripe.test/session"
    assert fake_stripe.api_key == "sk_test_unit"
    assert fake_stripe.api_version == donations.STRIPE_API_VERSION
    assert len(calls) == 1
    assert calls[0]["mode"] == "payment"
    assert calls[0]["submit_type"] == "donate"
    assert "payment_method_types" not in calls[0]
    assert calls[0]["success_url"] == "https://play-smear.com/?donation=success#donate"
    assert calls[0]["cancel_url"] == "https://play-smear.com/?donation=cancelled#donate"


def test_create_donation_checkout_session_requires_secret_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)

    with pytest.raises(donations.DonationConfigurationError):
        donations.create_donation_checkout_session(500)
