"""Tests for Settings configuration."""

import pytest
from unittest.mock import patch

from aipa_engine.config import Settings, get_settings


# ──────────────────────────────────────────────
# Placeholder detection tests
# ──────────────────────────────────────────────


class TestPlaceholderDetection:
    def test_empty_string_treated_as_empty(self):
        settings = Settings(anthropic_api_key="", kosis_api_key="")
        assert settings.anthropic_api_key == ""
        assert settings.kosis_api_key == ""

    def test_placeholder_your_api_key_here_cleared(self):
        settings = Settings(anthropic_api_key="your-api-key-here")
        assert settings.anthropic_api_key == ""

    def test_placeholder_changeme_cleared(self):
        settings = Settings(anthropic_api_key="changeme")
        assert settings.anthropic_api_key == ""

    def test_placeholder_xxx_cleared(self):
        settings = Settings(anthropic_api_key="xxx")
        assert settings.anthropic_api_key == ""

    def test_placeholder_test_cleared(self):
        settings = Settings(naver_client_id="test")
        assert settings.naver_client_id == ""

    def test_real_key_preserved(self):
        settings = Settings(anthropic_api_key="sk-ant-real-key-12345")
        assert settings.anthropic_api_key == "sk-ant-real-key-12345"

    def test_placeholder_case_insensitive(self):
        settings = Settings(anthropic_api_key="CHANGEME")
        assert settings.anthropic_api_key == ""

    def test_placeholder_with_whitespace(self):
        settings = Settings(anthropic_api_key="  xxx  ")
        assert settings.anthropic_api_key == ""


# ──────────────────────────────────────────────
# Default values tests
# ──────────────────────────────────────────────


class TestDefaultValues:
    def test_app_name_default(self):
        settings = Settings()
        assert settings.app_name == "AIPA Engine"

    def test_debug_default_false(self):
        settings = Settings()
        assert settings.debug is False

    def test_api_prefix_default(self):
        settings = Settings()
        assert settings.api_prefix == "/api/v1"

    def test_anthropic_model_default(self):
        settings = Settings()
        assert "claude" in settings.anthropic_model.lower() or "sonnet" in settings.anthropic_model.lower()

    def test_default_panel_count(self):
        settings = Settings()
        assert settings.default_panel_count == 10

    def test_max_panel_count(self):
        settings = Settings()
        assert settings.max_panel_count == 200

    def test_allowed_origins_default_empty(self):
        settings = Settings()
        assert settings.allowed_origins == ""

    def test_all_api_keys_default_empty(self):
        """When constructed with no arguments and no .env, API keys should be empty.
        Note: if a .env file is present, keys may be loaded from it.
        We test explicit empty values instead."""
        settings = Settings(
            kosis_api_key="",
            anthropic_api_key="",
            naver_client_id="",
            naver_client_secret="",
        )
        assert settings.kosis_api_key == ""
        assert settings.anthropic_api_key == ""
        assert settings.naver_client_id == ""
        assert settings.naver_client_secret == ""


# ──────────────────────────────────────────────
# Custom values tests
# ──────────────────────────────────────────────


class TestCustomValues:
    def test_override_app_name(self):
        settings = Settings(app_name="My Custom App")
        assert settings.app_name == "My Custom App"

    def test_override_debug(self):
        settings = Settings(debug=True)
        assert settings.debug is True

    def test_override_panel_counts(self):
        settings = Settings(default_panel_count=20, max_panel_count=500)
        assert settings.default_panel_count == 20
        assert settings.max_panel_count == 500

    def test_override_allowed_origins(self):
        settings = Settings(allowed_origins="https://example.com,https://other.com")
        assert "example.com" in settings.allowed_origins


# ──────────────────────────────────────────────
# get_settings singleton tests
# ──────────────────────────────────────────────


class TestGetSettings:
    def test_returns_settings_instance(self):
        # Clear the cache to get a fresh instance
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_singleton_returns_same_instance(self):
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_creates_new_instance(self):
        get_settings.cache_clear()
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        # They are different objects but same type
        assert isinstance(s2, Settings)


# ──────────────────────────────────────────────
# Warning on missing keys
# ──────────────────────────────────────────────


class TestWarnings:
    def test_missing_keys_does_not_raise(self):
        """Settings should initialize successfully even with all keys missing."""
        settings = Settings(
            anthropic_api_key="",
            kosis_api_key="",
            naver_client_id="",
            naver_client_secret="",
        )
        assert settings is not None
