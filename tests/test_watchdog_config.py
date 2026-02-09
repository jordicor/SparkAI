"""
Unit tests for watchdog configuration validation, JSON extraction,
message cleaning, and prompt building.

Tests cover Phase 8.2 of the watchdog implementation plan.
"""

import pytest

from prompts import get_default_watchdog_config, validate_watchdog_config
from tools.llm_caller import extract_json_from_llm_response
from tools.watchdog import (
    _build_evaluation_prompt,
    _clean_message_content,
    _format_messages_for_eval,
    DEFAULT_STEERING_PROMPTS,
    VALID_EVENT_TYPES,
    VALID_SEVERITIES,
)


# ===================================================================
# get_default_watchdog_config
# ===================================================================

class TestGetDefaultWatchdogConfig:

    def test_returns_dict(self):
        config = get_default_watchdog_config()
        assert isinstance(config, dict)

    def test_disabled_by_default(self):
        config = get_default_watchdog_config()
        assert config["enabled"] is False
        assert config["llm_id"] is None

    def test_returns_fresh_dict_each_call(self):
        """Factory function must not return a shared mutable reference."""
        a = get_default_watchdog_config()
        b = get_default_watchdog_config()
        assert a is not b
        assert a["thresholds"] is not b["thresholds"]
        a["objectives"].append("mutated")
        assert "mutated" not in b["objectives"]

    def test_default_values(self):
        config = get_default_watchdog_config()
        assert config["mode"] == "custom"
        assert config["objectives"] == []
        assert config["steering_prompt"] == ""
        assert config["frequency"] == 3
        assert config["max_hint_chars"] == 500
        assert config["thresholds"]["max_turns_off_topic"] == 3
        assert config["thresholds"]["max_turns_same_subtopic"] == 5


# ===================================================================
# validate_watchdog_config
# ===================================================================

class TestValidateWatchdogConfig:

    # --- Happy paths ---

    def test_valid_enabled_config(self):
        config = {
            "enabled": True,
            "llm_id": 7,
            "mode": "interview",
            "objectives": ["Track topics", "Detect drift"],
            "frequency": 3,
        }
        result = validate_watchdog_config(config)
        assert result["enabled"] is True
        assert result["llm_id"] == 7
        assert result["mode"] == "interview"
        assert len(result["objectives"]) == 2
        assert result["frequency"] == 3

    def test_disabled_config_allows_empty_fields(self):
        result = validate_watchdog_config({"enabled": False})
        assert result["enabled"] is False
        assert result["llm_id"] is None
        assert result["objectives"] == []

    def test_disabled_config_ignores_invalid_llm_id(self):
        """When disabled, llm_id validation is skipped."""
        result = validate_watchdog_config({"enabled": False, "llm_id": None})
        assert result["enabled"] is False

    def test_valid_all_modes(self):
        for mode in ("interview", "coaching", "education", "custom"):
            config = {
                "enabled": True,
                "llm_id": 1,
                "mode": mode,
                "objectives": ["Test"],
            }
            result = validate_watchdog_config(config)
            assert result["mode"] == mode

    # --- Required fields when enabled ---

    def test_enabled_without_llm_id_raises(self):
        with pytest.raises(ValueError, match="llm_id is required"):
            validate_watchdog_config({"enabled": True, "llm_id": None, "objectives": ["X"]})

    def test_enabled_without_objectives_raises(self):
        with pytest.raises(ValueError, match="At least one objective"):
            validate_watchdog_config({"enabled": True, "llm_id": 1, "objectives": []})

    def test_enabled_with_empty_string_objectives_raises(self):
        """Objectives that are only whitespace should be filtered and raise."""
        with pytest.raises(ValueError, match="At least one non-empty objective"):
            validate_watchdog_config({"enabled": True, "llm_id": 1, "objectives": ["  ", ""]})

    def test_enabled_with_non_string_objectives_raises(self):
        """Non-string objectives are silently skipped; if none remain, raise."""
        with pytest.raises(ValueError, match="At least one non-empty objective"):
            validate_watchdog_config({"enabled": True, "llm_id": 1, "objectives": [123, None]})

    # --- Invalid modes ---

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            validate_watchdog_config({
                "enabled": True, "llm_id": 1, "mode": "therapy", "objectives": ["X"]
            })

    # --- llm_id validation ---

    def test_llm_id_string_converts_to_int(self):
        result = validate_watchdog_config({
            "enabled": True, "llm_id": "7", "mode": "custom", "objectives": ["X"]
        })
        assert result["llm_id"] == 7

    def test_llm_id_non_numeric_raises(self):
        with pytest.raises(ValueError, match="llm_id must be an integer"):
            validate_watchdog_config({
                "enabled": True, "llm_id": "abc", "mode": "custom", "objectives": ["X"]
            })

    # --- Frequency ---

    def test_frequency_zero_raises(self):
        with pytest.raises(ValueError, match="frequency must be between 1 and 20"):
            validate_watchdog_config({
                "enabled": True, "llm_id": 1, "objectives": ["X"], "frequency": 0
            })

    def test_frequency_21_raises(self):
        with pytest.raises(ValueError, match="frequency must be between 1 and 20"):
            validate_watchdog_config({
                "enabled": True, "llm_id": 1, "objectives": ["X"], "frequency": 21
            })

    def test_frequency_defaults_to_3(self):
        result = validate_watchdog_config({
            "enabled": True, "llm_id": 1, "objectives": ["X"]
        })
        assert result["frequency"] == 3

    def test_frequency_non_numeric_defaults(self):
        """Non-convertible frequency falls back to default 3."""
        result = validate_watchdog_config({
            "enabled": True, "llm_id": 1, "objectives": ["X"], "frequency": "abc"
        })
        assert result["frequency"] == 3

    # --- max_hint_chars ---

    def test_max_hint_chars_below_100_raises(self):
        with pytest.raises(ValueError, match="max_hint_chars must be between 100 and 2000"):
            validate_watchdog_config({
                "enabled": True, "llm_id": 1, "objectives": ["X"], "max_hint_chars": 50
            })

    def test_max_hint_chars_above_2000_raises(self):
        with pytest.raises(ValueError, match="max_hint_chars must be between 100 and 2000"):
            validate_watchdog_config({
                "enabled": True, "llm_id": 1, "objectives": ["X"], "max_hint_chars": 3000
            })

    def test_max_hint_chars_defaults_to_500(self):
        result = validate_watchdog_config({
            "enabled": True, "llm_id": 1, "objectives": ["X"]
        })
        assert result["max_hint_chars"] == 500

    # --- Thresholds ---

    def test_thresholds_valid(self):
        result = validate_watchdog_config({
            "enabled": True, "llm_id": 1, "objectives": ["X"],
            "thresholds": {"max_turns_off_topic": 5, "max_turns_same_subtopic": 10}
        })
        assert result["thresholds"]["max_turns_off_topic"] == 5
        assert result["thresholds"]["max_turns_same_subtopic"] == 10

    def test_thresholds_unknown_keys_discarded(self):
        result = validate_watchdog_config({
            "enabled": True, "llm_id": 1, "objectives": ["X"],
            "thresholds": {"max_turns_off_topic": 5, "unknown_key": 99}
        })
        assert "unknown_key" not in result["thresholds"]
        assert result["thresholds"]["max_turns_off_topic"] == 5

    def test_thresholds_below_0_raises(self):
        with pytest.raises(ValueError, match="threshold.*must be between 0 and 50"):
            validate_watchdog_config({
                "enabled": True, "llm_id": 1, "objectives": ["X"],
                "thresholds": {"max_turns_off_topic": -1}
            })

    def test_thresholds_zero_is_valid(self):
        result = validate_watchdog_config({
            "enabled": True, "llm_id": 1, "objectives": ["X"],
            "thresholds": {"max_warnings_before_action": 0}
        })
        assert result["thresholds"]["max_warnings_before_action"] == 0

    def test_thresholds_above_50_raises(self):
        with pytest.raises(ValueError, match="threshold.*must be between 0 and 50"):
            validate_watchdog_config({
                "enabled": True, "llm_id": 1, "objectives": ["X"],
                "thresholds": {"max_turns_same_subtopic": 51}
            })

    def test_thresholds_defaults_when_missing(self):
        result = validate_watchdog_config({
            "enabled": True, "llm_id": 1, "objectives": ["X"]
        })
        assert result["thresholds"]["max_turns_off_topic"] == 3
        assert result["thresholds"]["max_turns_same_subtopic"] == 5

    def test_thresholds_non_dict_ignored(self):
        result = validate_watchdog_config({
            "enabled": True, "llm_id": 1, "objectives": ["X"],
            "thresholds": "not a dict"
        })
        # Should use defaults
        assert result["thresholds"]["max_turns_off_topic"] == 3

    # --- Sanitization ---

    def test_steering_prompt_truncated_to_5000(self):
        long_prompt = "A" * 10000
        result = validate_watchdog_config({
            "enabled": True, "llm_id": 1, "objectives": ["X"],
            "steering_prompt": long_prompt,
        })
        assert len(result["steering_prompt"]) == 5000

    def test_steering_prompt_stripped(self):
        result = validate_watchdog_config({
            "enabled": True, "llm_id": 1, "objectives": ["X"],
            "steering_prompt": "  padded text  ",
        })
        assert result["steering_prompt"] == "padded text"

    def test_steering_prompt_xss_preserved_as_string(self):
        """XSS payload in steering_prompt is stored as-is (no HTML rendering)."""
        xss = '<script>alert("xss")</script>'
        result = validate_watchdog_config({
            "enabled": True, "llm_id": 1, "objectives": ["X"],
            "steering_prompt": xss,
        })
        assert result["steering_prompt"] == xss

    def test_objectives_truncated_to_500_chars(self):
        long_obj = "O" * 1000
        result = validate_watchdog_config({
            "enabled": True, "llm_id": 1, "objectives": [long_obj],
        })
        assert len(result["objectives"][0]) == 500

    def test_objectives_max_20(self):
        with pytest.raises(ValueError, match="Maximum 20 objectives"):
            validate_watchdog_config({
                "enabled": True, "llm_id": 1,
                "objectives": [f"Obj {i}" for i in range(21)],
            })

    def test_objectives_stripped(self):
        result = validate_watchdog_config({
            "enabled": True, "llm_id": 1, "objectives": ["  trim me  "],
        })
        assert result["objectives"][0] == "trim me"

    # --- Type coercion ---

    def test_not_dict_raises(self):
        with pytest.raises(ValueError, match="must be a JSON object"):
            validate_watchdog_config("string")

    def test_enabled_truthy_converts_to_bool(self):
        result = validate_watchdog_config({"enabled": 1, "llm_id": 1, "objectives": ["X"]})
        assert result["enabled"] is True


# ===================================================================
# extract_json_from_llm_response
# ===================================================================

class TestExtractJsonFromLlmResponse:

    def test_pure_json(self):
        text = '{"event_type": "none", "severity": "info", "analysis": "OK", "hint": ""}'
        result = extract_json_from_llm_response(text)
        assert result is not None
        assert result["event_type"] == "none"

    def test_json_with_markdown_fences(self):
        text = '```json\n{"event_type": "drift", "severity": "nudge"}\n```'
        result = extract_json_from_llm_response(text)
        assert result is not None
        assert result["event_type"] == "drift"

    def test_json_with_preamble(self):
        text = 'Here is my analysis:\n\n{"event_type": "stuck", "severity": "redirect", "analysis": "User seems stuck", "hint": "Try new topic"}'
        result = extract_json_from_llm_response(text)
        assert result is not None
        assert result["event_type"] == "stuck"

    def test_json_with_trailing_text(self):
        text = '{"event_type": "none", "severity": "info"}\n\nI hope this helps!'
        result = extract_json_from_llm_response(text)
        assert result is not None
        assert result["event_type"] == "none"

    def test_empty_string_returns_none(self):
        assert extract_json_from_llm_response("") is None

    def test_none_returns_none(self):
        assert extract_json_from_llm_response(None) is None

    def test_no_json_returns_none(self):
        assert extract_json_from_llm_response("Just some text without JSON") is None

    def test_malformed_json_returns_none(self):
        assert extract_json_from_llm_response("{broken: json}") is None

    def test_nested_json_extracts_outermost(self):
        text = '{"outer": {"inner": true}}'
        result = extract_json_from_llm_response(text)
        assert result is not None
        assert "outer" in result


# ===================================================================
# _clean_message_content
# ===================================================================

class TestCleanMessageContent:

    def test_plain_text_unchanged(self):
        assert _clean_message_content("Hello world") == "Hello world"

    def test_empty_string(self):
        assert _clean_message_content("") == ""

    def test_none_returns_empty(self):
        assert _clean_message_content(None) == ""

    def test_truncates_long_messages(self):
        long_msg = "Hello world! " * 300  # ~3900 chars, not base64-like
        result = _clean_message_content(long_msg)
        assert result.endswith("[truncated]")
        assert len(result) == 2000 + len(" [truncated]")

    def test_base64_string_returns_empty(self):
        b64 = "A" * 300  # Pure base64-like chars, length >= 200
        result = _clean_message_content(b64)
        assert result == ""

    def test_multimodal_extracts_text(self):
        import orjson
        content = orjson.dumps([
            {"type": "text", "text": "Hello from user"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]).decode()
        result = _clean_message_content(content)
        assert "Hello from user" in result

    def test_multimodal_only_images_returns_empty(self):
        import orjson
        content = orjson.dumps([
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]).decode()
        result = _clean_message_content(content)
        assert result == ""

    def test_dict_with_content_array(self):
        import orjson
        content = orjson.dumps({
            "content": [
                {"type": "text", "text": "Inner text"},
                {"type": "audio", "data": "..."},
            ]
        }).decode()
        result = _clean_message_content(content)
        assert "Inner text" in result

    def test_dict_with_content_string(self):
        import orjson
        content = orjson.dumps({"content": "Simple string content"}).decode()
        result = _clean_message_content(content)
        assert "Simple string content" in result

    def test_invalid_json_treated_as_text(self):
        msg = "[not valid json but starts with bracket"
        result = _clean_message_content(msg)
        assert result == msg


# ===================================================================
# _format_messages_for_eval
# ===================================================================

class TestFormatMessagesForEval:

    def test_basic_formatting(self):
        messages = [
            {"role": "user", "content": "Hi", "date": "2026-01-01"},
            {"role": "bot", "content": "Hello!", "date": "2026-01-01"},
        ]
        result = _format_messages_for_eval(messages)
        assert "[USER]: Hi" in result
        assert "[BOT]: Hello!" in result

    def test_preserves_order(self):
        messages = [
            {"role": "user", "content": "First", "date": "2026-01-01"},
            {"role": "bot", "content": "Second", "date": "2026-01-01"},
            {"role": "user", "content": "Third", "date": "2026-01-01"},
        ]
        result = _format_messages_for_eval(messages)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert "First" in lines[0]
        assert "Third" in lines[2]

    def test_empty_list(self):
        assert _format_messages_for_eval([]) == ""


# ===================================================================
# _build_evaluation_prompt
# ===================================================================

class TestBuildEvaluationPrompt:

    def test_steering_prompt_not_in_evaluation_prompt(self):
        """Steering prompt is sent as system_prompt to the LLM, not embedded in the user message."""
        config = {
            "steering_prompt": "Custom steering instructions here",
            "mode": "custom",
            "objectives": ["Obj A", "Obj B"],
            "thresholds": {"max_turns_off_topic": 3},
        }
        messages = [{"role": "user", "content": "test", "date": "2026-01-01"}]
        result = _build_evaluation_prompt(config, messages)
        assert "Custom steering instructions here" not in result
        assert "Obj A" in result
        assert "Obj B" in result

    def test_default_steering_not_in_evaluation_prompt(self):
        """Default steering prompt per mode is sent as system_prompt, not embedded in user message."""
        config = {
            "steering_prompt": "",
            "mode": "interview",
            "objectives": ["Track topics"],
            "thresholds": {},
        }
        messages = [{"role": "user", "content": "test", "date": "2026-01-01"}]
        result = _build_evaluation_prompt(config, messages)
        assert DEFAULT_STEERING_PROMPTS["interview"] not in result

    def test_includes_thresholds(self):
        config = {
            "steering_prompt": "Test",
            "objectives": ["X"],
            "thresholds": {"max_turns_off_topic": 5, "max_turns_same_subtopic": 8},
        }
        messages = [{"role": "user", "content": "test", "date": "2026-01-01"}]
        result = _build_evaluation_prompt(config, messages)
        assert "max_turns_off_topic: 5" in result
        assert "max_turns_same_subtopic: 8" in result

    def test_includes_messages(self):
        config = {
            "steering_prompt": "Test",
            "objectives": ["X"],
            "thresholds": {},
        }
        messages = [
            {"role": "user", "content": "Hello AI", "date": "2026-01-01"},
            {"role": "bot", "content": "Hello user", "date": "2026-01-01"},
        ]
        result = _build_evaluation_prompt(config, messages)
        assert "[USER]: Hello AI" in result
        assert "[BOT]: Hello user" in result

    def test_includes_json_response_format(self):
        config = {
            "steering_prompt": "X",
            "objectives": ["Y"],
            "thresholds": {},
        }
        result = _build_evaluation_prompt(config, [{"role": "user", "content": "msg", "date": ""}])
        assert '"event_type"' in result
        assert '"severity"' in result

    def test_empty_objectives_shows_placeholder(self):
        config = {
            "steering_prompt": "Test",
            "objectives": [],
            "thresholds": {},
        }
        result = _build_evaluation_prompt(config, [{"role": "user", "content": "x", "date": ""}])
        assert "(none specified)" in result

    def test_empty_thresholds_shows_defaults(self):
        config = {
            "steering_prompt": "Test",
            "objectives": ["X"],
            "thresholds": {},
        }
        result = _build_evaluation_prompt(config, [{"role": "user", "content": "x", "date": ""}])
        assert "(defaults)" in result

    def test_hint_tracking_appended_when_present(self):
        """When hint_tracking has consecutive_hint_count > 0, it should be appended."""
        config = {
            "steering_prompt": "Test",
            "objectives": ["X"],
            "thresholds": {},
        }
        tracking = {
            "consecutive_hint_count": 3,
            "pending_hint": "Redirect the conversation",
            "hint_severity": "redirect",
        }
        result = _build_evaluation_prompt(
            config,
            [{"role": "user", "content": "x", "date": ""}],
            hint_tracking=tracking,
        )
        assert "IGNORED HINT TRACKING" in result
        assert "Consecutive hints generated: 3" in result
        assert "Last hint severity: redirect" in result
        assert "Redirect the conversation" in result

    def test_hint_tracking_omitted_when_zero(self):
        """When consecutive_hint_count is 0, no tracking section should appear."""
        config = {
            "steering_prompt": "Test",
            "objectives": ["X"],
            "thresholds": {},
        }
        tracking = {
            "consecutive_hint_count": 0,
            "pending_hint": None,
            "hint_severity": None,
        }
        result = _build_evaluation_prompt(
            config,
            [{"role": "user", "content": "x", "date": ""}],
            hint_tracking=tracking,
        )
        assert "IGNORED HINT TRACKING" not in result


# ===================================================================
# Constants sanity checks
# ===================================================================

class TestConstants:

    def test_valid_event_types_is_frozenset(self):
        assert isinstance(VALID_EVENT_TYPES, frozenset)
        assert "drift" in VALID_EVENT_TYPES
        assert "none" in VALID_EVENT_TYPES
        assert "error" in VALID_EVENT_TYPES
        assert "security" in VALID_EVENT_TYPES

    def test_valid_severities_is_frozenset(self):
        assert isinstance(VALID_SEVERITIES, frozenset)
        assert "info" in VALID_SEVERITIES
        assert "nudge" in VALID_SEVERITIES
        assert "redirect" in VALID_SEVERITIES
        assert "alert" in VALID_SEVERITIES

    def test_default_steering_prompts_cover_all_modes(self):
        for mode in ("interview", "coaching", "education", "custom"):
            assert mode in DEFAULT_STEERING_PROMPTS
            assert len(DEFAULT_STEERING_PROMPTS[mode]) > 0
