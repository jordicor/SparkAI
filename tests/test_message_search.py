"""Tests for message_search module (build_fts_query and sanitize_snippet)."""

import sys
import os

# Add project root to path so we can import message_search
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from message_search import build_fts_query, sanitize_snippet


class TestBuildFtsQuery:
    def test_single_word(self):
        assert build_fts_query("hello") == "hello"

    def test_multiple_words(self):
        result = build_fts_query("hello world")
        assert "hello" in result
        assert "world" in result

    def test_quoted_phrase(self):
        result = build_fts_query('"exact phrase"')
        assert '"exact phrase"' in result

    def test_mixed_words_and_phrase(self):
        result = build_fts_query('hello "exact phrase" world')
        assert '"exact phrase"' in result
        assert "hello" in result
        assert "world" in result

    def test_dangerous_operators_stripped(self):
        result = build_fts_query("test*|hack^inject{}")
        # Should not contain any FTS5 operators
        assert "*" not in result
        assert "|" not in result
        assert "^" not in result
        assert "{" not in result
        assert "}" not in result
        # Should still contain the cleaned word parts
        assert "test" in result or "hack" in result

    def test_empty_after_sanitize(self):
        assert build_fts_query("***|||") == ""

    def test_empty_string(self):
        assert build_fts_query("") == ""

    def test_whitespace_only(self):
        assert build_fts_query("   ") == ""

    def test_colon_stripped(self):
        result = build_fts_query("NEAR:test")
        assert ":" not in result


class TestSanitizeSnippet:
    def test_preserves_mark_tags(self):
        raw = "before <mark>keyword</mark> after"
        result = sanitize_snippet(raw)
        assert "<mark>keyword</mark>" in result
        assert "before" in result
        assert "after" in result

    def test_escapes_malicious_html(self):
        raw = '<script>alert("xss")</script> <mark>safe</mark>'
        result = sanitize_snippet(raw)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result
        assert "<mark>safe</mark>" in result

    def test_plain_text_unchanged(self):
        raw = "just plain text without any html"
        result = sanitize_snippet(raw)
        assert result == raw

    def test_multiple_mark_tags(self):
        raw = "<mark>one</mark> middle <mark>two</mark>"
        result = sanitize_snippet(raw)
        assert "<mark>one</mark>" in result
        assert "<mark>two</mark>" in result

    def test_angle_brackets_in_text(self):
        raw = "2 < 3 and 5 > 4 <mark>match</mark>"
        result = sanitize_snippet(raw)
        assert "<mark>match</mark>" in result
        assert "&lt;" in result.replace("<mark>", "").replace("</mark>", "") or "2 &lt; 3" in result
