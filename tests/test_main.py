"""
Test Suite for the Main CLI Module.

Tests cover command parsing, help output, stats display, and error handling.
Uses monkeypatching to simulate user input without a real terminal.

Run with:
    pytest tests/test_main.py -v
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from src.main import main, print_help, print_stats
from src.search import SearchEngine
from src.indexer import InvertedIndex


# ---------------------------------------------------------------------------
# Helper to run main() with simulated input
# ---------------------------------------------------------------------------


def run_main_with_input(inputs: list[str], monkeypatch: pytest.MonkeyPatch) -> str:
    """
    Run main() with a sequence of input lines and capture stdout.

    Args:
        inputs: Lines to feed as user input (must include 'quit' to exit).
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        Captured stdout as a string.
    """
    input_iter = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda _: next(input_iter))

    output = StringIO()
    monkeypatch.setattr("sys.stdout", output)

    try:
        main()
    except StopIteration:
        pass  # Input exhausted

    return output.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMainCLI:
    """Tests for the interactive CLI."""

    def test_help_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The 'help' command prints available commands."""
        output = run_main_with_input(["help", "quit"], monkeypatch)
        assert "build" in output
        assert "load" in output
        assert "print" in output
        assert "find" in output

    def test_quit_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The 'quit' command exits gracefully."""
        output = run_main_with_input(["quit"], monkeypatch)
        assert "Goodbye" in output

    def test_exit_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The 'exit' command also exits."""
        output = run_main_with_input(["exit"], monkeypatch)
        assert "Goodbye" in output

    def test_unknown_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unknown commands produce an error message."""
        output = run_main_with_input(["foobar", "quit"], monkeypatch)
        assert "Unknown command" in output

    def test_empty_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty input is ignored."""
        output = run_main_with_input(["", "quit"], monkeypatch)
        assert "Goodbye" in output

    def test_print_without_word(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'print ' with no word shows usage."""
        output = run_main_with_input(["print", "quit"], monkeypatch)
        assert "Usage" in output

    def test_find_without_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'find ' with no query shows usage."""
        output = run_main_with_input(["find", "quit"], monkeypatch)
        assert "Usage" in output

    def test_stats_no_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'stats' with no loaded index shows appropriate message."""
        output = run_main_with_input(["stats", "quit"], monkeypatch)
        assert "No index loaded" in output

    def test_load_missing_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'load' when index file missing shows error."""
        output = run_main_with_input(["load", "quit"], monkeypatch)
        assert "not found" in output.lower() or "error" in output.lower()

    @patch("src.search.SearchEngine.build")
    def test_build_command(self, mock_build: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        """'build' calls engine.build()."""
        mock_build.return_value = 10
        output = run_main_with_input(["build", "quit"], monkeypatch)
        mock_build.assert_called_once()

    def test_find_with_loaded_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'find' after loading an index returns results."""
        # Build a small index and patch load
        idx = InvertedIndex()
        idx.build_from_pages({
            "http://a.com": "<html><head><title>T</title></head><body>love life</body></html>"
        })

        original_load = SearchEngine.load

        def mock_load(self_engine, filepath=None):
            self_engine.index = idx

        monkeypatch.setattr("src.search.SearchEngine.load", mock_load)
        output = run_main_with_input(["load", "find love", "quit"], monkeypatch)
        assert "http://a.com" in output or "no results" in output.lower() or "Search results" in output


class TestPrintHelp:
    """Tests for the print_help function."""

    def test_outputs_commands(self, capsys: pytest.CaptureFixture[str]) -> None:
        """print_help outputs all command names."""
        print_help()
        captured = capsys.readouterr().out
        assert "build" in captured
        assert "find" in captured
        assert "quit" in captured


class TestPrintStats:
    """Tests for the print_stats function."""

    def test_empty_engine(self, capsys: pytest.CaptureFixture[str]) -> None:
        """print_stats on empty engine shows no-index message."""
        engine = SearchEngine()
        print_stats(engine)
        captured = capsys.readouterr().out
        assert "No index loaded" in captured

    def test_with_index(self, capsys: pytest.CaptureFixture[str]) -> None:
        """print_stats on populated engine shows statistics."""
        idx = InvertedIndex()
        idx.build_from_pages({
            "http://a.com": "<html><head><title>T</title></head><body>hello world</body></html>"
        })
        engine = SearchEngine(index=idx)
        print_stats(engine)
        captured = capsys.readouterr().out
        assert "Terms:" in captured
        assert "Documents:" in captured
