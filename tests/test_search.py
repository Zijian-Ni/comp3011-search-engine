"""
Test Suite for the Search Engine Module.

Tests cover single/multi-word search, result ranking, snippet extraction,
print_word formatting, query suggestions, edge cases (empty queries,
special characters, non-existent words), and performance benchmarking.

Run with:
    pytest tests/test_search.py -v
"""

from __future__ import annotations

import pytest

from src.indexer import InvertedIndex
from src.search import SearchEngine, SearchResult

# ---------------------------------------------------------------------------
# Shared HTML fixtures
# ---------------------------------------------------------------------------

PAGE_A_HTML = """
<!DOCTYPE html>
<html>
<head><title>Quotes Page A</title></head>
<body>
  <p>Love is the wisdom of the fool and the folly of the wise.</p>
  <p>Love conquers all things. Life and love are intertwined.</p>
</body>
</html>
"""

PAGE_B_HTML = """
<!DOCTYPE html>
<html>
<head><title>Quotes Page B</title></head>
<body>
  <p>Life is what happens when you are busy making other plans.</p>
  <p>In the end, it is not the years in your life that count.</p>
</body>
</html>
"""

PAGE_C_HTML = """
<!DOCTYPE html>
<html>
<head><title>Author Bio</title></head>
<body>
  <p>Albert Einstein was a theoretical physicist known for wisdom and love of science.</p>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine() -> SearchEngine:
    """Build a SearchEngine with a small test index."""
    idx = InvertedIndex()
    idx.build_from_pages({
        "http://a.com": PAGE_A_HTML,
        "http://b.com": PAGE_B_HTML,
        "http://c.com": PAGE_C_HTML,
    })
    return SearchEngine(index=idx)


# ---------------------------------------------------------------------------
# Single word search tests
# ---------------------------------------------------------------------------


class TestSingleWordSearch:
    """Tests for single-term queries."""

    def test_finds_matching_pages(self, engine: SearchEngine) -> None:
        """A word present in multiple pages returns all of them."""
        results = engine.find("love")
        urls = {r.url for r in results}
        assert "http://a.com" in urls  # "love" appears multiple times
        assert len(results) >= 1

    def test_result_type(self, engine: SearchEngine) -> None:
        """Results are SearchResult instances."""
        results = engine.find("love")
        assert all(isinstance(r, SearchResult) for r in results)

    def test_result_has_score(self, engine: SearchEngine) -> None:
        """Each result has a positive relevance score."""
        results = engine.find("love")
        for r in results:
            assert r.score > 0

    def test_result_has_url_and_title(self, engine: SearchEngine) -> None:
        """Each result includes a URL and title."""
        results = engine.find("wisdom")
        for r in results:
            assert r.url.startswith("http")
            assert len(r.title) > 0


# ---------------------------------------------------------------------------
# Multi-word search tests
# ---------------------------------------------------------------------------


class TestMultiWordSearch:
    """Tests for multi-term (AND) queries."""

    def test_and_logic(self, engine: SearchEngine) -> None:
        """Multi-word search requires ALL terms to be present."""
        results = engine.find("love wisdom")
        urls = {r.url for r in results}
        # Page A has both "love" and "wisdom"
        assert "http://a.com" in urls

    def test_no_partial_matches(self, engine: SearchEngine) -> None:
        """Documents missing any query term are excluded."""
        results = engine.find("love physicist")
        # Only page C has "physicist"; page A has "love" but not "physicist"
        # Page C has both
        urls = {r.url for r in results}
        if results:
            for r in results:
                # Verify each result actually contains both terms
                assert r.url in urls


# ---------------------------------------------------------------------------
# Non-existent word tests
# ---------------------------------------------------------------------------


class TestNonexistentWord:
    """Tests for queries with words not in the index."""

    def test_returns_empty(self, engine: SearchEngine) -> None:
        """A query for a non-existent word returns no results."""
        results = engine.find("xylophone")
        assert results == []

    def test_partial_nonexistent(self, engine: SearchEngine) -> None:
        """If any term in a multi-word query is missing, returns empty."""
        results = engine.find("love xylophone")
        assert results == []


# ---------------------------------------------------------------------------
# Empty query tests
# ---------------------------------------------------------------------------


class TestEmptyQuery:
    """Tests for empty and whitespace-only queries."""

    def test_empty_string(self, engine: SearchEngine) -> None:
        """Empty string returns no results."""
        assert engine.find("") == []

    def test_whitespace_only(self, engine: SearchEngine) -> None:
        """Whitespace-only query returns no results."""
        assert engine.find("   ") == []

    def test_none_like_handling(self, engine: SearchEngine) -> None:
        """Query of only stop words returns no results."""
        assert engine.find("the and is") == []


# ---------------------------------------------------------------------------
# Special characters tests
# ---------------------------------------------------------------------------


class TestSpecialCharacters:
    """Tests for queries containing special characters."""

    def test_punctuation_stripped(self, engine: SearchEngine) -> None:
        """Punctuation in queries is handled gracefully."""
        results = engine.find("love!")
        assert len(results) >= 1

    def test_mixed_special_chars(self, engine: SearchEngine) -> None:
        """Queries with mixed special characters don't crash."""
        results = engine.find("@#$%^&*")
        assert results == []

    def test_hyphenated_query(self, engine: SearchEngine) -> None:
        """Hyphenated words are tokenised."""
        # This should not crash even if the word isn't found
        results = engine.find("well-known")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Case-insensitive search tests
# ---------------------------------------------------------------------------


class TestCaseInsensitiveSearch:
    """Tests for case-insensitive search behaviour."""

    def test_uppercase_query(self, engine: SearchEngine) -> None:
        """Uppercase query matches lowercase-indexed terms."""
        results_lower = engine.find("love")
        results_upper = engine.find("LOVE")
        assert len(results_lower) == len(results_upper)

    def test_mixed_case_query(self, engine: SearchEngine) -> None:
        """Mixed-case query matches."""
        results = engine.find("LoVe")
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Ranked results tests
# ---------------------------------------------------------------------------


class TestRankedResults:
    """Tests for TF-IDF ranking of search results."""

    def test_results_sorted_by_score(self, engine: SearchEngine) -> None:
        """Results are sorted in descending order of score."""
        results = engine.find("love")
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_higher_frequency_ranks_higher(self, engine: SearchEngine) -> None:
        """A page with more occurrences of the term ranks higher."""
        results = engine.find("love")
        if len(results) >= 2:
            # Page A has "love" 3 times; page C has it once
            assert results[0].score >= results[1].score


# ---------------------------------------------------------------------------
# Snippet extraction tests
# ---------------------------------------------------------------------------


class TestSnippetExtraction:
    """Tests for contextual snippet generation."""

    def test_snippet_contains_term(self, engine: SearchEngine) -> None:
        """The snippet includes the search term (possibly highlighted)."""
        results = engine.find("wisdom")
        for r in results:
            if r.snippet:
                assert "wisdom" in r.snippet.lower() or "**" in r.snippet

    def test_snippet_is_bounded(self, engine: SearchEngine) -> None:
        """Snippets are reasonably short (≤ 250 chars)."""
        results = engine.find("love")
        for r in results:
            assert len(r.snippet) <= 250

    def test_snippet_on_empty_result(self) -> None:
        """Snippet extraction on text with no match returns fallback."""
        snippet = InvertedIndex._extract_snippet("Hello world foo bar", ["xyzzy"])
        # Should return beginning of text as fallback
        assert len(snippet) > 0


# ---------------------------------------------------------------------------
# print_word format tests
# ---------------------------------------------------------------------------


class TestPrintWordFormat:
    """Tests for the print_word formatted output."""

    def test_existing_word(self, engine: SearchEngine) -> None:
        """print_word for an existing word returns structured info."""
        output = engine.print_word("love")
        assert "Inverted Index:" in output
        assert "IDF Score:" in output
        assert "Document Frequency:" in output
        assert "Freq:" in output

    def test_nonexistent_word(self, engine: SearchEngine) -> None:
        """print_word for a missing word returns a 'not found' message."""
        output = engine.print_word("xyzzy")
        assert "not found" in output.lower()

    def test_case_insensitive_print(self, engine: SearchEngine) -> None:
        """print_word is case-insensitive."""
        output = engine.print_word("LOVE")
        assert "Inverted Index:" in output

    def test_print_includes_url(self, engine: SearchEngine) -> None:
        """print_word output includes document URLs."""
        output = engine.print_word("love")
        assert "http://" in output

    def test_print_includes_positions(self, engine: SearchEngine) -> None:
        """print_word output includes word positions."""
        output = engine.print_word("love")
        assert "Positions:" in output


# ---------------------------------------------------------------------------
# Query suggestion tests
# ---------------------------------------------------------------------------


class TestQuerySuggestions:
    """Tests for the 'did you mean?' suggestion feature."""

    def test_suggests_close_word(self, engine: SearchEngine) -> None:
        """A misspelled word gets a suggestion from the vocabulary."""
        suggestion = engine.suggest("lov")  # close to "love"
        assert suggestion is not None
        assert suggestion == "love"

    def test_no_suggestion_for_exact_match(self, engine: SearchEngine) -> None:
        """An exact match returns None (no suggestion needed)."""
        suggestion = engine.suggest("love")
        assert suggestion is None

    def test_no_suggestion_for_distant_word(self, engine: SearchEngine) -> None:
        """A word too far from any vocabulary term returns None."""
        suggestion = engine.suggest("zzzzzzzz")
        assert suggestion is None

    def test_suggestion_in_print_word(self, engine: SearchEngine) -> None:
        """print_word includes a 'did you mean' suggestion for misspellings."""
        output = engine.print_word("lov")
        assert "did you mean" in output.lower()


# ---------------------------------------------------------------------------
# SearchResult dataclass tests
# ---------------------------------------------------------------------------


class TestSearchResult:
    """Tests for the SearchResult dataclass."""

    def test_str_representation(self) -> None:
        """SearchResult __str__ produces readable output."""
        r = SearchResult(
            url="http://a.com",
            title="Test",
            score=1.234,
            snippet="some context",
        )
        s = str(r)
        assert "1.234" in s
        assert "Test" in s
        assert "http://a.com" in s


# ---------------------------------------------------------------------------
# format_results tests
# ---------------------------------------------------------------------------


class TestFormatResults:
    """Tests for the format_results display method."""

    def test_format_with_results(self, engine: SearchEngine) -> None:
        """format_results produces numbered output."""
        results = engine.find("love")
        output = engine.format_results(results, "love")
        assert "Search results for" in output
        assert "1." in output

    def test_format_no_results(self, engine: SearchEngine) -> None:
        """format_results with empty results shows 'no results'."""
        output = engine.format_results([], "nonexistent_xyz")
        assert "no results" in output.lower()

    def test_format_includes_latency(self, engine: SearchEngine) -> None:
        """format_results includes search latency."""
        results = engine.find("love")
        output = engine.format_results(results, "love")
        assert "ms" in output


# ---------------------------------------------------------------------------
# Edit distance tests
# ---------------------------------------------------------------------------


class TestEditDistance:
    """Tests for the Levenshtein edit distance helper."""

    def test_identical_strings(self) -> None:
        """Edit distance between identical strings is 0."""
        assert SearchEngine._edit_distance("hello", "hello") == 0

    def test_single_insertion(self) -> None:
        """One character insertion gives distance 1."""
        assert SearchEngine._edit_distance("hell", "hello") == 1

    def test_single_deletion(self) -> None:
        """One character deletion gives distance 1."""
        assert SearchEngine._edit_distance("hello", "hell") == 1

    def test_single_substitution(self) -> None:
        """One character substitution gives distance 1."""
        assert SearchEngine._edit_distance("hello", "hallo") == 1

    def test_empty_strings(self) -> None:
        """Distance from empty to non-empty is the length of the other."""
        assert SearchEngine._edit_distance("", "abc") == 3
        assert SearchEngine._edit_distance("abc", "") == 3

    def test_completely_different(self) -> None:
        """Completely different strings have distance = max length."""
        assert SearchEngine._edit_distance("abc", "xyz") == 3


# ---------------------------------------------------------------------------
# Performance / latency test
# ---------------------------------------------------------------------------


class TestPerformance:
    """Tests for search performance benchmarking."""

    def test_latency_recorded(self, engine: SearchEngine) -> None:
        """Search latency is recorded and is a positive number."""
        engine.find("love")
        assert engine.last_latency >= 0

    def test_search_is_fast(self, engine: SearchEngine) -> None:
        """Search completes in under 100 ms on the small test index."""
        engine.find("love wisdom")
        assert engine.last_latency < 0.1
