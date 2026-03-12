"""
Test Suite for the Inverted Index Module.

Tests cover index building, text tokenisation, stop-word removal,
frequency/position tracking, TF and IDF calculation, TF-IDF ranking,
serialisation/deserialisation, and edge cases.

Run with:
    pytest tests/test_indexer.py -v
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.indexer import DocumentMeta, InvertedIndex, PostingEntry, STOP_WORDS

# ---------------------------------------------------------------------------
# Sample HTML fixtures
# ---------------------------------------------------------------------------

SIMPLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
  <p>The quick brown fox jumps over the lazy dog.</p>
  <p>The fox is quick and the dog is lazy.</p>
</body>
</html>
"""

SECOND_HTML = """
<!DOCTYPE html>
<html>
<head><title>Second Page</title></head>
<body>
  <p>A wise fox knows many things about life and love.</p>
</body>
</html>
"""

EMPTY_HTML = """
<!DOCTYPE html>
<html>
<head><title>Empty Page</title></head>
<body></body>
</html>
"""

HTML_WITH_SCRIPT = """
<!DOCTYPE html>
<html>
<head><title>Script Page</title>
<script>var x = "hidden text";</script>
<style>.hidden { display: none; }</style>
</head>
<body><p>Visible text only.</p></body>
</html>
"""


# ---------------------------------------------------------------------------
# Text extraction tests
# ---------------------------------------------------------------------------


class TestTextExtraction:
    """Tests for HTML → plain text extraction."""

    def test_extracts_visible_text(self) -> None:
        """Visible paragraph text is extracted."""
        text = InvertedIndex.extract_text(SIMPLE_HTML)
        assert "quick brown fox" in text
        assert "lazy dog" in text

    def test_strips_script_and_style(self) -> None:
        """Script and style content is removed."""
        text = InvertedIndex.extract_text(HTML_WITH_SCRIPT)
        assert "hidden text" not in text
        assert "display: none" not in text
        assert "Visible text only" in text

    def test_extracts_title(self) -> None:
        """The <title> tag content is extracted."""
        title = InvertedIndex.extract_title(SIMPLE_HTML)
        assert title == "Test Page"

    def test_missing_title(self) -> None:
        """Missing title returns 'Untitled'."""
        title = InvertedIndex.extract_title("<html><body>No title</body></html>")
        assert title == "Untitled"


# ---------------------------------------------------------------------------
# Tokenisation tests
# ---------------------------------------------------------------------------


class TestTokenisation:
    """Tests for text tokenisation and normalisation."""

    def test_basic_tokenisation(self) -> None:
        """Words are split and lowercased."""
        tokens = InvertedIndex.tokenise("Hello World")
        assert tokens == ["hello", "world"]

    def test_punctuation_removal(self) -> None:
        """Punctuation is stripped from tokens."""
        tokens = InvertedIndex.tokenise("Hello, World! How are you?")
        assert "hello" in tokens
        assert "world" in tokens
        # Commas, exclamation marks etc. should not appear in tokens
        assert all("," not in t and "!" not in t for t in tokens)

    def test_preserves_contractions(self) -> None:
        """Contractions like don't and it's are preserved."""
        tokens = InvertedIndex.tokenise("Don't worry, it's fine.")
        assert "don't" in tokens
        assert "it's" in tokens

    def test_numbers_preserved(self) -> None:
        """Numeric tokens are preserved."""
        tokens = InvertedIndex.tokenise("Chapter 42 in 2024")
        assert "42" in tokens
        assert "2024" in tokens

    def test_empty_string(self) -> None:
        """Empty string produces no tokens."""
        assert InvertedIndex.tokenise("") == []

    def test_case_insensitivity(self) -> None:
        """All tokens are lowercased."""
        tokens = InvertedIndex.tokenise("FOX Fox fOx")
        assert tokens == ["fox", "fox", "fox"]


# ---------------------------------------------------------------------------
# Stop word tests
# ---------------------------------------------------------------------------


class TestStopWords:
    """Tests for stop-word filtering."""

    def test_common_stop_words(self) -> None:
        """Common English stop words are identified."""
        assert InvertedIndex.is_stop_word("the") is True
        assert InvertedIndex.is_stop_word("is") is True
        assert InvertedIndex.is_stop_word("and") is True
        assert InvertedIndex.is_stop_word("a") is True

    def test_content_words_not_stopped(self) -> None:
        """Content words are not flagged as stop words."""
        assert InvertedIndex.is_stop_word("fox") is False
        assert InvertedIndex.is_stop_word("love") is False
        assert InvertedIndex.is_stop_word("python") is False

    def test_stop_words_excluded_from_index(self) -> None:
        """Stop words are not present in the built index."""
        idx = InvertedIndex()
        idx.build_from_pages({"http://a.com": SIMPLE_HTML})

        for sw in ["the", "is", "and", "over"]:
            assert sw not in idx.index, f"Stop word '{sw}' should not be in index"


# ---------------------------------------------------------------------------
# Single document indexing tests
# ---------------------------------------------------------------------------


class TestSingleDocumentIndex:
    """Tests for indexing a single HTML document."""

    @pytest.fixture()
    def single_index(self) -> InvertedIndex:
        """Build an index from a single page."""
        idx = InvertedIndex()
        idx.build_from_pages({"http://a.com": SIMPLE_HTML})
        return idx

    def test_word_in_index(self, single_index: InvertedIndex) -> None:
        """Content words appear in the index."""
        assert "fox" in single_index.index
        assert "quick" in single_index.index
        assert "lazy" in single_index.index

    def test_document_metadata(self, single_index: InvertedIndex) -> None:
        """Document metadata is stored correctly."""
        meta = single_index.documents["http://a.com"]
        assert meta.title == "Test Page"
        assert meta.word_count > 0
        assert meta.url == "http://a.com"

    def test_word_frequency(self, single_index: InvertedIndex) -> None:
        """Word frequency is counted correctly."""
        # "fox" appears twice in SIMPLE_HTML
        entry = single_index.index["fox"]["http://a.com"]
        assert entry.frequency == 2

    def test_word_positions(self, single_index: InvertedIndex) -> None:
        """Word positions are recorded and are distinct."""
        entry = single_index.index["fox"]["http://a.com"]
        assert len(entry.positions) == 2
        # Positions should be different
        assert entry.positions[0] != entry.positions[1]

    def test_tf_calculation(self, single_index: InvertedIndex) -> None:
        """TF = frequency / total_words_in_document."""
        entry = single_index.index["fox"]["http://a.com"]
        meta = single_index.documents["http://a.com"]
        expected_tf = entry.frequency / meta.word_count
        assert abs(entry.tf - expected_tf) < 1e-6


# ---------------------------------------------------------------------------
# Multiple document indexing tests
# ---------------------------------------------------------------------------


class TestMultipleDocumentIndex:
    """Tests for indexing multiple documents."""

    @pytest.fixture()
    def multi_index(self) -> InvertedIndex:
        """Build an index from two pages."""
        idx = InvertedIndex()
        idx.build_from_pages({
            "http://a.com": SIMPLE_HTML,
            "http://b.com": SECOND_HTML,
        })
        return idx

    def test_shared_term_across_docs(self, multi_index: InvertedIndex) -> None:
        """A term appearing in both documents has two posting entries."""
        assert "fox" in multi_index.index
        assert len(multi_index.index["fox"]) == 2

    def test_unique_term_single_doc(self, multi_index: InvertedIndex) -> None:
        """A term unique to one document has one posting entry."""
        assert "lazy" in multi_index.index
        assert len(multi_index.index["lazy"]) == 1

    def test_idf_calculation(self, multi_index: InvertedIndex) -> None:
        """IDF is computed correctly using smoothed formula."""
        import math
        n = 2  # two documents

        # "fox" appears in both docs → df=2
        df_fox = len(multi_index.index["fox"])
        expected_idf_fox = math.log((n + 1) / (df_fox + 1)) + 1
        assert abs(multi_index.idf["fox"] - expected_idf_fox) < 1e-6

        # "lazy" appears in 1 doc → df=1
        df_lazy = len(multi_index.index["lazy"])
        expected_idf_lazy = math.log((n + 1) / (df_lazy + 1)) + 1
        assert abs(multi_index.idf["lazy"] - expected_idf_lazy) < 1e-6

    def test_idf_unique_higher_than_shared(self, multi_index: InvertedIndex) -> None:
        """A term unique to one doc has higher IDF than one in both docs."""
        assert multi_index.idf["lazy"] > multi_index.idf["fox"]

    def test_tfidf_ranking(self, multi_index: InvertedIndex) -> None:
        """Search results are ranked by TF-IDF score descending."""
        results = multi_index.search("fox")
        assert len(results) >= 2
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Search tests (on the index)
# ---------------------------------------------------------------------------


class TestIndexSearch:
    """Tests for the search method on InvertedIndex."""

    @pytest.fixture()
    def index(self) -> InvertedIndex:
        idx = InvertedIndex()
        idx.build_from_pages({
            "http://a.com": SIMPLE_HTML,
            "http://b.com": SECOND_HTML,
        })
        return idx

    def test_single_term_search(self, index: InvertedIndex) -> None:
        """Single-term search returns matching documents."""
        results = index.search("fox")
        assert len(results) >= 1

    def test_multi_term_and_search(self, index: InvertedIndex) -> None:
        """Multi-term search uses AND logic."""
        results = index.search("fox lazy")
        # Both terms only appear together in SIMPLE_HTML
        assert len(results) == 1
        assert results[0]["url"] == "http://a.com"

    def test_nonexistent_term(self, index: InvertedIndex) -> None:
        """Searching for a word not in the index returns empty list."""
        results = index.search("xyzzyspoon")
        assert results == []

    def test_empty_query(self, index: InvertedIndex) -> None:
        """Empty query returns empty list."""
        results = index.search("")
        assert results == []

    def test_stop_word_only_query(self, index: InvertedIndex) -> None:
        """A query of only stop words returns empty list."""
        results = index.search("the and is")
        assert results == []

    def test_results_have_snippet(self, index: InvertedIndex) -> None:
        """Search results include a snippet field."""
        results = index.search("fox")
        for r in results:
            assert "snippet" in r


# ---------------------------------------------------------------------------
# Save / Load tests
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Tests for index serialisation and deserialisation."""

    def test_save_and_load_uncompressed(self) -> None:
        """Index can be saved and loaded without compression."""
        idx = InvertedIndex()
        idx.build_from_pages({"http://a.com": SIMPLE_HTML})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            idx.save(path, compress=False)

            loaded = InvertedIndex()
            loaded.load(path)

            assert set(loaded.index.keys()) == set(idx.index.keys())
            assert set(loaded.documents.keys()) == set(idx.documents.keys())

    def test_save_and_load_compressed(self) -> None:
        """Index can be saved and loaded with gzip compression."""
        idx = InvertedIndex()
        idx.build_from_pages({"http://a.com": SIMPLE_HTML})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            idx.save(path, compress=True)

            loaded = InvertedIndex()
            loaded.load(path + ".gz")

            assert set(loaded.index.keys()) == set(idx.index.keys())

    def test_save_load_integrity(self) -> None:
        """Loaded index preserves frequency, position, and TF data."""
        idx = InvertedIndex()
        idx.build_from_pages({"http://a.com": SIMPLE_HTML})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            idx.save(path, compress=False)

            loaded = InvertedIndex()
            loaded.load(path)

            for word in idx.index:
                for url in idx.index[word]:
                    orig = idx.index[word][url]
                    load = loaded.index[word][url]
                    assert orig.frequency == load.frequency
                    assert orig.positions == load.positions
                    assert abs(orig.tf - load.tf) < 1e-6

    def test_load_nonexistent_file(self) -> None:
        """Loading a nonexistent file raises FileNotFoundError."""
        idx = InvertedIndex()
        with pytest.raises(FileNotFoundError):
            idx.load("/nonexistent/path/index.json")

    def test_load_corrupt_file(self) -> None:
        """Loading a file with bad checksum raises ValueError."""
        idx = InvertedIndex()
        idx.build_from_pages({"http://a.com": SIMPLE_HTML})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.json")
            idx.save(path, compress=False)

            # Corrupt the file by modifying a value
            with open(path, "r") as f:
                data = json.load(f)
            data["checksum"] = "deadbeef"
            with open(path, "w") as f:
                json.dump(data, f)

            loaded = InvertedIndex()
            with pytest.raises(ValueError, match="integrity"):
                loaded.load(path)

    def test_load_missing_section(self) -> None:
        """Loading a file missing required sections raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.json")
            with open(path, "w") as f:
                json.dump({"index": {}, "documents": {}}, f)  # Missing "idf"

            idx = InvertedIndex()
            with pytest.raises(ValueError, match="missing"):
                idx.load(path)


# ---------------------------------------------------------------------------
# Empty index edge case tests
# ---------------------------------------------------------------------------


class TestEmptyIndex:
    """Tests for edge cases with an empty index."""

    def test_empty_index_search(self) -> None:
        """Searching an empty index returns empty list."""
        idx = InvertedIndex()
        results = idx.search("anything")
        assert results == []

    def test_empty_index_get_word_info(self) -> None:
        """get_word_info on empty index returns None."""
        idx = InvertedIndex()
        assert idx.get_word_info("anything") is None

    def test_build_from_empty_pages(self) -> None:
        """Building from an empty dict produces an empty index."""
        idx = InvertedIndex()
        idx.build_from_pages({})
        assert len(idx.index) == 0
        assert len(idx.documents) == 0

    def test_build_from_empty_html(self) -> None:
        """Building from a page with no text content works without error."""
        idx = InvertedIndex()
        idx.build_from_pages({"http://a.com": EMPTY_HTML})
        assert "http://a.com" in idx.documents


# ---------------------------------------------------------------------------
# get_word_info tests
# ---------------------------------------------------------------------------


class TestGetWordInfo:
    """Tests for the get_word_info query method."""

    def test_returns_correct_structure(self) -> None:
        """get_word_info returns dict with expected keys."""
        idx = InvertedIndex()
        idx.build_from_pages({"http://a.com": SIMPLE_HTML})

        info = idx.get_word_info("fox")
        assert info is not None
        assert info["word"] == "fox"
        assert "idf" in info
        assert "document_frequency" in info
        assert "postings" in info

    def test_posting_has_tfidf(self) -> None:
        """Each posting in get_word_info includes a tf_idf score."""
        idx = InvertedIndex()
        idx.build_from_pages({"http://a.com": SIMPLE_HTML})

        info = idx.get_word_info("fox")
        assert info is not None
        for url, posting in info["postings"].items():
            assert "tf_idf" in posting
            assert posting["tf_idf"] > 0

    def test_case_insensitive_lookup(self) -> None:
        """get_word_info is case-insensitive."""
        idx = InvertedIndex()
        idx.build_from_pages({"http://a.com": SIMPLE_HTML})

        assert idx.get_word_info("FOX") is not None
        assert idx.get_word_info("Fox") is not None

    def test_nonexistent_word(self) -> None:
        """get_word_info returns None for words not in the index."""
        idx = InvertedIndex()
        idx.build_from_pages({"http://a.com": SIMPLE_HTML})

        assert idx.get_word_info("xyzzy") is None
