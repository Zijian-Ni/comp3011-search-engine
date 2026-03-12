"""
Inverted Index Module for COMP3011 CW2 Search Engine.

Builds, stores, and queries a TF-IDF weighted inverted index from
crawled HTML pages.  The index supports:
    - Word frequency and positional information per document
    - TF-IDF scoring for relevance ranking
    - Stop-word filtering and text normalisation
    - Persistent storage via compressed JSON
    - Integrity validation on load

Complexity Analysis:
    - build_from_pages:  O(N * W)  where N = documents, W = avg words/doc
    - search (single):   O(D)      where D = docs containing the term
    - search (multi):    O(T * D)  where T = query terms
    - save / load:       O(V * D)  where V = vocabulary size
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import os
import re
import string
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# English stop words (common list)
# ---------------------------------------------------------------------------
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "should", "shouldn't", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "were",
    "weren't", "what", "when", "where", "which", "while", "who",
    "whom", "why", "with", "won't", "would", "wouldn't", "you",
    "your", "yours", "yourself", "yourselves",
}


@dataclass
class DocumentMeta:
    """Metadata for a single crawled document.

    Attributes:
        url:        Canonical URL.
        title:      Page ``<title>`` or derived heading.
        word_count: Total token count after normalisation.
        raw_text:   Extracted plain text (used for snippet generation).
    """
    url: str = ""
    title: str = ""
    word_count: int = 0
    raw_text: str = ""


@dataclass
class PostingEntry:
    """A single posting-list entry for one (term, document) pair.

    Attributes:
        frequency:  How many times the term appears in the document.
        positions:  0-based word positions where the term occurs.
        tf:         Term frequency  = frequency / total_words_in_doc.
    """
    frequency: int = 0
    positions: List[int] = field(default_factory=list)
    tf: float = 0.0


class InvertedIndex:
    """
    TF-IDF weighted inverted index over a collection of HTML pages.

    Structure::

        index[word][url] = PostingEntry(frequency, positions, tf)
        idf[word]        = log(N / df)  where df = len(index[word])
        documents[url]   = DocumentMeta(...)

    Example:
        >>> idx = InvertedIndex()
        >>> idx.build_from_pages({"http://example.com": "<html><body>hello world</body></html>"})
        >>> "hello" in idx.index
        True
    """

    # Regex to split text into word tokens
    _TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

    def __init__(self) -> None:
        """Initialise an empty index."""
        # word -> {url -> PostingEntry}
        self.index: Dict[str, Dict[str, PostingEntry]] = {}
        # url -> DocumentMeta
        self.documents: Dict[str, DocumentMeta] = {}
        # word -> IDF score
        self.idf: Dict[str, float] = {}
        # Build-time benchmarking
        self.build_time: float = 0.0

    # ------------------------------------------------------------------
    # Text processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_text(html: str) -> str:
        """
        Extract visible text from raw HTML.

        Args:
            html: Raw HTML string.

        Returns:
            Plain-text content with excess whitespace collapsed.
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove script / style elements
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def extract_title(html: str) -> str:
        """
        Extract the ``<title>`` tag content from HTML.

        Args:
            html: Raw HTML string.

        Returns:
            Title text or ``"Untitled"`` if absent.
        """
        soup = BeautifulSoup(html, "lxml")
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()
        return "Untitled"

    @classmethod
    def tokenise(cls, text: str) -> List[str]:
        """
        Tokenise text into lowercase alphabetic/numeric tokens.

        Punctuation is stripped, contractions are kept (e.g. ``don't``).

        Args:
            text: Plain text to tokenise.

        Returns:
            List of token strings.

        Example:
            >>> InvertedIndex.tokenise("Hello, World! It's 2024.")
            ['hello', 'world', "it's", '2024']
        """
        return cls._TOKEN_RE.findall(text.lower())

    @staticmethod
    def is_stop_word(word: str) -> bool:
        """Return True if *word* is a common English stop word."""
        return word in STOP_WORDS

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_from_pages(self, pages: Dict[str, str]) -> None:
        """
        Build the inverted index from a dict of crawled pages.

        For each page the method:
            1. Extracts visible text and title.
            2. Tokenises and normalises the text.
            3. Records word positions and frequencies.
            4. Computes TF per (word, document) pair.
            5. After all documents, computes IDF globally.

        Args:
            pages: Mapping of URL → raw HTML content.

        Complexity:
            O(N * W) where N = number of pages, W = average words per page.
        """
        start = time.time()
        logger.info("Building index from %d pages …", len(pages))

        self.index.clear()
        self.documents.clear()
        self.idf.clear()

        num_docs = len(pages)

        for doc_num, (url, html) in enumerate(pages.items(), 1):
            # Extract text and metadata
            raw_text = self.extract_text(html)
            title = self.extract_title(html)
            tokens = self.tokenise(raw_text)
            word_count = len(tokens)

            # Store document metadata
            self.documents[url] = DocumentMeta(
                url=url,
                title=title,
                word_count=word_count,
                raw_text=raw_text,
            )

            # Build per-document posting entries
            # {word: PostingEntry} for this document
            local: Dict[str, PostingEntry] = {}

            for position, token in enumerate(tokens):
                # Skip stop words
                if self.is_stop_word(token):
                    continue

                if token not in local:
                    local[token] = PostingEntry()

                entry = local[token]
                entry.frequency += 1
                entry.positions.append(position)

            # Compute TF and merge into global index
            for word, entry in local.items():
                entry.tf = entry.frequency / word_count if word_count > 0 else 0.0

                if word not in self.index:
                    self.index[word] = {}
                self.index[word][url] = entry

            if doc_num % 20 == 0 or doc_num == num_docs:
                logger.info("Indexed %d / %d documents", doc_num, num_docs)

        # Compute IDF for every term
        for word, postings in self.index.items():
            df = len(postings)  # document frequency
            # Smoothed IDF: log((N + 1) / (df + 1)) + 1
            self.idf[word] = math.log((num_docs + 1) / (df + 1)) + 1

        self.build_time = time.time() - start
        logger.info(
            "Index built in %.2fs — %d terms across %d documents",
            self.build_time,
            len(self.index),
            len(self.documents),
        )

    # ------------------------------------------------------------------
    # Persistence (save / load)
    # ------------------------------------------------------------------

    def _compute_checksum(self, data: dict) -> str:
        """Compute MD5 checksum of the JSON payload for integrity validation."""
        raw = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.md5(raw).hexdigest()

    def save(self, filepath: str, compress: bool = True) -> None:
        """
        Serialise the index to a JSON file (optionally gzip-compressed).

        The output contains an integrity checksum so that ``load`` can
        verify the file has not been corrupted.

        Args:
            filepath: Destination file path.
            compress: If True, apply gzip compression (adds ``.gz`` if needed).
        """
        # Build serialisable representation
        index_data: Dict[str, Any] = {}
        for word, postings in self.index.items():
            index_data[word] = {}
            for url, entry in postings.items():
                index_data[word][url] = {
                    "frequency": entry.frequency,
                    "positions": entry.positions,
                    "tf": round(entry.tf, 8),
                }

        doc_data: Dict[str, Any] = {}
        for url, meta in self.documents.items():
            doc_data[url] = {
                "title": meta.title,
                "word_count": meta.word_count,
                "raw_text": meta.raw_text,
            }

        payload: Dict[str, Any] = {
            "index": index_data,
            "documents": doc_data,
            "idf": {k: round(v, 8) for k, v in self.idf.items()},
            "build_time": round(self.build_time, 4),
            "version": "1.0",
        }

        # Checksum (computed *before* adding checksum field)
        payload["checksum"] = self._compute_checksum({
            "index": payload["index"],
            "documents": payload["documents"],
            "idf": payload["idf"],
        })

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        if compress:
            if not filepath.endswith(".gz"):
                filepath += ".gz"
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info("Index saved (compressed) to %s", filepath)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info("Index saved to %s", filepath)

    def load(self, filepath: str) -> None:
        """
        Load a previously saved index from disk.

        Validates the embedded checksum to detect corruption.

        Args:
            filepath: Path to the index file (``.json`` or ``.json.gz``).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the checksum does not match or the format is invalid.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index file not found: {filepath}")

        if filepath.endswith(".gz"):
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)

        # Validate structure
        for key in ("index", "documents", "idf"):
            if key not in payload:
                raise ValueError(f"Invalid index file: missing '{key}' section")

        # Validate checksum
        stored_checksum = payload.get("checksum", "")
        computed_checksum = self._compute_checksum({
            "index": payload["index"],
            "documents": payload["documents"],
            "idf": payload["idf"],
        })
        if stored_checksum and stored_checksum != computed_checksum:
            raise ValueError(
                f"Index file integrity check failed: "
                f"expected {stored_checksum}, got {computed_checksum}"
            )

        # Deserialise into objects
        self.index.clear()
        for word, postings in payload["index"].items():
            self.index[word] = {}
            for url, data in postings.items():
                self.index[word][url] = PostingEntry(
                    frequency=data["frequency"],
                    positions=data["positions"],
                    tf=data["tf"],
                )

        self.documents.clear()
        for url, data in payload["documents"].items():
            self.documents[url] = DocumentMeta(
                url=url,
                title=data["title"],
                word_count=data["word_count"],
                raw_text=data.get("raw_text", ""),
            )

        self.idf = {k: float(v) for k, v in payload["idf"].items()}
        self.build_time = payload.get("build_time", 0.0)

        logger.info(
            "Index loaded: %d terms, %d documents",
            len(self.index),
            len(self.documents),
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_word_info(self, word: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the full inverted-index entry for a single word.

        Args:
            word: The search term (will be lowercased).

        Returns:
            A dict with ``idf`` and per-document posting data, or None
            if the word is not in the index.
        """
        word = word.lower().strip()
        if word not in self.index:
            return None

        result: Dict[str, Any] = {
            "word": word,
            "idf": self.idf.get(word, 0.0),
            "document_frequency": len(self.index[word]),
            "postings": {},
        }

        for url, entry in self.index[word].items():
            result["postings"][url] = {
                "frequency": entry.frequency,
                "positions": entry.positions,
                "tf": entry.tf,
                "tf_idf": entry.tf * self.idf.get(word, 0.0),
            }

        return result

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the index for documents matching all query terms (AND logic).

        Results are ranked by the *sum* of TF-IDF scores across query terms.

        Args:
            query: Space-separated search terms.

        Returns:
            A list of result dicts sorted by descending relevance score::

                [{"url": str, "title": str, "score": float, "snippet": str}, …]
        """
        tokens = self.tokenise(query)
        # Remove stop words from query
        terms = [t for t in tokens if not self.is_stop_word(t)]

        if not terms:
            return []

        # Find documents containing ALL terms (AND search)
        candidate_sets = []
        for term in terms:
            if term in self.index:
                candidate_sets.append(set(self.index[term].keys()))
            else:
                # Term not in index → no results for AND query
                return []

        # Intersection of all candidate sets
        matching_urls: Set[str] = candidate_sets[0]
        for s in candidate_sets[1:]:
            matching_urls &= s

        if not matching_urls:
            return []

        # Score each matching document
        results: List[Dict[str, Any]] = []
        for url in matching_urls:
            score = 0.0
            for term in terms:
                entry = self.index[term][url]
                idf = self.idf.get(term, 0.0)
                score += entry.tf * idf

            doc_meta = self.documents.get(url, DocumentMeta())
            snippet = self._extract_snippet(doc_meta.raw_text, terms)

            results.append({
                "url": url,
                "title": doc_meta.title,
                "score": round(score, 6),
                "snippet": snippet,
            })

        # Sort by score descending
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Snippet extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_snippet(
        text: str,
        terms: List[str],
        context_words: int = 8,
        max_length: int = 200,
    ) -> str:
        """
        Extract a contextual snippet around the first occurrence of any
        query term in the document text.

        Args:
            text:          Plain text of the document.
            terms:         Query terms to highlight.
            context_words: Number of words to show either side of the match.
            max_length:    Maximum character length of the returned snippet.

        Returns:
            A snippet string with ``**term**`` markers around matched words,
            or an empty string if no match is found.
        """
        if not text or not terms:
            return ""

        words = text.split()
        lower_words = [w.lower().strip(string.punctuation) for w in words]

        # Find the position of the first matching term
        best_pos: Optional[int] = None
        for i, lw in enumerate(lower_words):
            if lw in terms:
                best_pos = i
                break

        if best_pos is None:
            # Return the beginning of the text as fallback
            snippet = " ".join(words[:context_words * 2])
            return (snippet[:max_length] + "…") if len(snippet) > max_length else snippet

        start = max(0, best_pos - context_words)
        end = min(len(words), best_pos + context_words + 1)
        snippet_words = words[start:end]

        # Bold the matching terms
        highlighted = []
        for w in snippet_words:
            clean = w.lower().strip(string.punctuation)
            if clean in terms:
                highlighted.append(f"**{w}**")
            else:
                highlighted.append(w)

        prefix = "… " if start > 0 else ""
        suffix = " …" if end < len(words) else ""
        snippet = prefix + " ".join(highlighted) + suffix

        if len(snippet) > max_length:
            snippet = snippet[:max_length] + "…"

        return snippet
