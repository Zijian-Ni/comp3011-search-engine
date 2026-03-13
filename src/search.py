"""
Search Engine Module for COMP3011 CW2 Search Engine.

Provides a high-level search interface that wraps the InvertedIndex,
adding formatted output, query suggestion ("did you mean?"), and
performance benchmarking.

Complexity Analysis:
    - find:       O(T * D)  where T = query terms, D = matching docs
    - print_word: O(D)      where D = docs containing the word
    - suggest:    O(V)      where V = vocabulary size (edit distance scan)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.indexer import InvertedIndex

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """
    A single search result with relevance metadata.

    Attributes:
        url:     Document URL.
        title:   Page title.
        score:   Combined TF-IDF relevance score.
        snippet: Contextual text excerpt around the matched terms.
    """
    url: str
    title: str
    score: float
    snippet: str

    def __str__(self) -> str:
        """Human-readable single-line representation."""
        return f"[{self.score:.4f}] {self.title} — {self.url}\n         {self.snippet}"


# ---------------------------------------------------------------------------
# Search Engine
# ---------------------------------------------------------------------------


class SearchEngine:
    """
    High-level search engine combining crawling, indexing, and querying.

    Typical CLI workflow::

        engine = SearchEngine()
        engine.build()          # crawl + index + save
        engine.load()           # load from disk
        engine.find("love")     # search
        engine.print_word("life")

    Attributes:
        index:         The underlying InvertedIndex instance.
        last_latency:  Wall-clock time of the most recent ``find`` call (seconds).
    """

    # Default path for the persisted index file
    DEFAULT_INDEX_PATH = "data/index.json.gz"

    def __init__(self, index: Optional[InvertedIndex] = None) -> None:
        """
        Initialise the search engine.

        Args:
            index: An existing InvertedIndex to use.  If ``None``, a fresh
                   empty index is created.
        """
        self.index: InvertedIndex = index if index is not None else InvertedIndex()
        self.last_latency: float = 0.0

    # ------------------------------------------------------------------
    # Build / Load
    # ------------------------------------------------------------------

    def build(
        self,
        base_url: str = "https://quotes.toscrape.com",
        politeness_delay: float = 6.0,
        save_path: Optional[str] = None,
    ) -> int:
        """
        Crawl the target website, build the inverted index, and save it.

        Args:
            base_url:          Root URL to crawl.
            politeness_delay:  Seconds between HTTP requests.
            save_path:         Where to save the compiled index.

        Returns:
            The number of pages successfully crawled and indexed.
        """
        # Import here to avoid circular dependency issues in tests
        from src.crawler import WebCrawler

        print(f"[build] Crawling {base_url} (delay={politeness_delay}s) …")
        crawler = WebCrawler(base_url, politeness_delay=politeness_delay)
        pages = crawler.crawl()
        print(f"[build] Crawled {len(pages)} pages.")

        print("[build] Building inverted index …")
        self.index.build_from_pages(pages)
        print(
            f"[build] Index ready — {len(self.index.index)} terms, "
            f"{len(self.index.documents)} docs (took {self.index.build_time:.2f}s)"
        )

        path = save_path or self.DEFAULT_INDEX_PATH
        self.index.save(path)
        print(f"[build] Index saved to {path}")

        return len(pages)

    def load(self, filepath: Optional[str] = None) -> None:
        """
        Load a previously built index from disk.

        Args:
            filepath: Path to the index file.  Defaults to
                      ``data/index.json.gz``.

        Raises:
            FileNotFoundError: If the index file does not exist.
            ValueError: If the index file is corrupt.
        """
        path = filepath or self.DEFAULT_INDEX_PATH
        print(f"[load] Loading index from {path} …")
        self.index.load(path)
        print(
            f"[load] Loaded {len(self.index.index)} terms, "
            f"{len(self.index.documents)} documents."
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def find(self, query: str) -> List[SearchResult]:
        """
        Search for pages matching *query* (AND semantics for multi-word).

        Results are ranked by TF-IDF score.  Execution time is recorded
        in ``self.last_latency``.

        Args:
            query: The search query string.

        Returns:
            A list of :class:`SearchResult` objects, highest score first.
        """
        if not query or not query.strip():
            return []

        start = time.time()
        raw_results = self.index.search(query)
        self.last_latency = time.time() - start

        results = [
            SearchResult(
                url=r["url"],
                title=r["title"],
                score=r["score"],
                snippet=r["snippet"],
            )
            for r in raw_results
        ]

        return results

    # ------------------------------------------------------------------
    # Print word info
    # ------------------------------------------------------------------

    def print_word(self, word: str) -> str:
        """
        Format the inverted-index entry for a single word as a readable string.

        Args:
            word: The term to look up.

        Returns:
            A multi-line formatted string, or a "not found" message.
        """
        info = self.index.get_word_info(word)

        if info is None:
            suggestion = self.suggest(word)
            msg = f"Word '{word}' not found in index."
            if suggestion:
                msg += f" Did you mean '{suggestion}'?"
            return msg

        lines = [
            f"=== Inverted Index: '{info['word']}' ===",
            f"IDF Score:           {info['idf']:.6f}",
            f"Document Frequency:  {info['document_frequency']}",
            "",
        ]

        for url, posting in info["postings"].items():
            doc_meta = self.index.documents.get(url)
            title = doc_meta.title if doc_meta else "Unknown"
            lines.append(f"  Document: {title}")
            lines.append(f"  URL:      {url}")
            lines.append(f"  Freq:     {posting['frequency']}")
            lines.append(f"  TF:       {posting['tf']:.6f}")
            lines.append(f"  TF-IDF:   {posting['tf_idf']:.6f}")
            positions_str = ", ".join(str(p) for p in posting["positions"][:20])
            if len(posting["positions"]) > 20:
                positions_str += " …"
            lines.append(f"  Positions: [{positions_str}]")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Query suggestions ("did you mean?")
    # ------------------------------------------------------------------

    def suggest(self, word: str, max_distance: int = 2) -> Optional[str]:
        """
        Suggest a word from the index vocabulary that is close to *word*
        (using Levenshtein edit distance).

        Args:
            word:          The misspelled query term.
            max_distance:  Maximum edit distance to consider.

        Returns:
            The closest word in the index, or None if no close match exists.

        Complexity:
            O(V * L²) where V = vocabulary size, L = max word length.
        """
        word = word.lower().strip()
        if word in self.index.index:
            return None  # Word exists — no suggestion needed

        best_word: Optional[str] = None
        best_dist: int = max_distance + 1

        for candidate in self.index.index:
            # Quick length-based pruning
            if abs(len(candidate) - len(word)) > max_distance:
                continue

            dist = self._edit_distance(word, candidate)
            if dist < best_dist:
                best_dist = dist
                best_word = candidate

        return best_word if best_dist <= max_distance else None

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """
        Compute Levenshtein edit distance between two strings.

        Uses the standard dynamic-programming approach with O(min(m, n))
        space optimisation.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            The minimum number of single-character edits (insert, delete,
            substitute) to transform *s1* into *s2*.
        """
        if len(s1) < len(s2):
            return SearchEngine._edit_distance(s2, s1)

        # s1 is the longer string
        prev: List[int] = list(range(len(s2) + 1))

        for i, c1 in enumerate(s1, 1):
            curr = [i] + [0] * len(s2)
            for j, c2 in enumerate(s2, 1):
                if c1 == c2:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
            prev = curr

        return prev[len(s2)]

    # ------------------------------------------------------------------
    # Formatted output helpers
    # ------------------------------------------------------------------

    def format_results(self, results: List[SearchResult], query: str) -> str:
        """
        Format a list of search results as a human-readable string.

        Args:
            results: Search results to format.
            query:   The original query string (for the header).

        Returns:
            A formatted multi-line string.
        """
        if not results:
            # Try to suggest
            terms = self.index.tokenise(query)
            suggestions = []
            for t in terms:
                s = self.suggest(t)
                if s:
                    suggestions.append(f"'{t}' → '{s}'")
            msg = f"No results found for '{query}'."
            if suggestions:
                msg += "\nDid you mean: " + ", ".join(suggestions) + "?"
            return msg

        lines = [
            f"Search results for '{query}' ({len(results)} found, {self.last_latency*1000:.1f}ms):",
            "-" * 60,
        ]

        for i, r in enumerate(results, 1):
            lines.append(f"  {i}. [{r.score:.4f}] {r.title}")
            lines.append(f"     {r.url}")
            if r.snippet:
                lines.append(f"     {r.snippet}")
            lines.append("")

        return "\n".join(lines)
