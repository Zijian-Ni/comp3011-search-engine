"""
Web Crawler Module for COMP3011 CW2 Search Engine.

This module implements a polite web crawler that respects robots.txt,
enforces a configurable politeness delay between requests, handles errors
with exponential backoff retries, and discovers all reachable pages on
a target website (pagination, tag, and author pages).

Complexity Analysis:
    - Time:  O(N * D) where N = number of pages, D = politeness delay
    - Space: O(N * S) where S = average page size in characters
"""

from __future__ import annotations

import logging
import re
import time
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class CrawlStats:
    """Container for crawl statistics used in progress reporting."""

    def __init__(self) -> None:
        self.pages_crawled: int = 0
        self.pages_failed: int = 0
        self.total_bytes: int = 0
        self.start_time: float = 0.0

    def elapsed(self) -> float:
        """Return elapsed time in seconds since crawl started."""
        return time.time() - self.start_time

    def summary(self) -> str:
        """Return a human-readable summary of crawl statistics."""
        return (
            f"Crawled {self.pages_crawled} pages | "
            f"Failed {self.pages_failed} | "
            f"Downloaded {self.total_bytes / 1024:.1f} KB | "
            f"Elapsed {self.elapsed():.1f}s"
        )


class WebCrawler:
    """
    A polite web crawler for quotes.toscrape.com.

    Features:
        - Respects robots.txt
        - Configurable politeness delay (default 6 s)
        - Exponential-backoff retries on transient errors
        - Discovers pagination, tag, and author pages
        - Deduplicates URLs via normalisation
        - Emits progress reports to the logger

    Attributes:
        base_url:          Root URL to begin crawling.
        politeness_delay:  Minimum seconds between consecutive HTTP requests.
        max_retries:       Maximum retry attempts for failed requests.
        timeout:           HTTP request timeout in seconds.
        user_agent:        User-Agent string sent with every request.
        visited:           Set of already-visited normalised URLs.
        pages:             Mapping of URL → raw HTML content for crawled pages.
        stats:             Crawl statistics container.

    Example:
        >>> crawler = WebCrawler("https://quotes.toscrape.com")
        >>> pages = crawler.crawl()
        >>> len(pages) > 0
        True
    """

    # Patterns for links we want to follow on quotes.toscrape.com
    _FOLLOW_PATTERNS: List[re.Pattern[str]] = [
        re.compile(r"^/page/\d+/?$"),        # pagination
        re.compile(r"^/tag/[\w-]+/?$"),       # tag listing pages
        re.compile(r"^/tag/[\w-]+/page/\d+/?$"),  # tag pagination
        re.compile(r"^/author/[\w-]+/?$"),    # author bio pages
        re.compile(r"^/$"),                   # homepage
    ]

    def __init__(
        self,
        base_url: str,
        politeness_delay: float = 6.0,
        max_retries: int = 3,
        timeout: float = 30.0,
        user_agent: str = "COMP3011-SearchBot/1.0 (University of Leeds CW2)",
    ) -> None:
        """
        Initialise the crawler.

        Args:
            base_url:          Root URL of the target website.
            politeness_delay:  Seconds to wait between HTTP requests.
            max_retries:       How many times to retry a failed request.
            timeout:           HTTP timeout in seconds.
            user_agent:        User-Agent header value.
        """
        # Normalise base URL (strip trailing slash for consistency)
        self.base_url: str = base_url.rstrip("/")
        self.politeness_delay: float = politeness_delay
        self.max_retries: int = max_retries
        self.timeout: float = timeout
        self.user_agent: str = user_agent

        # Internal state
        self.visited: Set[str] = set()
        self.pages: Dict[str, str] = {}
        self.stats: CrawlStats = CrawlStats()

        # robots.txt parser (initialised lazily)
        self._robots: Optional[RobotFileParser] = None
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # robots.txt
    # ------------------------------------------------------------------

    def _load_robots_txt(self) -> None:
        """
        Parse the site's robots.txt so we can check crawl permissions.

        If robots.txt is unreachable we default to *allow all* — a common
        convention for polite crawlers when the file is absent.
        """
        self._robots = RobotFileParser()
        robots_url = f"{self.base_url}/robots.txt"
        try:
            self._robots.set_url(robots_url)
            self._robots.read()
            logger.info("Loaded robots.txt from %s", robots_url)
        except Exception as exc:
            logger.warning(
                "Could not load robots.txt (%s); assuming all paths allowed.",
                exc,
            )
            # Reset to a permissive parser
            self._robots = RobotFileParser()
            self._robots.allow_all = True  # type: ignore[attr-defined]

    def is_allowed(self, url: str) -> bool:
        """
        Check whether the crawler is permitted to fetch *url* per robots.txt.

        Args:
            url: Absolute URL to check.

        Returns:
            True if fetching is permitted, False otherwise.
        """
        if self._robots is None:
            self._load_robots_txt()
        assert self._robots is not None
        return self._robots.can_fetch(self.user_agent, url)

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalise_url(url: str) -> str:
        """
        Normalise a URL by stripping fragments, query strings, and trailing
        slashes so that equivalent URLs are recognised as the same page.

        Args:
            url: The URL string to normalise.

        Returns:
            A normalised URL string.

        Examples:
            >>> WebCrawler.normalise_url("https://example.com/page/1/")
            'https://example.com/page/1'
            >>> WebCrawler.normalise_url("https://example.com/page#top")
            'https://example.com/page'
        """
        parsed = urlparse(url)
        # Reconstruct without query and fragment, strip trailing slash
        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
        return clean

    def _is_internal(self, url: str) -> bool:
        """Return True if *url* belongs to the same domain as *base_url*."""
        return urlparse(url).netloc == urlparse(self.base_url).netloc

    def _should_follow(self, path: str) -> bool:
        """
        Decide whether a relative *path* matches one of our follow patterns.

        We restrict the crawl to pagination, tag, and author pages to avoid
        crawling login/external/API endpoints.

        Args:
            path: The path component of a URL (e.g. ``/page/2/``).

        Returns:
            True if the path matches a crawlable pattern.
        """
        path = path.rstrip("/") + "/" if not path.endswith("/") else path
        path_no_slash = path.rstrip("/")
        for pattern in self._FOLLOW_PATTERNS:
            if pattern.match(path) or pattern.match(path_no_slash):
                return True
        return False

    # ------------------------------------------------------------------
    # HTTP fetching with retries
    # ------------------------------------------------------------------

    def _wait_politeness(self) -> None:
        """Block until the politeness window has elapsed since the last request."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.politeness_delay:
            wait = self.politeness_delay - elapsed
            logger.debug("Politeness wait: %.2fs", wait)
            time.sleep(wait)

    def fetch(self, url: str) -> Optional[str]:
        """
        Fetch a single URL with retries and exponential backoff.

        Args:
            url: The absolute URL to fetch.

        Returns:
            The response body as a string, or None if all retries failed.
        """
        headers = {"User-Agent": self.user_agent}

        for attempt in range(1, self.max_retries + 1):
            try:
                self._wait_politeness()
                self._last_request_time = time.time()

                response = requests.get(url, headers=headers, timeout=self.timeout)
                response.raise_for_status()

                self.stats.total_bytes += len(response.content)
                logger.debug("Fetched %s (attempt %d, %d bytes)", url, attempt, len(response.content))
                return response.text

            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "?"
                logger.warning("HTTP %s for %s (attempt %d/%d)", status, url, attempt, self.max_retries)
                # Don't retry 404 — the page genuinely doesn't exist
                if exc.response is not None and exc.response.status_code == 404:
                    self.stats.pages_failed += 1
                    return None

            except requests.exceptions.ConnectionError:
                logger.warning("Connection error for %s (attempt %d/%d)", url, attempt, self.max_retries)

            except requests.exceptions.Timeout:
                logger.warning("Timeout for %s (attempt %d/%d)", url, attempt, self.max_retries)

            except requests.exceptions.RequestException as exc:
                logger.warning("Request error for %s: %s (attempt %d/%d)", url, exc, attempt, self.max_retries)

            # Exponential backoff (but still respect politeness delay)
            if attempt < self.max_retries:
                backoff = self.politeness_delay * (2 ** (attempt - 1))
                logger.debug("Backing off %.1fs before retry", backoff)
                time.sleep(backoff)

        logger.error("Failed to fetch %s after %d attempts", url, self.max_retries)
        self.stats.pages_failed += 1
        return None

    # ------------------------------------------------------------------
    # Link extraction
    # ------------------------------------------------------------------

    def extract_links(self, html: str, source_url: str) -> List[str]:
        """
        Extract and normalise all follow-worthy links from an HTML page.

        Args:
            html:        Raw HTML content.
            source_url:  The URL the HTML was fetched from (for resolving
                         relative links).

        Returns:
            A list of normalised absolute URLs to crawl next.
        """
        soup = BeautifulSoup(html, "lxml")
        links: List[str] = []

        for anchor in soup.find_all("a", href=True):
            href: str = anchor["href"]

            # Resolve relative URLs
            absolute = urljoin(source_url, href)

            # Must be on the same domain
            if not self._is_internal(absolute):
                continue

            # Check path against our follow patterns
            path = urlparse(absolute).path
            if not self._should_follow(path):
                continue

            normalised = self.normalise_url(absolute)
            if normalised not in self.visited:
                links.append(normalised)

        return links

    # ------------------------------------------------------------------
    # Main crawl loop
    # ------------------------------------------------------------------

    def crawl(self) -> Dict[str, str]:
        """
        Crawl the target website starting from *base_url*.

        Uses breadth-first traversal.  Respects robots.txt, politeness
        delay, and retry policy.  Returns all successfully fetched pages.

        Returns:
            A dict mapping normalised URL → raw HTML content.

        Complexity:
            O(N) pages visited, each taking O(D) seconds for the delay,
            giving O(N·D) wall-clock time.
        """
        self.stats = CrawlStats()
        self.stats.start_time = time.time()
        self.visited.clear()
        self.pages.clear()

        # Seed the queue with the start URL
        start = self.normalise_url(self.base_url)
        queue: List[str] = [start]
        self.visited.add(start)

        logger.info("Starting crawl from %s", start)

        while queue:
            url = queue.pop(0)  # BFS

            # robots.txt check
            if not self.is_allowed(url):
                logger.info("Blocked by robots.txt: %s", url)
                continue

            # Fetch
            html = self.fetch(url)
            if html is None:
                continue

            self.pages[url] = html
            self.stats.pages_crawled += 1

            # Progress report every page
            logger.info(
                "[%d] %s | %s",
                self.stats.pages_crawled,
                url,
                self.stats.summary(),
            )

            # Discover new links
            new_links = self.extract_links(html, url)
            for link in new_links:
                if link not in self.visited:
                    self.visited.add(link)
                    queue.append(link)

        logger.info("Crawl complete. %s", self.stats.summary())
        return self.pages
