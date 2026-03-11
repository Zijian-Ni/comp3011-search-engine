"""
Test Suite for the Web Crawler Module.

Tests cover initialisation, URL normalisation, link extraction, politeness
delay enforcement, pagination detection, visited-URL tracking, and HTTP
error handling.  All HTTP traffic is mocked — no real network calls.

Run with:
    pytest tests/test_crawler.py -v
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.crawler import CrawlStats, WebCrawler

# ---------------------------------------------------------------------------
# Sample HTML fixtures
# ---------------------------------------------------------------------------

SAMPLE_PAGE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Quotes to Scrape</title></head>
<body>
  <div class="quote">
    <span class="text">"The world is a book."</span>
    <span class="author">Saint Augustine</span>
    <a href="/author/Saint-Augustine">About</a>
    <div class="tags">
      <a href="/tag/reading/">reading</a>
      <a href="/tag/books/">books</a>
    </div>
  </div>
  <nav>
    <ul class="pager">
      <li class="next"><a href="/page/2/">Next</a></li>
    </ul>
  </nav>
</body>
</html>
"""

SAMPLE_PAGE_2_HTML = """
<!DOCTYPE html>
<html>
<head><title>Quotes to Scrape - Page 2</title></head>
<body>
  <div class="quote">
    <span class="text">"Life is what happens."</span>
    <span class="author">John Lennon</span>
    <a href="/author/John-Lennon">About</a>
  </div>
  <nav>
    <ul class="pager">
      <li class="previous"><a href="/page/1/">Previous</a></li>
    </ul>
  </nav>
</body>
</html>
"""

SAMPLE_AUTHOR_HTML = """
<!DOCTYPE html>
<html>
<head><title>About: Saint Augustine</title></head>
<body>
  <h3 class="author-title">Saint Augustine</h3>
  <p class="author-description">Saint Augustine was a theologian.</p>
  <a href="/">Back to quotes</a>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CrawlStats tests
# ---------------------------------------------------------------------------


class TestCrawlStats:
    """Tests for the CrawlStats helper class."""

    def test_initial_values(self) -> None:
        """CrawlStats starts with all counters at zero."""
        stats = CrawlStats()
        assert stats.pages_crawled == 0
        assert stats.pages_failed == 0
        assert stats.total_bytes == 0

    def test_elapsed_time(self) -> None:
        """elapsed() returns positive time after start_time is set."""
        stats = CrawlStats()
        stats.start_time = time.time() - 5.0
        assert stats.elapsed() >= 4.9

    def test_summary_format(self) -> None:
        """summary() returns a non-empty human-readable string."""
        stats = CrawlStats()
        stats.start_time = time.time()
        stats.pages_crawled = 10
        summary = stats.summary()
        assert "Crawled 10 pages" in summary
        assert "KB" in summary


# ---------------------------------------------------------------------------
# WebCrawler initialisation tests
# ---------------------------------------------------------------------------


class TestCrawlerInit:
    """Tests for WebCrawler initialisation."""

    def test_default_init(self) -> None:
        """Crawler initialises with correct defaults."""
        crawler = WebCrawler("https://quotes.toscrape.com")
        assert crawler.base_url == "https://quotes.toscrape.com"
        assert crawler.politeness_delay == 6.0
        assert crawler.max_retries == 3
        assert crawler.timeout == 30.0
        assert len(crawler.visited) == 0
        assert len(crawler.pages) == 0

    def test_custom_init(self) -> None:
        """Crawler accepts custom parameters."""
        crawler = WebCrawler(
            "https://example.com/",
            politeness_delay=2.0,
            max_retries=5,
            timeout=10.0,
        )
        # Trailing slash should be stripped
        assert crawler.base_url == "https://example.com"
        assert crawler.politeness_delay == 2.0
        assert crawler.max_retries == 5
        assert crawler.timeout == 10.0

    def test_user_agent_default(self) -> None:
        """Default user agent contains COMP3011."""
        crawler = WebCrawler("https://example.com")
        assert "COMP3011" in crawler.user_agent


# ---------------------------------------------------------------------------
# URL normalisation tests
# ---------------------------------------------------------------------------


class TestURLNormalisation:
    """Tests for URL normalisation and internal-link detection."""

    def test_strips_trailing_slash(self) -> None:
        """Trailing slashes are removed from URLs."""
        assert WebCrawler.normalise_url("https://example.com/page/1/") == \
            "https://example.com/page/1"

    def test_strips_fragment(self) -> None:
        """Fragment identifiers (#section) are stripped."""
        assert WebCrawler.normalise_url("https://example.com/page#top") == \
            "https://example.com/page"

    def test_strips_query_string(self) -> None:
        """Query strings (?key=val) are stripped."""
        assert WebCrawler.normalise_url("https://example.com/page?q=1") == \
            "https://example.com/page"

    def test_preserves_path(self) -> None:
        """Meaningful path components are preserved."""
        assert WebCrawler.normalise_url("https://example.com/tag/love") == \
            "https://example.com/tag/love"

    def test_internal_link_detection(self) -> None:
        """Internal links are correctly identified."""
        crawler = WebCrawler("https://quotes.toscrape.com")
        assert crawler._is_internal("https://quotes.toscrape.com/page/2") is True
        assert crawler._is_internal("https://other.com/page/2") is False


# ---------------------------------------------------------------------------
# Link extraction tests
# ---------------------------------------------------------------------------


class TestLinkExtraction:
    """Tests for extracting and filtering links from HTML."""

    def test_extracts_pagination_links(self) -> None:
        """Pagination links (/page/N/) are extracted."""
        crawler = WebCrawler("https://quotes.toscrape.com")
        links = crawler.extract_links(SAMPLE_PAGE_HTML, "https://quotes.toscrape.com/")
        urls = [l for l in links if "/page/" in l]
        assert len(urls) > 0
        assert any("/page/2" in u for u in urls)

    def test_extracts_tag_links(self) -> None:
        """Tag links (/tag/xxx/) are extracted."""
        crawler = WebCrawler("https://quotes.toscrape.com")
        links = crawler.extract_links(SAMPLE_PAGE_HTML, "https://quotes.toscrape.com/")
        tag_links = [l for l in links if "/tag/" in l]
        assert len(tag_links) >= 2

    def test_extracts_author_links(self) -> None:
        """Author links (/author/xxx/) are extracted."""
        crawler = WebCrawler("https://quotes.toscrape.com")
        links = crawler.extract_links(SAMPLE_PAGE_HTML, "https://quotes.toscrape.com/")
        author_links = [l for l in links if "/author/" in l]
        assert len(author_links) >= 1

    def test_skips_external_links(self) -> None:
        """External links are not extracted."""
        html = '<html><body><a href="https://google.com">Google</a></body></html>'
        crawler = WebCrawler("https://quotes.toscrape.com")
        links = crawler.extract_links(html, "https://quotes.toscrape.com/")
        assert len(links) == 0

    def test_skips_already_visited(self) -> None:
        """Links that have already been visited are excluded."""
        crawler = WebCrawler("https://quotes.toscrape.com")
        crawler.visited.add("https://quotes.toscrape.com/page/2")
        links = crawler.extract_links(SAMPLE_PAGE_HTML, "https://quotes.toscrape.com/")
        assert "https://quotes.toscrape.com/page/2" not in links

    def test_handles_empty_html(self) -> None:
        """Empty HTML produces no links."""
        crawler = WebCrawler("https://quotes.toscrape.com")
        links = crawler.extract_links("", "https://quotes.toscrape.com/")
        assert links == []


# ---------------------------------------------------------------------------
# Visited URL tracking tests
# ---------------------------------------------------------------------------


class TestVisitedTracking:
    """Tests for duplicate URL avoidance."""

    @patch("src.crawler.WebCrawler.fetch")
    @patch("src.crawler.WebCrawler.is_allowed", return_value=True)
    def test_no_duplicate_visits(self, mock_allowed: MagicMock, mock_fetch: MagicMock) -> None:
        """Each URL is visited at most once during a crawl."""
        # Return the same HTML for any URL, with no new links
        mock_fetch.return_value = "<html><body>Hello</body></html>"

        crawler = WebCrawler("https://quotes.toscrape.com", politeness_delay=0)
        pages = crawler.crawl()

        # The start URL should be fetched exactly once
        assert mock_fetch.call_count == 1
        assert len(pages) == 1


# ---------------------------------------------------------------------------
# Politeness delay tests
# ---------------------------------------------------------------------------


class TestPolitenessDelay:
    """Tests for the politeness delay between requests."""

    def test_wait_politeness_sleeps(self) -> None:
        """_wait_politeness sleeps when called too soon after last request."""
        crawler = WebCrawler("https://example.com", politeness_delay=1.0)
        crawler._last_request_time = time.time()

        start = time.time()
        crawler._wait_politeness()
        elapsed = time.time() - start

        # Should have waited close to 1 second
        assert elapsed >= 0.9

    def test_no_wait_when_enough_time_passed(self) -> None:
        """_wait_politeness does not sleep if enough time has passed."""
        crawler = WebCrawler("https://example.com", politeness_delay=1.0)
        crawler._last_request_time = time.time() - 5.0  # 5 seconds ago

        start = time.time()
        crawler._wait_politeness()
        elapsed = time.time() - start

        assert elapsed < 0.1


# ---------------------------------------------------------------------------
# HTTP error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for HTTP error handling with mocked responses."""

    @patch("src.crawler.requests.get")
    def test_handles_404(self, mock_get: MagicMock) -> None:
        """A 404 response returns None without retrying."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response

        crawler = WebCrawler("https://example.com", politeness_delay=0)
        result = crawler.fetch("https://example.com/nonexistent")

        assert result is None
        # 404 should not be retried
        assert mock_get.call_count == 1

    @patch("src.crawler.requests.get")
    def test_handles_timeout(self, mock_get: MagicMock) -> None:
        """Timeouts are retried up to max_retries."""
        mock_get.side_effect = requests.exceptions.Timeout()

        crawler = WebCrawler("https://example.com", politeness_delay=0, max_retries=2)
        result = crawler.fetch("https://example.com/slow")

        assert result is None
        assert mock_get.call_count == 2

    @patch("src.crawler.requests.get")
    def test_handles_connection_error(self, mock_get: MagicMock) -> None:
        """Connection errors are retried."""
        mock_get.side_effect = requests.exceptions.ConnectionError()

        crawler = WebCrawler("https://example.com", politeness_delay=0, max_retries=2)
        result = crawler.fetch("https://example.com/down")

        assert result is None
        assert mock_get.call_count == 2

    @patch("src.crawler.requests.get")
    def test_successful_fetch(self, mock_get: MagicMock) -> None:
        """A successful response returns the response text."""
        mock_response = MagicMock()
        mock_response.text = "<html>Hello</html>"
        mock_response.content = b"<html>Hello</html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        crawler = WebCrawler("https://example.com", politeness_delay=0)
        result = crawler.fetch("https://example.com/page")

        assert result == "<html>Hello</html>"
        assert mock_get.call_count == 1

    @patch("src.crawler.requests.get")
    def test_retry_then_success(self, mock_get: MagicMock) -> None:
        """A transient failure followed by success returns the page."""
        fail_response = MagicMock()
        fail_response.status_code = 500
        fail_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=fail_response
        )

        ok_response = MagicMock()
        ok_response.text = "<html>OK</html>"
        ok_response.content = b"<html>OK</html>"
        ok_response.raise_for_status = MagicMock()

        mock_get.side_effect = [fail_response, ok_response]

        crawler = WebCrawler("https://example.com", politeness_delay=0, max_retries=3)
        result = crawler.fetch("https://example.com/flaky")

        assert result == "<html>OK</html>"
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Pagination detection tests
# ---------------------------------------------------------------------------


class TestPaginationDetection:
    """Tests for the _should_follow path filter."""

    def test_follows_page_paths(self) -> None:
        """Pagination paths are accepted."""
        crawler = WebCrawler("https://example.com")
        assert crawler._should_follow("/page/1/") is True
        assert crawler._should_follow("/page/10/") is True
        assert crawler._should_follow("/page/2") is True

    def test_follows_tag_paths(self) -> None:
        """Tag listing paths are accepted."""
        crawler = WebCrawler("https://example.com")
        assert crawler._should_follow("/tag/love/") is True
        assert crawler._should_follow("/tag/life") is True

    def test_follows_author_paths(self) -> None:
        """Author bio paths are accepted."""
        crawler = WebCrawler("https://example.com")
        assert crawler._should_follow("/author/Einstein/") is True
        assert crawler._should_follow("/author/J-K-Rowling") is True

    def test_follows_homepage(self) -> None:
        """The homepage path is accepted."""
        crawler = WebCrawler("https://example.com")
        assert crawler._should_follow("/") is True

    def test_rejects_login_paths(self) -> None:
        """Login and API paths are rejected."""
        crawler = WebCrawler("https://example.com")
        assert crawler._should_follow("/login") is False
        assert crawler._should_follow("/api/data") is False


# ---------------------------------------------------------------------------
# robots.txt tests
# ---------------------------------------------------------------------------


class TestRobotsTxt:
    """Tests for robots.txt handling."""

    @patch("src.crawler.RobotFileParser")
    def test_is_allowed_checks_robots(self, mock_parser_class: MagicMock) -> None:
        """is_allowed delegates to RobotFileParser.can_fetch."""
        mock_parser = MagicMock()
        mock_parser.can_fetch.return_value = True
        mock_parser_class.return_value = mock_parser

        crawler = WebCrawler("https://example.com")
        result = crawler.is_allowed("https://example.com/page/1")

        assert result is True
        mock_parser.can_fetch.assert_called_once()

    @patch("src.crawler.RobotFileParser")
    def test_blocked_url(self, mock_parser_class: MagicMock) -> None:
        """A URL blocked by robots.txt returns False."""
        mock_parser = MagicMock()
        mock_parser.can_fetch.return_value = False
        mock_parser_class.return_value = mock_parser

        crawler = WebCrawler("https://example.com")
        result = crawler.is_allowed("https://example.com/secret")

        assert result is False


# ---------------------------------------------------------------------------
# Full crawl integration test (mocked HTTP)
# ---------------------------------------------------------------------------


class TestFullCrawl:
    """Integration test for the full crawl loop with mocked HTTP."""

    @patch("src.crawler.WebCrawler._load_robots_txt")
    @patch("src.crawler.requests.get")
    def test_crawl_discovers_multiple_pages(
        self, mock_get: MagicMock, mock_robots: MagicMock
    ) -> None:
        """The crawler discovers and fetches linked pages."""
        # Map URLs to their mock responses
        url_map = {
            "https://quotes.toscrape.com": SAMPLE_PAGE_HTML,
            "https://quotes.toscrape.com/page/2": SAMPLE_PAGE_2_HTML,
            "https://quotes.toscrape.com/tag/reading": "<html><body>tag page</body></html>",
            "https://quotes.toscrape.com/tag/books": "<html><body>tag page</body></html>",
            "https://quotes.toscrape.com/author/Saint-Augustine": SAMPLE_AUTHOR_HTML,
        }

        def side_effect(url: str, **kwargs):
            mock_resp = MagicMock()
            # Normalise (strip trailing slash) for lookup
            key = url.rstrip("/")
            if key in url_map:
                mock_resp.text = url_map[key]
                mock_resp.content = url_map[key].encode()
                mock_resp.raise_for_status = MagicMock()
            else:
                mock_resp.status_code = 404
                mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
                    response=mock_resp
                )
            return mock_resp

        mock_get.side_effect = side_effect

        crawler = WebCrawler("https://quotes.toscrape.com", politeness_delay=0)
        # Make is_allowed always return True since we mocked _load_robots_txt
        crawler._robots = MagicMock()
        crawler._robots.can_fetch.return_value = True

        pages = crawler.crawl()

        # Should have discovered at least the home page and page 2
        assert len(pages) >= 2
        assert "https://quotes.toscrape.com" in pages
