"""
Main CLI Module for COMP3011 CW2 Search Engine.

Provides an interactive command-line interface with the following commands:
    - ``build``         — Crawl the website, build and save the inverted index.
    - ``load``          — Load a previously built index from disk.
    - ``print <word>``  — Display the inverted-index entry for a word.
    - ``find <query>``  — Search for pages matching the query terms.
    - ``stats``         — Show index statistics.
    - ``help``          — Display available commands.
    - ``quit`` / ``exit`` — Exit the program.

Usage:
    python -m src.main
"""

from __future__ import annotations

import logging
import sys

from src.search import SearchEngine

# ---------------------------------------------------------------------------
# Configure logging for the whole application
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def print_help() -> None:
    """Print the list of available commands."""
    help_text = """
Available Commands:
  build             Crawl quotes.toscrape.com, build inverted index, save to disk.
  load              Load a previously built index from data/index.json.gz.
  print <word>      Display the inverted-index entry for a word.
  find <query>      Search for pages matching query terms (AND logic for multi-word).
  stats             Show index statistics (terms, documents, build time).
  help              Show this help message.
  quit / exit       Exit the program.
"""
    print(help_text.strip())


def print_stats(engine: SearchEngine) -> None:
    """Print summary statistics about the loaded index."""
    idx = engine.index
    if not idx.documents:
        print("No index loaded. Use 'build' or 'load' first.")
        return

    print(f"Index Statistics:")
    print(f"  Terms:      {len(idx.index)}")
    print(f"  Documents:  {len(idx.documents)}")
    print(f"  Build time: {idx.build_time:.2f}s")

    if idx.index:
        avg_df = sum(len(p) for p in idx.index.values()) / len(idx.index)
        print(f"  Avg doc freq per term: {avg_df:.2f}")

    if idx.documents:
        avg_wc = sum(d.word_count for d in idx.documents.values()) / len(idx.documents)
        print(f"  Avg words per document: {avg_wc:.1f}")


def main() -> None:
    """
    Interactive CLI shell for the search engine.

    Reads commands from stdin in a REPL loop.  Type ``help`` for usage.
    """
    print("=" * 60)
    print("  COMP3011 CW2 — Search Engine Tool")
    print("  Target: https://quotes.toscrape.com/")
    print("  Type 'help' for available commands.")
    print("=" * 60)
    print()

    engine = SearchEngine()

    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not cmd:
            continue

        # ---- build ----
        if cmd == "build":
            try:
                engine.build()
            except Exception as exc:
                print(f"[error] Build failed: {exc}")

        # ---- load ----
        elif cmd == "load":
            try:
                engine.load()
            except FileNotFoundError:
                print("[error] Index file not found. Run 'build' first.")
            except ValueError as exc:
                print(f"[error] Index file corrupt: {exc}")

        # ---- print <word> ----
        elif cmd == "print" or cmd.startswith("print "):
            word = cmd[6:].strip() if cmd.startswith("print ") else ""
            if not word:
                print("Usage: print <word>")
            else:
                output = engine.print_word(word)
                print(output)

        # ---- find <query> ----
        elif cmd == "find" or cmd.startswith("find "):
            query = cmd[5:].strip() if cmd.startswith("find ") else ""
            if not query:
                print("Usage: find <query>")
            else:
                results = engine.find(query)
                formatted = engine.format_results(results, query)
                print(formatted)

        # ---- stats ----
        elif cmd == "stats":
            print_stats(engine)

        # ---- help ----
        elif cmd == "help":
            print_help()

        # ---- quit / exit ----
        elif cmd in ("quit", "exit"):
            print("Goodbye!")
            break

        else:
            print(f"Unknown command: '{cmd}'. Type 'help' for available commands.")


if __name__ == "__main__":
    main()
