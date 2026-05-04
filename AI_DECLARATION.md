# GenAI Use Declaration

This document is a written backup of the GenAI critical evaluation that
also forms part of the video demonstration submitted for COMP3011 CW2.
It is written in English and submitted alongside the project repository.

## Tools used

- **Anthropic Claude (Sonnet and Opus)**, accessed through the Claude Code
  command-line agent inside the project's working directory.
- No other LLMs, code-completion plugins, or third-party AI services were
  used in the production of this submission.

## How AI was used

I treated the assistant as a pair-programming collaborator that I drove
explicitly, rather than as an autonomous code generator. Concretely:

- **Project scaffolding and boilerplate.** I asked the assistant to
  produce the initial layout for the four `src/` modules, the matching
  test files, `requirements.txt`, and the project README. I then read
  every file before committing it.
- **Drafting docstrings and inline comments.** Module headers, complexity
  annotations and Google-style docstrings were drafted with assistance
  and then edited for accuracy against the actual code paths.
- **Iterating on the inverted index design.** I sketched the required
  data shape (`word -> {url -> {frequency, positions, tf}}`), asked the
  assistant for a smoothed-IDF formulation, and discussed the trade-off
  between storing raw text for snippet generation versus keeping the
  index file small. The final choices (smoothed IDF
  `log((N+1)/(df+1)) + 1`, gzip JSON with an MD5 checksum) are decisions
  I made and can defend.
- **Test design.** The unit tests, including the use of `pytest`
  monkeypatching to fake HTTP responses, were planned with the
  assistant. I added the integration cases that exercise AND-logic
  search, snippet extraction edge cases, and the CLI REPL.
- **Pre-submission audit.** During the final pre-submission pass I asked
  the assistant to audit the project against the marking rubric and to
  flag any missing artefacts. The audit is included as `AUDIT.md`.

## Where AI suggestions needed correction

These are real corrections I made to AI-generated material during the
project. They are described in general terms because chat transcripts
cannot be exported from the tool; however, every change is reflected in
the Git history and can be traced through `git log` and `git diff`.

- **Politeness window.** Early scaffolding used a delay shorter than the
  6 second floor required by the brief. I corrected the default to
  `6.0` and verified that the backoff path multiplies the politeness
  delay rather than overriding it (`src/crawler.py:223-280`).
- **Stop-word filtering.** The first draft of the indexer treated stop
  words as ordinary tokens, which inflated the index and produced poor
  ranking. I added a curated stop-word set and made tokenisation skip
  stop words entirely (`src/indexer.py:43-62`, `src/indexer.py:243-245`).
- **Search semantics.** An initial draft used OR semantics for
  multi-word queries. The brief requires AND, so I rewrote the query
  path to intersect candidate sets and to short-circuit when any term
  is missing from the index (`src/indexer.py:475-486`).
- **Index integrity on load.** I added the MD5 checksum check on `load`
  after observing that a corrupted gzip file would otherwise propagate
  silently (`src/indexer.py:283-286`, `src/indexer.py:376-386`).
- **CLI argument parsing.** The original REPL split commands on
  whitespace and treated everything after the verb as a single flag.
  I rewrote the dispatcher to keep the full query string intact for
  `print` and `find` (`src/main.py:112-128`).
- **Failing test in the audit.** A late integration test
  (`tests/test_main.py::TestMainCLI::test_load_missing_file`) was
  written assuming the index file would be absent at test time. Once
  the real index was committed to `data/`, the test silently started
  passing the load and failing the assertion. I fixed it by
  monkeypatching `SearchEngine.DEFAULT_INDEX_PATH` to a `tmp_path` that
  does not exist.

## What I did manually without AI

- All design decisions about scope (which CLI commands to expose, which
  TF-IDF variant to ship, how to balance index size against snippet
  quality) and all final reviews of every file before commit.
- The Git workflow itself, including commit message phrasing, commit
  granularity, and the decision to keep one logical change per commit.
- Running the live crawl that produced `data/index.json.gz` and
  verifying that the politeness window was respected (`commit 364dceb`).
- Writing the marking-rubric audit (`AUDIT.md`) and validating every
  file:line citation by reading the file rather than trusting the
  assistant's summary.

## Critical evaluation

Working with the assistant compressed the time required for
boilerplate, documentation, and test scaffolding, which let me spend
more time on the requirements that earn marks: search quality,
politeness compliance, and test coverage. The main risks I had to guard
against were three: confidently wrong claims about library behaviour
(checked by running the code), omissions in security or compliance
details (checked against the brief), and a tendency to produce overly
generic tests that pass without exercising the underlying logic
(checked by reading each test and confirming it failed when I broke
the production code).

I am responsible for every line in the submission. I have read each
file in `src/` and `tests/`, can explain the rationale for each design
decision, and can run the project, the test suite and the crawl on
demand during the video demonstration.

## Academic integrity statement

The use of GenAI tools described above complies with the University of
Leeds policy on the responsible use of generative AI in assessed work
for COMP3011. No part of this declaration, the audit, or the supporting
documents has been fabricated.
