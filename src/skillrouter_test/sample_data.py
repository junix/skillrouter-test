"""A tiny built-in skill pool + queries so the CLI runs with zero setup.

Includes near-duplicate / topically-overlapping skills on purpose — that is
exactly the homogeneous-pool regime the paper targets, where metadata alone is
ambiguous and the full body becomes the deciding signal.
"""

from .models import Skill

SAMPLE_SKILLS: list[Skill] = [
    Skill(
        "speech-to-text",
        "Transcribe audio/video locally with Whisper and return timestamped text.",
        "Runs faster-whisper on a local file, emits SRT/VTT with word-level timestamps. "
        "No network calls; supports diarization and language auto-detect.",
    ),
    Skill(
        "audio-transcriber",
        "General-purpose cloud transcription service for uploaded audio files.",
        "Uploads audio to a cloud API and polls for a transcript. Requires an API key; "
        "no local model, no timestamp granularity below the sentence level.",
    ),
    Skill(
        "video-subtitle-sync",
        "Synchronize subtitle timing to video playback using audio cues.",
        "Takes an existing subtitle file plus a video and re-aligns timecodes; does NOT "
        "produce a transcript from scratch.",
    ),
    Skill(
        "git-feature-branch",
        "Create a feature branch, open a PR, and wire up required status checks.",
        "Implements trunk-based git flow: branch naming, conventional commits, opens a PR "
        "via gh, and enables required CI checks before merge.",
    ),
    Skill(
        "git-history-rewrite",
        "Rewrite git history: interactive rebase, squash, and force-push safely.",
        "Wraps interactive rebase, autosquash fixups, and a guarded force-push-with-lease. "
        "For cleaning up history, not for branching workflows.",
    ),
    Skill(
        "docker-compose-up",
        "Bring up a multi-service stack with docker compose and health checks.",
        "Generates and runs a docker-compose stack, waits on healthchecks, streams logs.",
    ),
    Skill(
        "pdf-extract-tables",
        "Extract tables from a PDF into clean CSV/DataFrame output.",
        "Uses camelot/pdfplumber to detect ruled and borderless tables, returns tidy CSV.",
    ),
    Skill(
        "sql-schema-migrate",
        "Generate and apply reversible SQL schema migrations.",
        "Diffs models against the live schema, writes up/down migrations, applies in a txn.",
    ),
]

SAMPLE_QUERIES: list[str] = [
    "I have a local lecture recording and need a transcript with exact word timings, offline.",
    "Set up a pull-request workflow on my repo so every change runs CI before it can merge.",
    "Pull the financial tables out of this PDF report into a spreadsheet.",
]
