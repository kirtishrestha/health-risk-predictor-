"""Shared UI helpers for the Streamlit pipeline runner."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Tuple

import streamlit as st

from src.etl.pandas_metrics import find_fitbit_base_dir


REQUIRED_ENV_VARS = ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY")
REDACT_ENV_VARS = ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_DB_URL")


def env_ok() -> Tuple[bool, list[str]]:
    """Return whether required Supabase env vars are available."""

    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    return len(missing) == 0, missing


def _redact_values(text: str, values: Iterable[str]) -> str:
    redacted = text
    for value in values:
        if value:
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def redact_secrets(text: str) -> str:
    """Redact sensitive secrets from text."""

    return re.sub(r"(SUPABASE_SERVICE_ROLE_KEY=)(\S+)", r"\1[REDACTED]", text)


def redact_log_line(line: str) -> str:
    """Redact secrets from log lines before rendering."""

    env_values = [os.getenv(name, "") for name in REDACT_ENV_VARS]
    redacted = _redact_values(line, env_values)
    redacted = redact_secrets(redacted)
    redacted = re.sub(r"(SUPABASE_DB_URL=)(\S+)", r"\1[REDACTED]", redacted)
    return redacted


def find_fitabase_raw_dir(extracted_root: Path) -> Path | None:
    """Return the Fitabase raw data folder inside an extracted ZIP."""

    return find_fitbit_base_dir(extracted_root)


def initialize_upload_state(uploaded_zip) -> None:
    """Extract a newly uploaded ZIP and store metadata in session state."""

    if uploaded_zip is None:
        st.session_state.pop("uploaded_zip_name", None)
        st.session_state.pop("uploaded_zip_dir", None)
        st.session_state.pop("uploaded_zip_base_dir", None)
        st.session_state.pop("uploaded_zip_csv_count", None)
        return

    if st.session_state.get("uploaded_zip_name") == uploaded_zip.name:
        return

    tmp_dir = Path(tempfile.mkdtemp(prefix="fitbit_upload_"))
    uploaded_zip.seek(0)
    with zipfile.ZipFile(uploaded_zip) as zip_file:
        zip_file.extractall(tmp_dir)

    base_dir = find_fitbit_base_dir(tmp_dir)
    csv_count = len(list(tmp_dir.rglob("*.csv")))

    st.session_state["uploaded_zip_name"] = uploaded_zip.name
    st.session_state["uploaded_zip_dir"] = str(tmp_dir)
    st.session_state["uploaded_zip_base_dir"] = str(base_dir) if base_dir else None
    st.session_state["uploaded_zip_csv_count"] = csv_count


def stream_command_logs(command: list[str], log_key: str) -> Tuple[bool, str]:
    """Run a CLI command and stream logs into the UI with redaction."""

    st.session_state[log_key] = ""
    placeholder = st.empty()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: list[str] = []
    if process.stdout:
        for line in process.stdout:
            safe_line = redact_log_line(line.rstrip())
            output_lines.append(safe_line)
            st.session_state[log_key] = "\n".join(output_lines)
            placeholder.code(st.session_state[log_key])

    return_code = process.wait()
    return return_code == 0, st.session_state[log_key]


def run_command(command: list[str]) -> bool:
    """Run a CLI command and stream logs only to the terminal."""

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0
