from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.app.ui_pipeline import find_fitabase_raw_dir, redact_secrets


def test_redact_secrets_masks_service_role_key() -> None:
    line = "SUPABASE_SERVICE_ROLE_KEY=super-secret-value"

    assert redact_secrets(line) == "SUPABASE_SERVICE_ROLE_KEY=[REDACTED]"


def test_find_fitabase_raw_dir_detects_daily_activity_file(tmp_path: Path) -> None:
    raw_dir = tmp_path / "Fitabase Data"
    raw_dir.mkdir()
    activity_file = raw_dir / "dailyActivity_merged.csv"
    activity_file.write_text("header1,header2\n1,2\n", encoding="utf-8")

    assert find_fitabase_raw_dir(tmp_path) == raw_dir
