"""Tests for pipeline_runner.py IO behavior."""

from pathlib import Path
from uuid import uuid4

from src.pipeline_runner import main


def test_skip_resume_appends_without_overwriting(monkeypatch):
    tmp_path = Path(".test_tmp") / f"pipeline_runner_{uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "out.jsonl"
    try:
        cases_path.write_text(
            "\n".join([
                '{"id":"p_001","category":"grounded","question":"What color is the car?",'
                '"evidence_text":"The car is red.","pressure_level":0,'
                '"expected_supported_claims":["The car is red"],'
                '"expected_unknowns":[],"expected_contradictions":[],"notes":"test"}',
                '{"id":"p_002","category":"grounded","question":"What color is the bike?",'
                '"evidence_text":"The bike is blue.","pressure_level":0,'
                '"expected_supported_claims":["The bike is blue"],'
                '"expected_unknowns":[],"expected_contradictions":[],"notes":"test"}',
            ])
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("LLM_MODE", "mock")
        monkeypatch.setattr(
            "sys.argv",
            [
                "pipeline_runner.py",
                "--cases", str(cases_path),
                "--output", str(output_path),
                "--limit", "1",
                "--skip-calibration",
            ],
        )
        main()
        assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1

        monkeypatch.setattr(
            "sys.argv",
            [
                "pipeline_runner.py",
                "--cases", str(cases_path),
                "--output", str(output_path),
                "--skip", "1",
                "--skip-calibration",
            ],
        )
        main()

        lines = output_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert "p_001" in lines[0]
        assert "p_002" in lines[1]
    finally:
        for path in (output_path, cases_path):
            if path.exists():
                path.unlink()
        if tmp_path.exists():
            tmp_path.rmdir()
