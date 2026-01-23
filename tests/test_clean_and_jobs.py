from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import app


def _sample_imdb_csv_path() -> Path:
    return Path("data/raw/messy_IMDB_dataset.csv")


def test_clean_basic_creates_job_and_artifacts() -> None:
    client = TestClient(app)

    csv_path = _sample_imdb_csv_path()
    assert csv_path.exists(), f"missing test csv at {csv_path}"

    with csv_path.open("rb") as f:
        resp = client.post(
            "/clean/basic",
            files={"file": ("messy_IMDB_dataset.csv", f, "text/csv")},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert "job_id" in body
    assert body["cleaning_mode"] == "basic"
    assert "artifacts" in body
    assert "cleaned_csv" in body["artifacts"]
    assert "report_json" in body["artifacts"]

    cleaned_path = Path(body["artifacts"]["cleaned_csv"])
    report_path = Path(body["artifacts"]["report_json"])

    assert cleaned_path.exists(), f"expected cleaned csv at {cleaned_path}"
    assert report_path.exists(), f"expected report json at {report_path}"


def test_job_endpoints_return_report_and_csv() -> None:
    client = TestClient(app)

    csv_path = _sample_imdb_csv_path()
    assert csv_path.exists(), f"missing test csv at {csv_path}"

    # create a job
    with csv_path.open("rb") as f:
        create = client.post(
            "/clean/basic",
            files={"file": ("messy_IMDB_dataset.csv", f, "text/csv")},
        )

    assert create.status_code == 200, create.text
    job_id = create.json()["job_id"]

    # fetch report
    report = client.get(f"/jobs/{job_id}")
    assert report.status_code == 200, report.text
    report_json = report.json()
    assert report_json.get("job_id") == job_id
    assert report_json.get("cleaning_mode") == "basic"

    # download cleaned csv
    dl = client.get(f"/jobs/{job_id}/cleaned.csv")
    assert dl.status_code == 200, dl.text
    assert "text/csv" in dl.headers.get("content-type", "")
    assert len(dl.content) > 0