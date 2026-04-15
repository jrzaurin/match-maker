"""TF-IDF title encoder — fit on job corpus, encode job titles and CV titles."""

import re
import pickle
from pathlib import Path

from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from matchmaker.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR
from matchmaker.utils.save_load_utils import load_job_records

VECTORIZER_PATH = Path(ARTIFACTS_DIR) / "title_vectorizer.pkl"


def _normalize(text: str) -> str:
    """Lowercase and strip non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


class TitleEncoder:
    """TF-IDF encoder for job titles.

    Workflow:
        encoder = TitleEncoder()
        encoder.fit_and_save(jobs_path)   # once, offline

        encoder = TitleEncoder.load()     # at query time
        job_matrix = encoder.encode_many(job_titles)
        cv_vec     = encoder.encode_one(cv_title)
    """

    def __init__(self, vec: TfidfVectorizer | None = None) -> None:
        self._vec = vec

    def fit_and_save(self, jobs_path: str | Path | None = None) -> None:
        """Fit on job titles from the JSONL corpus and save to artifacts/."""
        path = Path(jobs_path) if jobs_path else Path(PROCESSED_DATA_DIR) / "jobs.jsonl"
        jobs = load_job_records(path)
        if not jobs:
            raise ValueError(f"No job records found at {path}")

        titles = [_normalize(j.title) for j in jobs]
        self._vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
        self._vec.fit(titles)

        VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with VECTORIZER_PATH.open("wb") as f:
            pickle.dump(self._vec, f)

        print(f"Fitted on {len(titles)} titles → saved to {VECTORIZER_PATH}")
        for t in titles:
            print(f"  {t}")

    @classmethod
    def load(cls) -> "TitleEncoder":
        """Load a previously fitted encoder from disk."""
        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"No encoder found at {VECTORIZER_PATH}. "
                "Run TitleEncoder().fit_and_save() first."
            )
        with VECTORIZER_PATH.open("rb") as f:
            vec = pickle.load(f)
        return cls(vec=vec)

    def encode_one(self, title: str) -> spmatrix:
        """Encode a single title (e.g. from an incoming CV). Returns (1, vocab) matrix."""
        self._check_fitted()
        return self._vec.transform([_normalize(title)])  # type: ignore[union-attr]

    def encode_many(self, titles: list[str]) -> spmatrix:
        """Encode a list of titles (e.g. the job-spec corpus). Returns (n, vocab) matrix."""
        self._check_fitted()
        return self._vec.transform([_normalize(t) for t in titles])  # type: ignore[union-attr]

    def _check_fitted(self) -> None:
        if self._vec is None:
            raise RuntimeError(
                "Encoder is not fitted. Call fit_and_save() or load() first."
            )


if __name__ == "__main__":
    encoder = TitleEncoder()
    encoder.fit_and_save()
