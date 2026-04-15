import os

# All paths are overridable via environment variables so Docker bind-mounts
# can point to the correct locations without changing code.
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "data")
PROCESSED_DATA_DIR = os.environ.get("PROCESSED_DATA_DIR", "processed_data")
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

CV_DIR = "cvs"
JOB_DIR = "job_descriptions"

TOP_N = int(os.environ.get("TOP_N", "3"))
