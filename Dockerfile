FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    PY=python

WORKDIR /app

# Dependencies first so this layer is cached until pyproject.toml or src changes.
COPY pyproject.toml ./
COPY src ./src
RUN pip install -e ".[dev]"

COPY tests ./tests
COPY scripts ./scripts

# The image carries no dataset: tests that need it skip. Mount a cache to run them for real:
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" eeg-moabb
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
