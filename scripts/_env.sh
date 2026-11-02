# Sourced by the other scripts: run from the repo root and pick a python.
# A preset PY (for example in the Docker image, which has no .venv) wins.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -z "${PY:-}" ]; then
    if [ -x .venv/Scripts/python.exe ]; then
        PY=.venv/Scripts/python.exe
    else
        PY=.venv/bin/python
    fi
fi
