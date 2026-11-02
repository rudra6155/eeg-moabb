import pytest

from eeg.data import data_available, load_subject

_loaded = {}


@pytest.fixture(scope="session")
def real_subject():
    """Load a real subject, or skip the test when the dataset has not been downloaded.

    Skipping (never downloading) keeps `make test` offline; CI restores the data cache
    and so runs these tests for real.
    """

    def get(subject: int):
        ok, why = data_available([subject])
        if not ok:
            pytest.skip(why)
        if subject not in _loaded:
            _loaded[subject] = load_subject(subject)
        return _loaded[subject]

    return get
