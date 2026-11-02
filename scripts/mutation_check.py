"""Prove the leakage tests are not vacuous: run them with the audit switched off.

With every audit function replaced by a no-op, each deliberately-leaky test must FAIL (the
leak now goes unnoticed) while the clean-pipeline tests must still PASS. Writes
results/leakage_mutation_check.json; exits non-zero if any leak test survives the bypass.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

import eeg.evaluate as evaluate
import eeg.leakage as leakage

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "tests" / "test_leakage_catches_leak.py"
CLEAN_PREFIX = "test_real_", "test_watcher_is_transparent"  # these must keep passing


class Outcomes:
    def __init__(self):
        self.results = {}

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when == "setup" and report.failed):
            self.results[report.nodeid.split("::", 1)[1]] = report.outcome


def main() -> int:
    noop = lambda *a, **k: None  # noqa: E731
    leakage.audit_split = leakage.audit_splits = evaluate.audit_splits = noop
    leakage.FitWatcher.check = noop

    outcomes = Outcomes()
    with contextlib.redirect_stdout(io.StringIO()):
        pytest.main([str(TEST_FILE), "--rootdir", str(ROOT), "-c", str(ROOT / "pyproject.toml"),
                     "-p", "no:cacheprovider", "-q", "--tb=no"], plugins=[outcomes])

    is_clean = lambda name: name.startswith(CLEAN_PREFIX)  # noqa: E731
    leak_tests = {n: o for n, o in outcomes.results.items() if not is_clean(n)}
    clean_tests = {n: o for n, o in outcomes.results.items() if is_clean(n)}
    summary = {
        "audit_disabled": True,
        "leak_tests_total": len(leak_tests),
        "leak_tests_failed_as_required": sum(o == "failed" for o in leak_tests.values()),
        "clean_tests_total": len(clean_tests),
        "clean_tests_passed": sum(o == "passed" for o in clean_tests.values()),
        "outcomes": outcomes.results,
    }
    summary["ok"] = (
        summary["leak_tests_total"] > 0
        and summary["leak_tests_failed_as_required"] == summary["leak_tests_total"]
        and summary["clean_tests_passed"] == summary["clean_tests_total"] > 0
    )
    out = ROOT / "results" / "leakage_mutation_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"audit disabled: {summary['leak_tests_failed_as_required']}/{summary['leak_tests_total']} leak tests "
          f"failed as required, {summary['clean_tests_passed']}/{summary['clean_tests_total']} clean tests passed")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
