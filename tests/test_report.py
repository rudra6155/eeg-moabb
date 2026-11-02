"""The report generator's two safeguards: claims are re-verified, and hand-typed numbers are caught."""

from __future__ import annotations

from eeg.report import Ledger, undeclared_numbers


def test_claim_is_verified_only_while_artifact_text_and_document_agree(tmp_path):
    artifact = tmp_path / "value.txt"
    artifact.write_text("0.123")
    ledger = Ledger()
    read = lambda: artifact.read_text()  # noqa: E731
    text = ledger.cite("DOC.md", "a value", artifact, "make x", read)
    claim = ledger.claims[0]

    assert ledger.status(claim, {"DOC.md": f"the value is {text}"}) == "VERIFIED"
    assert ledger.status(claim, {"DOC.md": "the value is 0.999"}) == "PENDING"  # document drifted
    artifact.write_text("0.456")
    assert ledger.status(claim, {"DOC.md": f"the value is {text}"}) == "PENDING"  # artifact drifted
    artifact.unlink()
    assert ledger.status(claim, {"DOC.md": f"the value is {text}"}) == "BLOCKED"  # artifact gone


def test_missing_artifact_at_render_time_is_blocked_not_a_crash(tmp_path):
    missing = tmp_path / "nope.csv"

    def read():
        return missing.read_text()

    ledger = Ledger()
    text = ledger.cite("DOC.md", "x", missing, "make x", read)
    assert "BLOCKED" in text
    assert ledger.status(ledger.claims[0], {"DOC.md": text}) == "BLOCKED"


def test_a_hand_typed_decimal_is_caught_but_versions_urls_and_cited_numbers_are_not(tmp_path):
    artifact = tmp_path / "v.txt"
    artifact.write_text("0.618")
    ledger = Ledger()
    cited = ledger.cite("DOC.md", "acc", artifact, "make x", artifact.read_text)
    doc = (f"Accuracy was {cited}, on MOABB 1.7.2, see https://example.org/v2.5/x and doi 10.3389/fnins.2012.00055, "
           f"but I typed 0.731 by hand.")
    assert undeclared_numbers(doc, ledger.claims, "DOC.md") == ["0.731"]
