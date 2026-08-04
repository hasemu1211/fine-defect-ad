import json
import re
from pathlib import Path


HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def test_mvtec_metric_evidence_remains_fail_closed_and_public_safe():
    protocol = json.loads(Path("evidence/mvtec-metric-protocol.json").read_text())
    provenance = json.loads(Path("evidence/mvtec-metric-provenance.json").read_text())

    assert protocol["comparator"] is None
    assert protocol["official_benchmark_claim"] == "BLOCKED_MISSING_VERIFIED_PROTOCOL_PROVENANCE"
    assert protocol["ad2_server_equivalence_status"] == "UNVERIFIED"
    assert protocol["comparator_limitation"]
    assert provenance["inspection_status"] == "VERIFIED_EPHEMERAL_RETRIEVAL"
    assert provenance["retrieved_at_utc"].endswith("Z")
    for source in provenance["sources"].values():
        assert source["inspection_status"] == "VERIFIED_EPHEMERAL_RETRIEVAL"
        if "ad2_server_equivalence_status" in source:
            assert source["ad2_server_equivalence_status"] == "UNVERIFIED"
        assert source["official_page_url"].startswith("https://")
        assert source["archive_url"].startswith("https://")
        assert HEX64.fullmatch(source["archive_sha256"])
        assert source["license_marker"]
        assert source["observed_behavior"]
        for member in source["inspected_members"]:
            assert member["path"] and member["locators"]
            assert HEX64.fullmatch(member["sha256"])

    assert "checker" in protocol["code_utils_status"]
    assert protocol["official_sources"]["mvtec_ad_evaluator_v1"]["scope_limit"]
    assert all(not value.startswith(("/", "~/")) for value in _strings([protocol, provenance]))
