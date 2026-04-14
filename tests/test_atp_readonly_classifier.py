from __future__ import annotations

from kuzualchemy.atp_integration import ATPIntegration


def test_checkpoint_is_not_classified_as_readonly() -> None:
    integration = object.__new__(ATPIntegration)

    assert integration._is_readonly_query("CHECKPOINT;") is False
    assert integration._is_readonly_query("MATCH (n) RETURN n") is True
