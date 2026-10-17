"""Guard the boundary between research checklists and adjudicated violations."""

from copy import deepcopy

import pytest

from vision.pipeline import describe_rule, rules_for
from vision.scenarios import emitted_types, load, observation_prompt, validate


def validation_inputs():
    cases = load()
    ids = {c["type_id"] for c in cases if c["type_id"]}
    points = {}
    for case in cases:
        for act, point in case["references"]:
            points.setdefault(act, set()).add(point)
    return {"version": 1, "purpose": "development_checklist_only", "scenarios": cases}, ids, points


def test_research_cannot_gain_generic_type_or_invented_law():
    document, ids, points = validation_inputs()
    for modify, match in [
        (lambda c: c.update(type_id="T0"), "generic T0"),
        (lambda c: c.update(references=[["231", "999999"]]), "reference absent"),
        (lambda c: c.update(references=[["foreign", "20"]]), "reference absent"),
        (lambda c: c.update(counterexamples=[]), "counterexamples"),
    ]:
        bad = deepcopy(document)
        modify(bad["scenarios"][0])
        with pytest.raises(ValueError, match=match):
            validate(bad, ids | {"T0"}, points, emitted_types())
    bad = deepcopy(document)
    candidate = next(c for c in bad["scenarios"] if c["coverage"] == "candidate")
    candidate["type_id"] = "T1"
    with pytest.raises(ValueError, match="cannot advertise"):
        validate(bad, ids, points, emitted_types())
    bad = deepcopy(document)
    catalog_only = next(c for c in bad["scenarios"] if c["coverage"] == "catalog_only")
    catalog_only["coverage"] = "family_rule"
    with pytest.raises(ValueError, match="coverage disagrees"):
        validate(bad, ids, points, emitted_types())


def test_prompt_stays_observational_and_rejects_hidden_facts():
    for case in load():
        if case["coverage"] == "not_visual":
            with pytest.raises(ValueError, match="non-video facts"):
                observation_prompt(case)
        else:
            prompt = observation_prompt(case)
            assert "observed / contradicted / unknown" in prompt
            assert all(text in prompt for text in case["counterexamples"])
            assert "не готовый вывод" in prompt
            assert "потеря трека" in prompt


def test_legal_basis_is_not_a_categorical_event_conclusion():
    for type_id in ("T1", "T9", "T13"):
        _, rules, _ = rules_for(type_id)
        result = describe_rule(rules)
        assert rules[0].ref in result
        assert "Нарушен" not in result
        assert result.startswith("Признаки возможного нарушения" if rules[0].basis == "direct"
                                 else "Может противоречить")
    with pytest.raises(KeyError):
        rules_for("UNSEEN_AUTO_FALLBACK")
