import pandas as pd
import pytest

from data_anonymizer import (
    AnonymizationPlan,
    Anonymizer,
    ColumnAction,
    Fidelity,
    PlanError,
    Protection,
    ProtectionLevel,
    StatisticalEffect,
    Technique,
    TechniqueFamily,
)
from data_anonymizer.taxonomy import ColumnKind


def test_statistical_plan_pseudonymizes_identifiers_and_keeps_statistics(people):
    anonymizer = Anonymizer(secret_key="key", seed=1)
    plan = anonymizer.propose(people, fidelity=Fidelity.STATISTICAL)

    result = anonymizer.apply(people, plan)
    data = result.data

    assert plan.action_for("email").technique == "surrogate"
    assert plan.action_for("basket_value") is None
    assert not data["email"].isin(people["email"]).any()
    assert not data["full_name"].isin(people["full_name"]).any()
    assert data["customer_id"].nunique() == people["customer_id"].nunique()
    assert data["basket_value"].equals(people["basket_value"])
    assert sorted(data["salary"]) == sorted(people["salary"])
    assert result.utility.for_column("age").mean_change < 0.01
    assert result.utility.max_correlation_change < 0.1
    assert "<EMAIL>" in data["notes"].iloc[0]


def test_privacy_pseudonymization_hashes_identifiers_and_reaches_k(people):
    anonymizer = Anonymizer(secret_key="key", seed=1)
    plan = anonymizer.propose(people, fidelity=Fidelity.PRIVACY)

    result = anonymizer.apply(people, plan)

    assert plan.enforce_k_anonymity
    assert result.privacy_before.k_anonymity == 1
    assert result.privacy_after.share_below_target <= 0.01
    assert result.suppressed_cells > 0
    assert result.data["email"].str.fullmatch(r"[0-9a-f]{16}").all()
    assert result.data["age"].dropna().str.startswith("[").all()
    assert result.protection.level is ProtectionLevel.PSEUDONYMIZED
    assert result.protection.reached


@pytest.mark.parametrize("fidelity", list(Fidelity))
def test_anonymization_leaves_no_link_to_people(people, fidelity):
    anonymizer = Anonymizer(secret_key="key", seed=1)
    plan = anonymizer.propose(people, fidelity=fidelity, protection=Protection.ANONYMIZATION)

    result = anonymizer.apply(people, plan)

    assert plan.enforce_k_anonymity
    assert not any("link" in problem for problem in result.protection.problems)
    assert not result.data["email"].isin(people["email"]).any()
    assert not result.data["cpf"].isin(people["cpf"]).any()
    assert result.protection.reached == (result.protection.level is ProtectionLevel.ANONYMIZED)
    assert result.privacy_after.share_below_target <= 0.01


def test_random_surrogates_differ_for_the_same_person(people):
    repeated = pd.concat([people.head(50)] * 2, ignore_index=True)
    plan = AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        protection=Protection.ANONYMIZATION,
        actions=[
            ColumnAction(
                column="email",
                technique="surrogate",
                params={"category": "email", "consistent": False},
            )
        ],
        direct_identifiers=["email"],
    )

    emails = Anonymizer(seed=2).apply(repeated, plan).data["email"]

    assert (emails.iloc[:50].to_numpy() != emails.iloc[50:].to_numpy()).mean() > 0.9


def test_anonymization_refuses_linking_techniques(people):
    plan = AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        protection=Protection.ANONYMIZATION,
        actions=[ColumnAction(column="email", technique="keyed_hash")],
    )

    with pytest.raises(PlanError, match="keeps a link"):
        Anonymizer().apply(people, plan)


def test_an_untouched_direct_identifier_makes_the_result_identifiable(people):
    plan = AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        actions=[ColumnAction(column="email", technique="keep")],
        direct_identifiers=["email", "cpf"],
    )

    protection = Anonymizer().apply(people, plan).protection

    assert protection.level is ProtectionLevel.IDENTIFIABLE
    assert not protection.reached
    assert len([problem for problem in protection.problems if "left as it was" in problem]) == 2


def test_same_key_gives_the_same_pseudonyms_across_instances(people):
    plan = AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        actions=[ColumnAction(column="email", technique="keyed_hash")],
    )

    first = Anonymizer(secret_key="shared").apply(people, plan).data["email"]
    second = Anonymizer(secret_key="shared").apply(people, plan).data["email"]

    assert first.equals(second)


def test_plan_round_trips_through_json(people):
    plan = Anonymizer().propose(people, fidelity=Fidelity.PRIVACY)

    assert AnonymizationPlan.from_json(plan.to_json()) == plan


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (ColumnAction(column="missing", technique="redact"), "not in the table"),
        (ColumnAction(column="email", technique="teleport"), "unknown technique"),
        (
            ColumnAction(column="email", technique="redact", params={"colour": "red"}),
            "parameters",
        ),
    ],
)
def test_invalid_plans_are_rejected_before_anything_runs(people, action, message):
    plan = AnonymizationPlan(fidelity=Fidelity.STATISTICAL, actions=[action])

    with pytest.raises(PlanError, match=message):
        Anonymizer().apply(people, plan)


def test_duplicate_actions_for_one_column_are_rejected():
    with pytest.raises(ValueError, match="more than one action"):
        AnonymizationPlan(
            fidelity=Fidelity.STATISTICAL,
            actions=[
                ColumnAction(column="a", technique="redact"),
                ColumnAction(column="a", technique="suppress"),
            ],
        )


def test_extra_techniques_can_be_used_in_plans(people):
    def uppercase(values, context):
        return values.str.upper()

    shout = Technique(
        name="uppercase",
        family=TechniqueFamily.MASKING,
        effect=StatisticalEffect.PRESERVES_RELATIONSHIPS,
        kinds=frozenset({ColumnKind.TEXT}),
        function=uppercase,
        summary="Upper-case text.",
    )
    plan = AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        actions=[ColumnAction(column="full_name", technique="uppercase")],
    )

    result = Anonymizer(extra_techniques=[shout]).apply(people, plan)

    assert result.data["full_name"].equals(people["full_name"].str.upper())


def test_low_confidence_findings_ask_for_review():
    frame = pd.DataFrame({"email": ["alpha", "beta", "gamma"] * 10})

    plan = Anonymizer().propose(frame)

    assert "please review" in plan.action_for("email").reason
