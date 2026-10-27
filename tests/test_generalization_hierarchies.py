import pandas as pd
import pytest
from pydantic import ValidationError

from oblitus import AnonymizationPlan, Anonymizer, ColumnAction, Fidelity, PlanError, Protection
from oblitus.generalization_hierarchies import GeneralizationHierarchy

REGIONS = GeneralizationHierarchy(
    levels=[
        {"Santos": "Baixada", "Guarujá": "Baixada", "Campinas": "Interior", "Bauru": "Interior"},
        {"Santos": "SP", "Guarujá": "SP", "Campinas": "SP", "Bauru": "SP", "Niterói": "RJ"},
    ]
)


def test_a_value_climbs_one_level_at_a_time_up_to_the_top():
    assert REGIONS.ancestor("Santos", 0) == "Santos"
    assert REGIONS.ancestor("Santos", 1) == "Baixada"
    assert REGIONS.ancestor("Santos", 2) == "SP"
    assert REGIONS.ancestor("Santos", 3) == "*"


def test_a_value_the_mapping_does_not_list_stays_as_it_is():
    assert REGIONS.ancestor("Niterói", 1) == "Niterói"
    assert REGIONS.ancestor("Niterói", 2) == "RJ"


def test_the_lowest_common_level_is_where_all_values_share_a_stand_in():
    assert REGIONS.lowest_common_level(["Santos"]) == 0
    assert REGIONS.lowest_common_level(["Santos", "Guarujá"]) == 1
    assert REGIONS.lowest_common_level(["Santos", "Bauru"]) == 2
    assert REGIONS.lowest_common_level(["Santos", "Niterói"]) == 3


def test_the_cep_preset_keeps_the_leading_digits_and_ignores_the_dash():
    cep = GeneralizationHierarchy.model_validate({"preset": "cep"})

    assert cep.ancestor("01310-100", 1) == "01310***"
    assert cep.ancestor("01310-100", 2) == "013*****"
    assert cep.ancestor("01310-100", 4) == "0*******"
    assert cep.ancestor("01310-100", 5) == "*"


@pytest.mark.parametrize(
    "definition",
    [
        {},
        {"levels": [{"a": "b"}], "prefix_lengths": [2]},
        {"prefix_lengths": [2, 3]},
        {"preset": "x"},
    ],
)
def test_a_hierarchy_must_have_exactly_one_valid_form(definition):
    with pytest.raises(ValidationError):
        GeneralizationHierarchy.model_validate(definition)


def _plan_for(level: int, hierarchy: dict) -> AnonymizationPlan:
    return AnonymizationPlan(
        fidelity=Fidelity.PRIVACY,
        protection=Protection.ANONYMIZATION,
        shuffle_rows=False,
        actions=[
            ColumnAction(
                column="city",
                technique="generalize_hierarchy",
                params={"hierarchy": hierarchy, "level": level},
            )
        ],
    )


def test_the_technique_replaces_cities_with_their_region_through_apply():
    frame = pd.DataFrame({"city": ["Santos", "Bauru", "Niterói", None]})
    levels = REGIONS.model_dump()

    result = Anonymizer(seed=1).apply(frame, _plan_for(1, levels))

    released = result.data["city"].tolist()
    assert released[:3] == ["Baixada", "Interior", "Niterói"]
    assert pd.isna(released[3])


def test_the_technique_accepts_the_cep_preset_in_a_plan():
    frame = pd.DataFrame({"city": ["01310-100", "01311-200", "20040-020"]})

    result = Anonymizer(seed=1).apply(frame, _plan_for(2, {"preset": "cep"}))

    assert result.data["city"].tolist() == ["013*****", "013*****", "200*****"]


def test_a_level_beyond_the_top_is_a_plan_error():
    frame = pd.DataFrame({"city": ["Santos"]})

    with pytest.raises(PlanError, match="level must be between"):
        Anonymizer(seed=1).apply(frame, _plan_for(9, REGIONS.model_dump()))
