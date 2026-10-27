import numpy as np
import pandas as pd
import pytest

from data_anonymizer.detection.patterns import is_valid_cpf
from data_anonymizer.techniques import BUILTIN_TECHNIQUES
from data_anonymizer.techniques.generalization import (
    bin_numeric,
    suppress_rare,
    top_bottom_code,
    truncate_date,
    truncate_string,
)
from data_anonymizer.techniques.masking import partial_mask, redact, redact_entities
from data_anonymizer.techniques.noise import (
    date_shift,
    gaussian_noise,
    laplace_noise,
    multiplicative_noise,
)
from data_anonymizer.techniques.perturbation import (
    microaggregate,
    randomized_response,
    rank_swap,
    shuffle,
)
from data_anonymizer.techniques.pseudonymization import (
    keyed_hash,
    scramble_characters,
    surrogate,
)


def test_redact_keeps_missing_values_missing(context):
    result = redact(pd.Series(["a", None, "b"]), context)

    assert result.iloc[0] == "[REDACTED]"
    assert pd.isna(result.iloc[1])


def test_partial_mask_keeps_separators_and_the_last_digits(context):
    result = partial_mask(pd.Series(["529.982.247-25"]), context, keep_last=3)

    assert result.iloc[0] == "***.***.**7-25"


def test_partial_mask_hides_everything_when_too_short(context):
    assert partial_mask(pd.Series(["1234"]), context, keep_last=4).iloc[0] == "****"


def test_redact_entities_labels_personal_data_inside_text(context):
    text = "Mail ana@example.com or call +55 21 98765-4321, CPF 529.982.247-25."

    result = redact_entities(pd.Series([text]), context).iloc[0]

    assert result == "Mail <EMAIL> or call <PHONE>, CPF <CPF>."


def test_keyed_hash_is_consistent_and_depends_on_the_key(context):
    values = pd.Series(["alice", "bob", "alice"])

    first = keyed_hash(values, context)
    other_key = keyed_hash(values, context.__class__(context.rng, b"other", context.surrogates))

    assert first.iloc[0] == first.iloc[2]
    assert first.iloc[0] != first.iloc[1]
    assert first.iloc[0] != other_key.iloc[0]


def test_scramble_characters_keeps_the_format(context):
    result = scramble_characters(pd.Series(["AB-1234", "AB-1234"]), context)

    assert result.iloc[0] == result.iloc[1]
    assert result.str.fullmatch(r"[A-Z]{2}-\d{4}").all()
    assert result.iloc[0] != "AB-1234"


def test_surrogate_cpfs_are_valid_unique_and_keep_bare_digit_style(context):
    originals = pd.Series(["52998224725", "11144477735", "52998224725", None])

    result = surrogate(originals, context, category="cpf")

    assert result.iloc[0] == result.iloc[2]
    assert result.iloc[0] != result.iloc[1]
    assert pd.isna(result.iloc[3])
    assert all(value.isdigit() and is_valid_cpf(value) for value in result.iloc[:3])


def test_surrogate_never_maps_two_people_to_one_fake(context):
    originals = pd.Series([f"person {index}" for index in range(500)])

    result = surrogate(originals, context, category="person_name")

    assert result.nunique() == 500


def test_bin_numeric_with_width_makes_decade_labels(context):
    result = bin_numeric(pd.Series([23, 37, 40, None]), context, width=10)

    assert result.tolist() == ["[20, 30)", "[30, 40)", "[40, 50)", None]


def test_bin_numeric_quantiles_put_every_value_in_a_bin(context):
    values = pd.Series(np.arange(100, dtype=float))

    result = bin_numeric(values, context, bins=4)

    assert result.notna().all()
    assert result.nunique() == 4


def test_truncate_date_and_string(context):
    dates = pd.Series(pd.to_datetime(["1990-07-15", "1985-02-01"]))

    assert truncate_date(dates, context).dt.month.tolist() == [1, 1]
    assert truncate_string(pd.Series(["22041-001"]), context, keep=5).iloc[0] == "22041"


def test_top_bottom_code_clips_outliers(context):
    values = pd.Series([1, 2, 3, 4, 1000])

    result = top_bottom_code(values, context, lower=2, upper=10)

    assert result.tolist() == [2, 2, 3, 4, 10]


def test_suppress_rare_replaces_small_categories(context):
    values = pd.Series(["a"] * 5 + ["b"])

    result = suppress_rare(values, context, min_count=5, replacement="OTHER")

    assert result.tolist() == ["a"] * 5 + ["OTHER"]


def test_gaussian_noise_changes_values_but_keeps_mean_and_std(context):
    values = pd.Series(np.random.default_rng(1).normal(50, 10, 1000))

    result = gaussian_noise(values, context, relative_scale=0.3)

    assert not np.allclose(result, values)
    assert result.mean() == pytest.approx(values.mean())
    assert result.std() == pytest.approx(values.std())


def test_noise_keeps_integer_columns_integer(context):
    values = pd.Series([10, 20, 30, 40, 50])

    for technique in (gaussian_noise, multiplicative_noise):
        assert pd.api.types.is_integer_dtype(technique(values, context))


def test_laplace_noise_scale_follows_epsilon(context):
    values = pd.Series(np.zeros(5000))

    loose = laplace_noise(values, context, epsilon=10, lower=0, upper=1)
    strict = laplace_noise(values, context, epsilon=0.5, lower=0, upper=1)

    assert loose.std() < strict.std()


def test_date_shift_stays_within_the_window(context):
    dates = pd.Series(pd.to_datetime(["2020-01-01"] * 200))

    shifted = date_shift(dates, context, max_days=7)

    offsets = (shifted - dates).dt.days
    assert offsets.between(-7, 7).all()
    assert offsets.nunique() > 1


def test_shuffle_and_rank_swap_keep_the_exact_distribution(context):
    values = pd.Series(np.random.default_rng(2).normal(0, 1, 500))

    for technique in (shuffle, rank_swap):
        result = technique(values, context)
        assert sorted(result) == sorted(values)
        assert not result.equals(values)


def test_rank_swap_keeps_values_near_their_rank(context):
    values = pd.Series(np.arange(1000, dtype=float))

    result = rank_swap(values, context, max_distance_percent=1)

    assert (result - values).abs().max() <= 10


def test_microaggregate_keeps_the_mean_and_groups_of_k(context):
    values = pd.Series(np.random.default_rng(3).normal(100, 15, 103))

    result = microaggregate(values, context, group_size=5)

    assert result.mean() == pytest.approx(values.mean())
    assert result.value_counts().min() >= 5


def test_randomized_response_keeps_most_values(context):
    values = pd.Series(["yes", "no"] * 2000)

    result = randomized_response(values, context, keep_probability=0.8)

    changed = (result != values).mean()
    # Of the 20% re-drawn, half land on the original value again.
    assert changed == pytest.approx(0.1, abs=0.03)


def test_unknown_parameters_are_rejected_for_every_builtin():
    for technique in BUILTIN_TECHNIQUES.values():
        with pytest.raises(TypeError):
            technique.check_parameters({"not_a_parameter": 1})
