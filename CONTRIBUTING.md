# Contributing and project standards

This page lists the standards every change to oblitus follows: code, design, tests,
documentation, privacy review, commits and releases. Read [ARCHITECTURE.md](ARCHITECTURE.md)
first: it says where each concept lives.

## Set up and run the checks

You need Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/rmonteiro-pereira/oblitus.git
cd oblitus
uv sync --all-extras
```

Every change passes these checks before it is committed:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest -q
```

GitHub Actions runs the same checks on Python 3.12 and 3.13 for every push to `main` and every
pull request (`.github/workflows/ci.yml`), with `ruff format --check` so formatting is checked
rather than rewritten. A red run blocks a release.

## Code style

The formatter decides layout: `ruff format`, at 100 columns (`pyproject.toml`). Beyond that:

- One statement per line, and one variable per assignment.
- No line over 100 characters, except a string or URL that cannot be split.
- Names say what a thing is, in words. Single letters only for loop indexes and tiny lambdas.
- A function does one thing, at about 40 lines. A module holds one concept, at about 400
  lines; past that, split it by concept.
- No nested conditional expressions. Use `if`/`else` or a lookup table.
- Comments explain why, in plain English, one idea per sentence: a constraint, a trade-off, or
  what breaks if the code changes. They never carry ticket, task or requirement IDs, or notes
  about how the change was made. Those belong in the commit message.
- Public functions, classes and modules get a docstring saying what they are for.

## Design

- **Follow the repository's way.** Find how a neighbouring module does the same thing (errors,
  validation, tests, engine access) and do it the same way. Extend an existing function rather
  than writing a parallel version.
- **Name modules after domain concepts**, such as `local_suppression.py` or
  `identifying_columns.py`, never `utils`, `helpers` or `manager`.
- **Keep decisions pure and I/O at the edges.** Only `table_files.py`, `noise_explorer/page.py`,
  `cli.py`, and the JSON `save` and `load` of `plan.py` and `synthesis/schema.py` touch files. The library never reads environment variables or configures
  logging; the CLI does.
- **Validate at the boundary.** Plans and schemas are pydantic models validated on load, and a
  plan is checked against the table before anything runs. Inside, trust the typed objects.
- **Raise specific errors** from `errors.py` (`PlanError`, `SchemaError`, `TableFormatError`),
  with the column or file in the message. Never swallow an error.
- **One implementation for every engine.** A technique is
  `(table, column, context, **params) -> narwhals expression`, so it runs on pandas, Polars and
  DuckDB alike. Engine-specific code lives only in `engines.py`.
- **No abstraction before it is needed.** Add an interface or a pattern only when its problem
  exists in the code now, and use its usual name.
- **A decision with real trade-offs gets an ADR** in `docs/adr/`, numbered, such as
  [0001-dataframe-backends.md](docs/adr/0001-dataframe-backends.md).

## Privacy rules the code must keep

These rules protect the people in the data. A change that would break one needs a decision
recorded in an ADR, not a quiet edit.

- An anonymization plan never runs a technique that keeps a link to a person, and runs under a
  throwaway key. Category codes and random fakes are fresh in every anonymized release.
- A fake value never equals a real value of the column.
- The protection level comes from checks on the released data, never from technique names
  alone. A release that fails a check is `at_risk`, and the problem text says why and what to
  change.
- A stricter check may change a test that expected `anonymized` only when the new problem is
  real. Say which tests changed, and why, in the commit message.
- A limit that cannot be fixed is written down where users will see it: `docs/protection.md`
  ("What keyed noise cannot hide", "What the checks cannot catch") or
  `docs/business-confidentiality.md`.
- Tests and examples use fake keys and synthetic data. Never commit real personal data.

## Tests

- Tests live in `tests/`, one file per area, named after the behaviour (`test_row_order.py`,
  `test_untreated_columns.py`). Shared fixtures are in `tests/conftest.py`.
- A test name states the behaviour in a sentence:
  `test_a_fake_never_repeats_a_real_value_of_the_column`.
- Every bug fix comes with a test that fails without the fix. Run it against the unfixed code
  once to be sure.
- A technique that must behave the same on every engine joins `MAPPING_TECHNIQUES` or
  `EXPRESSION_TECHNIQUES` in `tests/test_engines.py`.
- Pin values, not bits. A pinned output is hashed after rounding floats, because numpy's
  `exp` and `log` can differ in the last bit between machines.
- Tests run offline and fast: the full suite takes under a minute.

## Documentation

Documentation changes in the same commit as the code that needs them.

| When a change... | Update |
|---|---|
| adds or changes a CLI command or flag | the command table in `README.md` |
| adds or changes a public function, plan field or technique | `docs/techniques.md`, `docs/protection.md` or `docs/synthesis.md` |
| adds a module or moves a boundary | the code map and views in `ARCHITECTURE.md` |
| changes what a user sees or gets | a line under `## Unreleased` in `CHANGELOG.md` |
| makes a choice between serious alternatives | an ADR in `docs/adr/` |

Write for a reader who has not seen the project: the answer first, ordinary words, a working
command with its expected result, and the limits next to the step they affect. No promotional
adjectives, no notes about how the text was written.

## Privacy review: attack rounds

Protection claims are tested by attacking the library's own outputs, after a feature is
complete and before a release that changes protection.

1. **Build hard datasets with known truth.** Use planted patterns such as rare diagnoses, a
   dominant store, a two-store segment, repeated accounts or obfuscated identifiers in text,
   not only public datasets.
2. **State each attack before running it.** Say what the attacker holds and the fact they are
   after.
3. **Keep the attacker realistic.** They may hold their own records, a handful of known
   values, or public facts. An attacker who already knows most of the original data is out
   of scope, because the release's purpose is lost by then.
4. **Score information gain, not pass or fail.** Measure the attacker's best guess twice:
   from what they held beforehand, then with the release too. Report, for example, the error
   before and after, or a success rate against chance.
5. **Give each finding one verdict.**

   | Verdict | Meaning |
   |---|---|
   | `intended` | The release is meant to show it, such as seasonality |
   | `acceptable` | A relative fact the docs accept, such as the biggest store staying biggest |
   | `leak` | Something the docs claim to hide, or that combines with public data into a secret figure |
   | `break` | A protection claim is false, such as `anonymized` beside a successful realistic attack |
   | `safe` | The attack failed, with the numbers that show it |

6. **Fix, then re-attack the fixed version.** Value-level attacks are scored on releases
   with `shuffle_rows=False` or a hidden truth column, so a row shuffle cannot hide a leak.
7. **Stop** when a full round finds no `break` and no undocumented serious `leak`, or after
   three confirmation rounds. Whatever remains is documented as a known limit.

## Commits and pull requests

- One logical change per commit, with the tests and docs it needs.
- The subject line is an imperative sentence of at most about 72 characters, such as
  "Never draw a fake equal to a real value of the column".
- The body says why the change was needed and what it changes for users. Traceability, such
  as an issue number, goes here, not in code comments.
- No attribution trailers: no `Co-Authored-By`, and no tool or session lines.

## Releases

Releases go to PyPI as `oblitus` and to GitHub. Publishing to PyPI is automatic: creating a
GitHub release runs `.github/workflows/publish.yml`, which builds the package and uploads it
through PyPI trusted publishing, so no token is stored anywhere.

1. Set `version` in `pyproject.toml`, and rename `## Unreleased` in `CHANGELOG.md` to the
   version.
2. Build with `uv build`, then install the wheel into a fresh virtual environment outside the
   repository. Run the quick start's commands with that environment's `oblitus`, passing
   `--schema` the repository's `examples/customers_schema.json`, and check that `apply` prints
   `result is anonymized`.
3. Push, and wait for CI to pass on the release commit.
4. Tag `vX.Y.Z` on that commit and create the GitHub release, with the wheel and the source
   archive attached. The notes cover install, highlights and known limits. Publishing the
   release starts the PyPI upload.
5. Once the upload finishes, install with `pip install oblitus==X.Y.Z` in a fresh
   environment, as a user would.
