"""The `oblitus` command line.

The commands and their flags are listed in the README's command table and in
`oblitus --help`. Pseudonyms are keyed by the OBLITUS_SECRET_KEY environment
variable. Without it, a random key is used and pseudonyms change on every run.
"""

import argparse
import base64
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from pydantic import ValidationError

from oblitus.anonymizer import AnonymizationResult, Anonymizer
from oblitus.entropy_sources import sources_by_name
from oblitus.errors import AnonymizerError, PlanError
from oblitus.key_generation import GeneratedKey, SourceStatus, generate_key
from oblitus.noise_explorer import explore_noise, save_page
from oblitus.noise_explorer.sweep import DEFAULT_MAX_ROWS
from oblitus.plan import AnonymizationPlan, Fidelity, Protection
from oblitus.synthesis import DatasetSchema, infer_schema
from oblitus.synthesis.rare_values import min_records_for
from oblitus.table_files import read_table, write_table

SECRET_KEY_VARIABLE = "OBLITUS_SECRET_KEY"

# Exit code when the output was written but does not meet the plan's protection target.
EXIT_PROTECTION_NOT_REACHED = 3


class ProtectionNotReachedError(Exception):
    """The table was written, but it is less protected than the plan asked for."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the process exit code."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "keygen":
        return _keygen(arguments)
    secret_key = os.environ.get(SECRET_KEY_VARIABLE)
    if secret_key is None and arguments.command == "reveal":
        print(
            f"error: reveal needs {SECRET_KEY_VARIABLE}, the key used to anonymize", file=sys.stderr
        )
        return 2
    if secret_key is None and arguments.command == "apply":
        print(
            f"warning: {SECRET_KEY_VARIABLE} is not set; "
            "pseudonymization plans will give different pseudonyms on every run",
            file=sys.stderr,
        )
    try:
        anonymizer = Anonymizer(
            secret_key=secret_key,
            seed=arguments.seed,
            locale=arguments.locale,
        )
    except ValueError as error:
        print(f"error: {SECRET_KEY_VARIABLE}: {error}", file=sys.stderr)
        return 2
    try:
        arguments.handler(anonymizer, arguments)
    except (AnonymizerError, ValidationError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except ProtectionNotReachedError as error:
        print(f"warning: {error}", file=sys.stderr)
        return EXIT_PROTECTION_NOT_REACHED
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oblitus",
        description="Detect, anonymize and synthesize sensitive tabular data.",
    )
    parser.add_argument("--seed", type=int, default=None, help="seed for reproducible output")
    parser.add_argument("--locale", default="en_US", help="Faker locale for fake values")
    commands = parser.add_subparsers(dest="command", required=True)

    scan_parser = commands.add_parser("scan", help="report what each column holds")
    scan_parser.add_argument("input", type=Path)
    scan_parser.add_argument("--sample-size", type=int, default=1000)
    scan_parser.set_defaults(handler=_scan)

    propose_parser = commands.add_parser("propose", help="write a reviewable plan")
    propose_parser.add_argument("input", type=Path)
    _add_fidelity(propose_parser)
    _add_protection(propose_parser)
    propose_parser.add_argument("-o", "--output", type=Path, help="plan file; stdout if absent")
    propose_parser.set_defaults(handler=_propose)

    apply_parser = commands.add_parser("apply", help="anonymize a table")
    apply_parser.add_argument("input", type=Path)
    apply_parser.add_argument("-o", "--output", type=Path, required=True)
    apply_parser.add_argument("--plan", type=Path, help="reviewed plan; proposed if absent")
    _add_fidelity(apply_parser)
    _add_protection(apply_parser)
    apply_parser.add_argument("--report", type=Path, help="write privacy and utility metrics")
    _add_row_key(apply_parser)
    apply_parser.set_defaults(handler=_apply)

    reveal_parser = commands.add_parser("reveal", help="undo reversible techniques")
    reveal_parser.add_argument("input", type=Path)
    reveal_parser.add_argument("--plan", type=Path, required=True)
    reveal_parser.add_argument("-o", "--output", type=Path, required=True)
    _add_row_key(reveal_parser)
    reveal_parser.set_defaults(handler=_reveal)

    schema_parser = commands.add_parser("infer-schema", help="learn a schema from a table")
    schema_parser.add_argument("input", type=Path)
    _add_fidelity(schema_parser)
    schema_parser.add_argument("-o", "--output", type=Path, help="schema file; stdout if absent")
    schema_parser.set_defaults(handler=_infer_schema)

    synth_parser = commands.add_parser("synthesize", help="generate a synthetic table")
    synth_parser.add_argument("-o", "--output", type=Path, required=True)
    synth_parser.add_argument("--rows", type=int, default=None)
    synth_parser.add_argument("--schema", type=Path)
    synth_parser.add_argument("--reference", type=Path)
    _add_fidelity(synth_parser)
    synth_parser.set_defaults(handler=_synthesize)

    noise_parser = commands.add_parser(
        "explore-noise", help="write an interactive page showing how noise moves a column"
    )
    noise_parser.add_argument("input", type=Path)
    noise_parser.add_argument("--column", required=True)
    noise_parser.add_argument("-o", "--output", type=Path, required=True, help="HTML file")
    noise_parser.add_argument(
        "--band", type=float, default=0.1, help="exposure band, in standard deviations"
    )
    noise_parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"explore a random sample of this many values (default {DEFAULT_MAX_ROWS:,})",
    )
    noise_parser.set_defaults(handler=_explore_noise)

    challenge_parser = commands.add_parser(
        "challenge", help="generate a data-science challenge with a known answer"
    )
    challenge_parser.add_argument("--schema", type=Path, required=True)
    challenge_parser.add_argument("--rows", type=int, default=None)
    challenge_parser.add_argument("-o", "--output", type=Path, required=True, help="folder")
    challenge_parser.set_defaults(handler=_challenge)

    keygen_parser = commands.add_parser(
        "keygen", help="print a secret key mixed from several entropy sources"
    )
    keygen_parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME",
        help="extra source: hardware, microphone, camera, timing or beacon; repeatable. "
        "os is always included",
    )
    keygen_parser.add_argument("--format", choices=["hex", "base64"], default="hex")

    techniques_parser = commands.add_parser("techniques", help="list available techniques")
    techniques_parser.set_defaults(handler=_list_techniques)
    return parser


def _keygen(arguments: argparse.Namespace) -> int:
    try:
        generated = generate_key(sources_by_name(arguments.source))
    except AnonymizerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(_encode_key(generated.key, arguments.format))
    _print_key_report(generated)
    return 0


def _encode_key(key: bytes, key_format: str) -> str:
    if key_format == "base64":
        return base64.b64encode(key).decode("ascii")
    return key.hex()


def _print_key_report(generated: GeneratedKey) -> None:
    for report in generated.sources:
        note = f" ({report.note})" if report.note else ""
        if report.status is SourceStatus.SKIPPED:
            print(f"  {report.name:<11} skipped{note}", file=sys.stderr)
            continue
        print(
            f"  {report.name:<11} used: {report.byte_count} bytes, "
            f"min-entropy about {report.min_entropy_bits:.0f} bits{note}",
            file=sys.stderr,
        )


def _add_fidelity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fidelity",
        type=Fidelity,
        choices=list(Fidelity),
        default=Fidelity.STATISTICAL,
        help="statistical keeps distributions and joins; privacy generalizes and randomizes",
    )


def _add_protection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--protection",
        type=Protection,
        choices=list(Protection),
        default=Protection.ANONYMIZATION,
        help="pseudonymization keeps keyed links; anonymization removes every link and checks it",
    )


def _add_row_key(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--row-key",
        metavar="COLUMN",
        help="column that identifies each row; overrides the plan's row_key",
    )


def _with_row_key(plan: AnonymizationPlan, row_key: str | None) -> AnonymizationPlan:
    if row_key is None:
        return plan
    return plan.model_copy(update={"row_key": row_key})


def _scan(anonymizer: Anonymizer, arguments: argparse.Namespace) -> None:
    findings = anonymizer.scan(read_table(arguments.input), sample_size=arguments.sample_size)
    print(f"{'column':<24} {'category':<20} {'sensitivity':<18} {'conf':>5}  evidence")
    for finding in findings:
        print(
            f"{finding.column:<24} {finding.category.value:<20} "
            f"{finding.sensitivity.value:<18} {finding.confidence:>5.2f}  {finding.evidence}"
        )


def _propose(anonymizer: Anonymizer, arguments: argparse.Namespace) -> None:
    plan = anonymizer.propose(
        read_table(arguments.input),
        fidelity=arguments.fidelity,
        protection=arguments.protection,
    )
    _emit(plan.to_json(), arguments.output)


def _apply(anonymizer: Anonymizer, arguments: argparse.Namespace) -> None:
    frame = read_table(arguments.input)
    if arguments.plan is not None:
        plan = AnonymizationPlan.load(arguments.plan)
    else:
        plan = anonymizer.propose(
            frame, fidelity=arguments.fidelity, protection=arguments.protection
        )

    result = anonymizer.apply(frame, _with_row_key(plan, arguments.row_key))
    write_table(result.data, arguments.output)
    if arguments.report is not None:
        arguments.report.write_text(_report_json(result) + "\n", encoding="utf-8")
    print(
        f"wrote {len(result.data)} rows to {arguments.output}; "
        f"k-anonymity {result.privacy_before.k_anonymity} -> {result.privacy_after.k_anonymity}; "
        f"result is {result.protection.level.value}"
    )
    for problem in result.protection.problems:
        print(f"  - {problem}")
    if not result.protection.reached:
        raise ProtectionNotReachedError(f"the {plan.protection.value} target was not reached")


def _report_json(result: AnonymizationResult) -> str:
    report = {
        "plan": result.plan.model_dump(mode="json"),
        "privacy_before": asdict(result.privacy_before),
        "privacy_after": asdict(result.privacy_after),
        "protection": asdict(result.protection),
        "utility": asdict(result.utility),
    }
    return json.dumps(report, indent=2)


def _reveal(anonymizer: Anonymizer, arguments: argparse.Namespace) -> None:
    frame = read_table(arguments.input)
    plan = _with_row_key(AnonymizationPlan.load(arguments.plan), arguments.row_key)
    revealed = anonymizer.reveal(frame, plan)
    write_table(revealed, arguments.output)
    print(f"wrote {len(revealed)} rows to {arguments.output}")


def _infer_schema(anonymizer: Anonymizer, arguments: argparse.Namespace) -> None:
    frame = read_table(arguments.input)
    findings = anonymizer.scan(frame)
    min_count = min_records_for(arguments.fidelity)
    schema = infer_schema(frame.to_pandas(), findings, min_category_count=min_count)
    _emit(schema.to_json(), arguments.output)


def _synthesize(anonymizer: Anonymizer, arguments: argparse.Namespace) -> None:
    schema = None if arguments.schema is None else DatasetSchema.load(arguments.schema)
    reference = None
    if arguments.reference is not None:
        reference = read_table(arguments.reference).to_pandas()
    frame = anonymizer.synthesize(
        arguments.rows, schema=schema, reference=reference, fidelity=arguments.fidelity
    )
    write_table(frame, arguments.output)
    print(f"wrote {len(frame)} rows to {arguments.output}")


def _explore_noise(anonymizer: Anonymizer, arguments: argparse.Namespace) -> None:
    frame = read_table(arguments.input)
    try:
        exploration = explore_noise(
            frame,
            arguments.column,
            seed=arguments.seed or 0,
            band_sd=arguments.band,
            max_rows=arguments.max_rows,
        )
    except ValueError as error:
        raise PlanError(str(error)) from error
    save_page(exploration, arguments.output)
    print(f"wrote {arguments.output}; open it in a browser")


def _challenge(anonymizer: Anonymizer, arguments: argparse.Namespace) -> None:
    challenge = anonymizer.challenge(DatasetSchema.load(arguments.schema), arguments.rows)
    folder: Path = arguments.output
    folder.mkdir(parents=True, exist_ok=True)
    write_table(challenge.train, folder / "train.csv")
    write_table(challenge.test_features, folder / "test.csv")
    private = folder / "answers"
    private.mkdir(exist_ok=True)
    write_table(challenge.test_answers, private / "test_answers.csv")
    key_path = private / "answer_key.json"
    key_path.write_text(json.dumps(challenge.answer_key, indent=2, default=str) + "\n")
    (folder / "QUESTION.md").write_text(_question_page(challenge), encoding="utf-8")
    print(f"wrote the challenge to {folder}; keep {private} away from participants")


def _question_page(challenge) -> str:
    key = challenge.answer_key
    columns = "\n".join(f"- `{name}`" for name in challenge.train.columns)
    return (
        f"# {key['title']}\n\n{key['prompt']}\n\n"
        f"Task: {key['task']}. Target column: `{key['target']}`.\n\n"
        f"Files: `train.csv` ({key['rows']['train']} rows, with the target) and "
        f"`test.csv` ({key['rows']['test']} rows, without it).\n\n"
        f"{_treatment_note(key)}"
        f"{_panel_note(key)}"
        f"## Columns\n\n{columns}\n"
    )


def _treatment_note(key: dict) -> str:
    causal = key.get("causal")
    if causal is None:
        return ""
    values = causal["treatment_values"]
    return (
        f"Treatment column: `{causal['treatment']}` (control {values['control']!r}, "
        f"treated {values['treated']!r}).\n\n"
    )


def _panel_note(key: dict) -> str:
    series = key.get("time_series")
    if series is None:
        return ""
    note = f"Each row is one `{series['entity']}` in one period, dated by `{series['time']}`."
    first_test = series["first_test_period"]
    if first_test is not None:
        periods = series["periods_per_split"]["test"]
        note += f" The test rows are the last {periods} periods, from {first_test['date']}."
    return note + "\n\n"


def _list_techniques(anonymizer: Anonymizer, arguments: argparse.Namespace) -> None:
    for technique in anonymizer.techniques.values():
        parameters = ", ".join(technique.parameter_names()) or "-"
        direction = "reversible" if technique.reversible else "one-way"
        print(
            f"{technique.name:<26} {technique.family.value:<17} "
            f"{technique.effect.value:<24} {direction}"
        )
        print(f"    {technique.summary} Parameters: {parameters}")


def _emit(text: str, output: Path | None) -> None:
    if output is None:
        print(text)
        return
    output.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    sys.exit(main())
