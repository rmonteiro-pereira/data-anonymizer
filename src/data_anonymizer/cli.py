"""The `anonymize` command: scan, propose, apply, infer-schema, synthesize and techniques.

Usage:
    anonymize scan INPUT [--sample-size N]
    anonymize propose INPUT [--fidelity statistical|privacy] [-o PLAN.json]
    anonymize apply INPUT -o OUTPUT [--plan PLAN.json | --fidelity F] [--report REPORT.json]
    anonymize reveal INPUT --plan PLAN.json -o OUTPUT
    anonymize infer-schema INPUT [-o SCHEMA.json] [--fidelity F]
    anonymize synthesize -o OUTPUT --rows N [--schema SCHEMA.json] [--reference INPUT]
    anonymize techniques

Every command also takes --seed and --locale. Pseudonyms are keyed by the
ANONYMIZER_SECRET_KEY environment variable; without it, a random key is used and
pseudonyms change on every run.
"""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from pydantic import ValidationError

from data_anonymizer.anonymizer import AnonymizationResult, Anonymizer
from data_anonymizer.errors import AnonymizerError
from data_anonymizer.plan import AnonymizationPlan, Fidelity
from data_anonymizer.synthesis import DatasetSchema, infer_schema
from data_anonymizer.table_files import read_table, write_table

SECRET_KEY_VARIABLE = "ANONYMIZER_SECRET_KEY"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the process exit code."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    secret_key = os.environ.get(SECRET_KEY_VARIABLE)
    if secret_key is None and arguments.command == "reveal":
        print(
            f"error: reveal needs {SECRET_KEY_VARIABLE}, the key used to anonymize", file=sys.stderr
        )
        return 2
    if secret_key is None and arguments.command == "apply":
        print(
            f"warning: {SECRET_KEY_VARIABLE} is not set; pseudonyms will differ on every run",
            file=sys.stderr,
        )
    anonymizer = Anonymizer(
        secret_key=secret_key,
        seed=arguments.seed,
        locale=arguments.locale,
    )
    try:
        arguments.handler(anonymizer, arguments)
    except (AnonymizerError, ValidationError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anonymize",
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
    propose_parser.add_argument("-o", "--output", type=Path, help="plan file; stdout if absent")
    propose_parser.set_defaults(handler=_propose)

    apply_parser = commands.add_parser("apply", help="anonymize a table")
    apply_parser.add_argument("input", type=Path)
    apply_parser.add_argument("-o", "--output", type=Path, required=True)
    apply_parser.add_argument("--plan", type=Path, help="reviewed plan; proposed if absent")
    _add_fidelity(apply_parser)
    apply_parser.add_argument("--report", type=Path, help="write privacy and utility metrics")
    apply_parser.set_defaults(handler=_apply)

    reveal_parser = commands.add_parser("reveal", help="undo reversible techniques")
    reveal_parser.add_argument("input", type=Path)
    reveal_parser.add_argument("--plan", type=Path, required=True)
    reveal_parser.add_argument("-o", "--output", type=Path, required=True)
    reveal_parser.set_defaults(handler=_reveal)

    schema_parser = commands.add_parser("infer-schema", help="learn a schema from a table")
    schema_parser.add_argument("input", type=Path)
    _add_fidelity(schema_parser)
    schema_parser.add_argument("-o", "--output", type=Path, help="schema file; stdout if absent")
    schema_parser.set_defaults(handler=_infer_schema)

    synth_parser = commands.add_parser("synthesize", help="generate a synthetic table")
    synth_parser.add_argument("-o", "--output", type=Path, required=True)
    synth_parser.add_argument("--rows", type=int, required=True)
    synth_parser.add_argument("--schema", type=Path)
    synth_parser.add_argument("--reference", type=Path)
    _add_fidelity(synth_parser)
    synth_parser.set_defaults(handler=_synthesize)

    techniques_parser = commands.add_parser("techniques", help="list available techniques")
    techniques_parser.set_defaults(handler=_list_techniques)
    return parser


def _add_fidelity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fidelity",
        type=Fidelity,
        choices=list(Fidelity),
        default=Fidelity.STATISTICAL,
        help="statistical keeps distributions and joins; privacy generalizes and randomizes",
    )


def _scan(anonymizer: Anonymizer, arguments: argparse.Namespace) -> None:
    findings = anonymizer.scan(read_table(arguments.input), sample_size=arguments.sample_size)
    print(f"{'column':<24} {'category':<20} {'sensitivity':<18} {'conf':>5}  evidence")
    for finding in findings:
        print(
            f"{finding.column:<24} {finding.category.value:<20} "
            f"{finding.sensitivity.value:<18} {finding.confidence:>5.2f}  {finding.evidence}"
        )


def _propose(anonymizer: Anonymizer, arguments: argparse.Namespace) -> None:
    plan = anonymizer.propose(read_table(arguments.input), fidelity=arguments.fidelity)
    _emit(plan.to_json(), arguments.output)


def _apply(anonymizer: Anonymizer, arguments: argparse.Namespace) -> None:
    frame = read_table(arguments.input)
    if arguments.plan is not None:
        plan = AnonymizationPlan.load(arguments.plan)
    else:
        plan = anonymizer.propose(frame, fidelity=arguments.fidelity)

    result = anonymizer.apply(frame, plan)
    write_table(result.data, arguments.output)
    if arguments.report is not None:
        arguments.report.write_text(_report_json(result) + "\n", encoding="utf-8")
    print(
        f"wrote {len(result.data)} rows to {arguments.output}; "
        f"k-anonymity {result.privacy_before.k_anonymity} -> {result.privacy_after.k_anonymity}"
    )


def _report_json(result: AnonymizationResult) -> str:
    report = {
        "plan": result.plan.model_dump(mode="json"),
        "privacy_before": asdict(result.privacy_before),
        "privacy_after": asdict(result.privacy_after),
        "utility": asdict(result.utility),
    }
    return json.dumps(report, indent=2)


def _reveal(anonymizer: Anonymizer, arguments: argparse.Namespace) -> None:
    frame = read_table(arguments.input)
    revealed = anonymizer.reveal(frame, AnonymizationPlan.load(arguments.plan))
    write_table(revealed, arguments.output)
    print(f"wrote {len(revealed)} rows to {arguments.output}")


def _infer_schema(anonymizer: Anonymizer, arguments: argparse.Namespace) -> None:
    frame = read_table(arguments.input)
    min_count = 5 if arguments.fidelity is Fidelity.PRIVACY else 1
    schema = infer_schema(frame, anonymizer.scan(frame), min_category_count=min_count)
    _emit(schema.to_json(), arguments.output)


def _synthesize(anonymizer: Anonymizer, arguments: argparse.Namespace) -> None:
    schema = None if arguments.schema is None else DatasetSchema.load(arguments.schema)
    reference = None if arguments.reference is None else read_table(arguments.reference)
    frame = anonymizer.synthesize(
        arguments.rows, schema=schema, reference=reference, fidelity=arguments.fidelity
    )
    write_table(frame, arguments.output)
    print(f"wrote {len(frame)} rows to {arguments.output}")


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
