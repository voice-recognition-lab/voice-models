"""Load and validate the project's actions.jsonl and samples.jsonl contracts."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator, SchemaError

from .common.io import read_jsonl
from .common.types import Action, ActionCatalog, PathLike, Sample


SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas"
ACTION_SCHEMA = json.loads((SCHEMAS_DIR / "action.schema.json").read_text(encoding="utf-8"))
SAMPLE_SCHEMA = json.loads((SCHEMAS_DIR / "sample.schema.json").read_text(encoding="utf-8"))
SPLITS = tuple(SAMPLE_SCHEMA["properties"]["split"]["enum"])


@dataclass
class Dataset:
    """Validated records, retaining file order and all original fields."""

    actions: ActionCatalog
    samples: list[Sample]

    def samples_for_split(self, split: str) -> list[Sample]:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
        return [sample for sample in self.samples if sample["split"] == split]

    def candidates_for(self, sample: Sample) -> list[Action]:
        """Resolve candidate IDs; an explicit empty list stays empty."""
        ids = sample.get("available_action_ids", self.actions)
        return [self.actions[action_id] for action_id in ids]


def validate_record(validator: Draft202012Validator, record: dict[str, Any], location: str) -> None:
    error = next(validator.iter_errors(record), None)
    if error is not None:
        field = ".".join(str(part) for part in error.absolute_path) or "record"
        raise ValueError(f"{location}: {field}: {error.message}")


def load_actions(path: PathLike) -> ActionCatalog:
    """Return a catalog indexed by action_id, checking argument schemas."""
    path = Path(path)
    actions: ActionCatalog = {}
    validator = Draft202012Validator(ACTION_SCHEMA)
    for location, action in read_jsonl(path):
        validate_record(validator, action, location)
        action_id = action["action_id"]
        if action_id in actions:
            raise ValueError(f"{location}: duplicate action_id {action_id!r}")
        parameters = action["parameters"]
        try:
            Draft202012Validator.check_schema(parameters)
        except SchemaError as exc:
            raise ValueError(f"{location}: invalid parameters schema: {exc.message}") from exc
        unknown = set(parameters["required"]) - parameters["properties"].keys()
        if unknown:
            raise ValueError(f"{location}: required parameters absent from properties: {sorted(unknown)}")
        actions[action_id] = cast(Action, action)
    if not actions:
        raise ValueError(f"{path}: empty action catalog")
    return actions


def load_samples(path: PathLike, actions: ActionCatalog) -> list[Sample]:
    """Validate labels, candidate membership and context object references."""
    path = Path(path)
    samples: list[Sample] = []
    seen_ids = set()
    validator = Draft202012Validator(SAMPLE_SCHEMA)
    argument_validators = {
        action_id: Draft202012Validator(action["parameters"])
        for action_id, action in actions.items()
    }
    for location, sample in read_jsonl(path):
        validate_record(validator, sample, location)
        sample_id = sample["sample_id"]
        if sample_id in seen_ids:
            raise ValueError(f"{location}: duplicate sample_id {sample_id!r}")
        seen_ids.add(sample_id)

        candidate_ids = sample.get("available_action_ids", actions)
        unknown = set(candidate_ids) - actions.keys()
        if unknown:
            raise ValueError(f"{location}: unknown candidate IDs: {sorted(unknown)}")
        action_id = sample["action_id"]
        if action_id is None:
            if sample["args"]:
                raise ValueError(f"{location}: args must be empty when action_id is null")
        else:
            if action_id not in candidate_ids:
                raise ValueError(f"{location}: gold action {action_id!r} is absent from candidates")
            validate_record(argument_validators[action_id], sample["args"], f"{location}: args")

        context = sample["context"]
        object_ids = [obj["object_id"] for obj in context["objects"]]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError(f"{location}: duplicate object_id in context")
        missing = set(context["selected_object_ids"]) - set(object_ids)
        if missing:
            raise ValueError(f"{location}: selected objects absent from context: {sorted(missing)}")
        samples.append(cast(Sample, sample))
    if not samples:
        raise ValueError(f"{path}: empty sample dataset")
    return samples


def resolve_dataset_paths(dataset_dir: PathLike, actions_path: PathLike | None = None) -> dict[str, Path]:
    """Resolve the selected action catalog and the original sample file."""
    dataset_dir = Path(dataset_dir)
    return {
        "inputs/actions.jsonl": Path(actions_path) if actions_path is not None else dataset_dir / "inputs/actions.jsonl",
        "dataset/samples.jsonl": dataset_dir / "dataset/samples.jsonl",
    }


def load_dataset(dataset_dir: PathLike, actions_path: PathLike | None = None) -> Dataset:
    """Load a selected catalog and validate the original samples without resplitting."""
    paths = resolve_dataset_paths(dataset_dir, actions_path)
    actions = load_actions(paths["inputs/actions.jsonl"])
    samples = load_samples(paths["dataset/samples.jsonl"], actions)
    return Dataset(actions=actions, samples=samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/v1.1"))
    parser.add_argument("--actions", type=Path, help="Alternative actions.jsonl; samples still come from --dataset")
    args = parser.parse_args()
    try:
        dataset = load_dataset(args.dataset, args.actions)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    counts = Counter(sample["split"] for sample in dataset.samples)
    print(json.dumps({
        "dataset": str(args.dataset),
        "actions_path": str(resolve_dataset_paths(args.dataset, args.actions)["inputs/actions.jsonl"]),
        "actions": len(dataset.actions),
        "samples": len(dataset.samples),
        "splits": {split: counts[split] for split in SPLITS},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
