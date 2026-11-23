"""Request and action serialization shared by both encoder architectures."""

import json

from .types import Action, Sample


def action_text(action: Action) -> str:
    """Exclude action IDs from the representation used by the encoder."""
    return action["description"] + "\nParameters: " + json.dumps(
        action["parameters"], ensure_ascii=False, sort_keys=True,
    )


def query_text(sample: Sample) -> str:
    """Serialize only request fields, excluding labels and dataset metadata."""
    context = sample["context"]
    if not context["objects"] and not context["selected_object_ids"]:
        return sample["text"]
    return sample["text"] + "\nContext: " + json.dumps(
        context, ensure_ascii=False, sort_keys=True,
    )


