"""Load the bi-encoder used for hybrid candidate retrieval."""

from ..bi_encoder.model import BiEncoder
from ..common.config import load_encoder_config
from ..common.types import EncoderConfig


def create_retriever(
    config: EncoderConfig, saved_config: EncoderConfig | None = None,
) -> tuple[BiEncoder, EncoderConfig]:
    """Load a retriever for mining or evaluation and retain its resolved revision."""
    saved = saved_config if saved_config is not None else config.get("retriever")
    retriever_config = saved.copy() if saved else load_encoder_config(
        config["retriever_config"], config.get("retriever_profile"),
    )
    retriever_config["device"] = config["device"]
    retriever = BiEncoder.from_config(retriever_config)
    if retriever.revision is not None:
        retriever_config["revision"] = retriever.revision
    return retriever, retriever_config
