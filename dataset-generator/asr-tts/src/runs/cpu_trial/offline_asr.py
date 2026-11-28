import re
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub

def cached_model_info(self, repo_id, *, revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"Offline run requires a pinned commit for {repo_id}: {revision}")
    return SimpleNamespace(sha=revision)

from functools import partial
huggingface_hub.HfApi.model_info = cached_model_info
huggingface_hub.snapshot_download = partial(huggingface_hub.snapshot_download, local_files_only=True)
script = Path(sys.argv[1]).resolve()
sys.argv = sys.argv[1:]
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name="__main__")
