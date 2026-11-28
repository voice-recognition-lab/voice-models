"""GPU capability probes used by the adapters and their own Python environments."""

import ctypes
import json
import sys

from recognition_config import asr_device


def nvidia_device_name(index):
    # The CUDA driver uses the same visible-device ordinals as CTranslate2.
    driver = (ctypes.WinDLL("nvcuda.dll", winmode=0x800) if sys.platform == "win32"
              else ctypes.CDLL("libcuda.so.1"))
    device = ctypes.c_int()
    name = ctypes.create_string_buffer(256)
    for function, arguments in (
        (driver.cuInit, (0,)),
        (driver.cuDeviceGet, (ctypes.byref(device), index)),
        (driver.cuDeviceGetName, (name, len(name), device)),
    ):
        status = function(*arguments)
        if status:
            raise RuntimeError(f"{function.__name__} failed with CUDA driver status {status}")
    return name.value.decode("utf-8", errors="replace")


def probe_cuda(backend, index, compute_type):
    try:
        if backend == "whisper":
            import ctranslate2

            count = ctranslate2.get_cuda_device_count()
            if index >= count:
                raise ValueError(f"CTranslate2 sees {count} CUDA device(s)")
            supported = ctranslate2.get_supported_compute_types("cuda", index)
            if compute_type not in supported:
                raise ValueError(f"CTranslate2 supports {sorted(supported)} on this GPU")
            name = nvidia_device_name(index)
        else:
            import torch

            if not torch.version.cuda:
                raise ValueError("PyTorch is not a CUDA build")
            count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if index >= count:
                raise ValueError(f"PyTorch sees {count} CUDA device(s)")
            device = torch.device("cuda", index)
            # A tiny FP32 kernel detects incompatible GPU architectures as well as missing drivers.
            torch.ones(1, device=device, dtype=torch.float32).add_(1)
            torch.cuda.synchronize(device)
            name = torch.cuda.get_device_name(device)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"{backend} cannot use cuda:{index} with {compute_type}: {error}") from error
    return {"name": name, "index": index}


if __name__ == "__main__":
    backend, device, compute_type = sys.argv[1:]
    try:
        _, index = asr_device(device)
        print(json.dumps(probe_cuda(backend, index, compute_type)))
    except ValueError as error:
        sys.exit(str(error))
