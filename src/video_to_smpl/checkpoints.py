"""Load trusted thesis checkpoints, including paths saved on another OS."""
import io
import pickle
from pathlib import Path
import types

import torch


class PortableUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name in ("WindowsPath", "PosixPath"):
            return Path
        return super().find_class(module, name)


def load_checkpoint(path):
    portable_pickle = types.ModuleType("portable_pickle")
    portable_pickle.Unpickler = PortableUnpickler
    portable_pickle.load = lambda file, **kwargs: PortableUnpickler(file, **kwargs).load()
    portable_pickle.loads = lambda data, **kwargs: PortableUnpickler(io.BytesIO(data), **kwargs).load()
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=portable_pickle)
