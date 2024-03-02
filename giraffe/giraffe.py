from giraffe.backend.backend import Backend
from giraffe.globals import BACKEND as B

from pathlib import Path

from typing import Union, Iterable, Callable
import numpy as np


class Giraffe:
    def __init__(
        self,
        preds_source: Union[Path, str, Iterable[Path], Iterable[str]],
        gt_path: Union[Path, str],
        population_size: int,
        population_multiplier: int,
        fitness_function: Callable,
        allow_all_ops: bool = True,
        mutation_chance_crossover: bool = True,
        seed=None,
        backend=None,
        loader=None,
    ):
        if self.backend is not None:
            Backend.set_backend(backend)

        self.input_paths = self._get_input_paths(preds_source)
        self.gt_path = Path(gt_path)
        self.loader = loader if loader is not None else self._determine_loader(self.input_paths + [gt_path])
        self.population_size = population_size
        self.population_multiplier = population_multiplier
        self.fitness_function = fitness_function  # TODO: add some checking here
        self.allow_all_ops = allow_all_ops
        self.mutation_chance_crossover = mutation_chance_crossover
        if seed is not None:
            np.random.seed(seed)

    def _get_input_paths(self, preds_source):
        if hasattr(preds_source, "__iter__"):
            return [Path(preds_source) for preds_source in preds_source]
        elif isinstance(preds_source, (Path, str)):
            if Path(preds_source).is_dir():
                return list(Path(preds_source).glob("*"))
            else:
                raise ValueError("If preds_source is a single path, it must be a directory")

    def _determine_loader(self, input_paths):
        extensions = set([path.suffix for path in input_paths])
        if len(extensions) > 1:
            raise ValueError(f"Multiple file extensions found in input paths: {extensions}")
        else:
            extension = extensions.pop()
            if extension == ".pt":
                loader = getattr(B, "load_torch", None)
                if loader is None:
                    raise ValueError("Torch tensors found but selected backend does not have load_torch method")
            elif extension == ".npy":
                loader = getattr(B, "load_numpy", None)
                if loader is None:
                    raise ValueError("Numpy arrays found but selected backend does not have load_numpy method")
            elif extension == ".tf":
                loader = getattr(B, "load_tensorflow", None)
                if loader is None:
                    raise ValueError(
                        "Tensorflow tensors found but selected backend does not have load_tensorflow method"
                    )
            else:
                raise ValueError(f"Unknown file extension {extension}")
