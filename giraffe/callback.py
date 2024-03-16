from giraffe.giraffe import Giraffe
import numpy as np
from pathlib import Path
from typing import Union


class Callback:
    def __init__(self) -> None:
        pass

    def on_generation_end(self, giraffe: Giraffe) -> None:
        pass

    def on_evolution_end(self, giraffe: Giraffe) -> None:
        pass

    def on_evolution_start(self, giraffe: Giraffe) -> None:
        pass

    def on_generation_start(self, giraffe: Giraffe) -> None:
        pass


class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int) -> None:
        super().__init__()
        self.patience = patience
        self.counter = 0
        self.fitnesses = None

    def on_generation_end(self, giraffe: Giraffe) -> None:
        if self.fitnesses is not None:
            if not np.allclose(giraffe.fitnesses, self.fitnesses, atol=1e-5):
                self.fitnesses = giraffe.fitnesses
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("Patience reached, stopping evolution...")
                    giraffe.should_stop = True
        else:
            self.fitnesses = giraffe.fitnesses


class SaveParetoCallback(Callback):
    def __init__(self, path: Union[str, Path], filename="tree") -> None:
        super().__init__()
        self.path = Path(path)
        self.filename = filename

    def on_evolution_end(self, giraffe: Giraffe) -> None:
        for index, pareto_model in enumerate(giraffe.pareto_chosen):
            pareto_model.save_tree_architecture(self.path / f"{self.filename}_{index}.tree")
