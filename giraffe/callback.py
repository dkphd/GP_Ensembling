from giraffe.giraffe import Giraffe
import numpy as np
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
from giraffe.ops import choose_pareto_rest_sorted


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


class PrintBestFitnessCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_generation_end(self, giraffe: Giraffe) -> None:
        print(f"Fitness after iteration {giraffe.iteration} = {np.max(giraffe.fitnesses)}")


class SaveParetoCallback(Callback):
    def __init__(self, path: Union[str, Path], filename="tree") -> None:
        super().__init__()
        self.path = Path(path)
        self.filename = filename

    def on_evolution_end(self, giraffe: Giraffe) -> None:
        for index, pareto_model in enumerate(giraffe.pareto_chosen):
            pareto_model.save_tree_architecture(self.path / f"{self.filename}_{index}.tree")


class DrawParetoCallback(Callback):
    def __init__(self, path: Union[str, Path], filename="tree", every_epoch=False):
        self.path = Path(path)
        self.filename = filename
        self.every_epoch = every_epoch

        self.counter = 0

    def on_generation_end(self, giraffe: Giraffe) -> None:
        filename = self.filename + ("" if not self.every_epoch else "_{self.counter}") + ".png"
        fig = self._draw_pareto(giraffe)
        fig.savefig(self.path / filename)
        self.counter += 1

    def on_evolution_end(self, giraffe: Giraffe) -> None:
        self.counter = 0

    def _draw_pareto(self, giraffe: Giraffe):
        fig, ax = plt.subplots()

        population, fitnesses, population_pareto = choose_pareto_rest_sorted(
            giraffe.population, giraffe.fitnesses, giraffe.population_size
        )
        sizes = np.array([ind.nodes_count for ind in population])

        pareto_codes = [ind.__repr__() for ind in population_pareto]

        mask = np.array([True if ind.__repr__() in pareto_codes else False for ind in population])

        _ = ax.scatter(sizes[~mask], fitnesses[~mask], c="black", label="Not pareto optimal")
        _ = ax.scatter(sizes[mask], fitnesses[mask], c="#62b879", label="Pareto optimal")
        ax.set_xlabel("Ensemble size")
        ax.set_ylabel("Fitness")
        ax.legend()

        return fig
