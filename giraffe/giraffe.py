import giraffe
from giraffe.backend.backend import Backend
from giraffe.globals import BACKEND as B
from giraffe.node import MeanNode, MaxNode, MinNode, OperatorNode
from giraffe.fitness import average_precision_fitness, calculate_fitnesses
from giraffe.gp_ops import regenerate_population, mutate_population, crossover
from giraffe.ops import first_uniques_mask, join_populations, choose_pareto_rest_sorted
from pathlib import Path

from typing import Union, Iterable, Callable
import numpy as np
import tqdm


class Giraffe:
    def __init__(
        self,
        preds_source: Union[Path, str, Iterable[Path], Iterable[str]],
        gt_path: Union[Path, str],
        population_size: int,
        population_multiplier: int,
        tournament_size: int = 5,
        fitness_function: Callable = average_precision_fitness,
        allowed_op_nodes: Iterable[OperatorNode] = (MeanNode, MaxNode, MinNode),
        callbacks: Iterable["giraffe.callback.Callback"] = [],
        mutation_chance_crossover: bool = True,
        seed=None,
        backend=None,
        loader=None,
    ):
        if backend is not None:
            Backend.set_backend(backend)

        ## casts
        preds_source = Path(preds_source)
        gt_path = Path(gt_path)

        # BASIC GIRAFFE VARIABLES
        self.should_stop = False
        self.iteration = 0
        self.population_size = population_size
        self.population = []
        self.additional_population = []
        self.population_codes = []
        self.fitnesses = []
        self.pareto_chosen = []
        # BASIC GP VARIABLES
        self.population_multiplier = population_multiplier
        self.tournament_size = tournament_size
        self.fitness_function = fitness_function  # TODO: add some checking here
        self.allowed_op_nodes = allowed_op_nodes
        self.mutation_chance_crossover = mutation_chance_crossover
        ###

        self.input_paths = self._get_input_paths(preds_source)
        self.gt_path = Path(gt_path)
        self.loader = (
            loader if loader is not None else self._determine_loader(self.input_paths + [gt_path])
        )  # TODO: add some checking here
        if seed is not None:
            np.random.seed(seed)

        self.callbacks = callbacks

        self.tensors, self.gt = self._load_data(self.input_paths, self.gt_path)
        self._validate_input()

    def train(self, iterations: int):
        self._initialize_pop()
        self._call_hook("on_evolution_start")

        self.should_stop = False
        for i in tqdm.tqdm(range(iterations)):
            self.iteration = i
            self._call_hook("on_generation_start")
            self.fitnesses = calculate_fitnesses(self.population, self.gt, self.fitness_function)

            self.additional_population = self._create_additional_population()
            self.additional_population += mutate_population(
                self.additional_population,
                list(self.tensors.values()),
                list(self.tensors.keys()),
                allowed_ops=self.allowed_op_nodes,
            )

            population, self.fitnesses = join_populations(
                self.population, self.fitnesses, self.additional_population, self.gt, self.fitness_function
            )
            self.additional_population = []

            # At this place we remove duplicates and regenerate the population
            self.population_codes = [tree.__repr__() for tree in population]
            unique_codes = first_uniques_mask(self.population_codes)

            self.population = [tree for tree, unique in zip(population, unique_codes) if unique]
            self.fitnesses = self.fitnesses[unique_codes]
            self.population, self.fitnesses, self.pareto_chosen = choose_pareto_rest_sorted(
                self.population, self.fitnesses, self.population_size
            )

            self._call_hook("on_generation_end")

            if self.should_stop:
                break

        self._call_hook("on_evolution_end")

        return self.population

    def _call_hook(self, hook_name):
        for callback in self.callbacks:
            getattr(callback, hook_name)(self)

    def _get_input_paths(self, preds_source):
        if hasattr(preds_source, "__iter__"):
            input_paths = [Path(preds_source) for preds_source in preds_source]
        elif isinstance(preds_source, (Path, str)):
            if Path(preds_source).is_dir():
                input_paths = list(Path(preds_source).glob("*"))
            else:
                raise ValueError("If preds_source is a single path, it must be a directory")

        if len(input_paths) < self.population_size:
            raise ValueError(
                f"Too little base models found to run giraffe with population size: {self.population_size}, number of models: {len(input_paths)}"
            )

        return input_paths

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

            return loader

    def _load_data(self, input_paths, gt_path):
        tensors = {path.name: self.loader(path) for path in input_paths}
        gt = self.loader(gt_path)
        if len(tensors) < self.tournament_size:
            self.tournament_size = len(tensors)
        return tensors, gt

    def _validate_input(self, fix_swapped=True):  # no way to change this argument for now TODO
        # check if all tensors have the same shape
        shapes = [B.shape(tensor) for tensor in self.tensors.values()]
        if len(set(shapes)) > 1:
            raise ValueError(f"Tensors have different shapes: {shapes}")

        if B.shape(self.gt) != shapes[0]:
            if fix_swapped:
                if (shapes[0] == B.shape(self.gt)[::-1]) and (len(shapes[0]) == 2):
                    self.gt = B.reshape(self.gt, shapes[0])

            else:
                raise ValueError(
                    f"Ground truth tensor has different shape than input tensors: {shapes[0]} != {B.shape(self.gt)}"
                )

    def _initialize_pop(self):
        self.population, _ = regenerate_population(
            population=[],
            fitnesses=[],
            n=self.population_size,
            tensors=list(self.tensors.values()),
            gt=self.gt,
            ids=list(self.tensors.keys()),
            fitness_function=self.fitness_function,
            population_codes=None,
        )

    def _create_additional_population(self):
        additional_population = []

        ids = np.arange(self.population_size)  # should be done each time in case population size changes
        while len(additional_population) < self.population_size * self.population_multiplier:
            idxs = np.random.choice(ids, self.tournament_size, replace=False)
            tournament_fitnesses = self.fitnesses[idxs]
            parent_idxs = idxs[np.argsort(tournament_fitnesses)[-2:]]
            parent1, parent2 = self.population[parent_idxs[0]], self.population[parent_idxs[1]]

            try:
                child1, child2 = crossover(parent1, parent2, mutation_chance_crossover=self.mutation_chance_crossover)
            except Exception:
                # print("Crossover failed due to: ", e)
                continue

            child1.update_nodes()
            child2.update_nodes()

            additional_population += [child1, child2]

        return additional_population
