from argparse import ArgumentParser
from pathlib import Path

from giraffe.fitness import FitnessFunction, distance_optimal_f1_score_fitness
from giraffe.giraffe import Giraffe
from giraffe.node import MeanNode, MaxNode, MinNode
from giraffe.callback import EarlyStoppingCallback, SaveParetoCallback


def load_args():
    parser = ArgumentParser()
    parser.add_argument("--input_path", default="./test_probs")
    parser.add_argument("--gt_path", default="./gt/test_labels.pt")
    parser.add_argument("--population_size", type=int, default=20)
    parser.add_argument("--population_multiplier", type=int, default=1)
    parser.add_argument("--tournament_size", type=int, default=5)
    parser.add_argument(
        "--fitness_function",
        type=lambda func: FitnessFunction[func].value,
        choices=list(FitnessFunction),
        default="AVERAGE_PRECISION",
    )
    parser.add_argument("--allow_all_ops", action="store_true")
    parser.add_argument("--use_distance_optimal_threshold", action="store_true")
    parser.add_argument("--mutation_chance_crossover", action="store_true")
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    return (
        Path(args.input_path),
        Path(args.gt_path),
        args.population_size,
        args.population_multiplier,
        args.tournament_size,
        args.fitness_function,
        args.allow_all_ops,
        args.use_distance_optimal_threshold,
        args.mutation_chance_crossover,
        args.seed,
    )


def main(
    input_path,
    gt_path,
    population_size,
    population_multiplier,
    tournament_size,
    fitness_function,
    allow_all_ops,
    use_distance_optimal_threshold,
    mutation_chance_crossover,
    seed,
):
    if (fitness_function is FitnessFunction.F1_SCORE) and use_distance_optimal_threshold:
        fitness_function = distance_optimal_f1_score_fitness

    allowed_ops = (MeanNode, MaxNode, MinNode) if allow_all_ops else (MeanNode,)

    early_stopping_callback = EarlyStoppingCallback(patience=10)
    save_pareto_callback = SaveParetoCallback(path="./", filename="tree")

    giraffe = Giraffe(
        preds_source=input_path,
        gt_path=gt_path,
        population_size=population_size,
        population_multiplier=population_multiplier,
        tournament_size=tournament_size,
        fitness_function=fitness_function,
        allowed_op_nodes=allowed_ops,
        mutation_chance_crossover=mutation_chance_crossover,
        seed=seed,
        callbacks=[early_stopping_callback, save_pareto_callback],
    )

    giraffe.train(iterations=200)
    print(giraffe.fitnesses)


if __name__ == "__main__":
    (
        input_path,
        gt_path,
        population_size,
        population_multiplier,
        tournament_size,
        fitness_function,
        allow_all_ops,
        use_distance_optimal_threshold,
        mutation_chance_crossover,
        seed,
    ) = load_args()

    main(
        input_path,
        gt_path,
        population_size,
        population_multiplier,
        tournament_size,
        fitness_function,
        allow_all_ops,
        use_distance_optimal_threshold,
        mutation_chance_crossover,
        seed,
    )
