import csv
import json
import math
import random
import statistics
from pathlib import Path

from hypothesis_lab.agents.disagreement_agent import DisagreementAgent
from hypothesis_lab.agents.greedy_agent import GreedyAgent
from hypothesis_lab.agents.information_gain_agent import InformationGainAgent
from hypothesis_lab.agents.random_agent import RandomAgent
from hypothesis_lab.environments.pendulum import Pendulum
from hypothesis_lab.experiments.experiment import Experiment
from hypothesis_lab.experiments.observation import Observation
from hypothesis_lab.hypotheses.hypothesis import Hypothesis
from hypothesis_lab.hypotheses.scorer import score_hypotheses
from hypothesis_lab.runner.experiment_loop import ExperimentLoop


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# Hypothesis functions
# ============================================================

def constant(experiment: Experiment) -> float:
    return 1.0


def linear_length(experiment: Experiment) -> float:
    return experiment.length


def sqrt_length(experiment: Experiment) -> float:
    return math.sqrt(experiment.length)


def sqrt_length_over_gravity(experiment: Experiment) -> float:
    return math.sqrt(
        experiment.length / experiment.gravity
    )


# ============================================================
# Scenario
# ============================================================

def build_scenario(
    seed: int,
    noise_std: float
):
    """
    Build a fresh pendulum discovery scenario.

    Initial observations vary mass and length while gravity
    stays fixed at 9.81.

    This makes sqrt(length) and sqrt(length / gravity)
    initially difficult to distinguish.
    """

    pendulum = Pendulum(
        noise_std=noise_std,
        seed=seed
    )

    hypotheses = [
        Hypothesis(
            name="constant",
            function=constant
        ),
        Hypothesis(
            name="linear_length",
            function=linear_length
        ),
        Hypothesis(
            name="sqrt_length",
            function=sqrt_length
        ),
        Hypothesis(
            name="sqrt_length_over_gravity",
            function=sqrt_length_over_gravity
        ),
    ]

    # --------------------------------------------------------
    # Initial evidence
    # --------------------------------------------------------

    initial_experiments = [
        Experiment(
            id=1,
            mass=1.0,
            length=1.0,
            gravity=9.81
        ),
        Experiment(
            id=2,
            mass=10.0,
            length=1.0,
            gravity=9.81
        ),
        Experiment(
            id=3,
            mass=1.0,
            length=4.0,
            gravity=9.81
        ),
    ]

    history = []

    for experiment in initial_experiments:
        period = pendulum.run_experiment(
            experiment.mass,
            experiment.length,
            experiment.gravity
        )

        history.append(
            Observation(
                experiment=experiment,
                period=period
            )
        )

    # --------------------------------------------------------
    # Candidate experiments
    # --------------------------------------------------------

    candidates = [
        # Gravity changes
        Experiment(4, 1.0, 1.0, 6.0),
        Experiment(5, 1.0, 1.0, 7.0),
        Experiment(6, 1.0, 1.0, 8.0),
        Experiment(7, 1.0, 1.0, 10.5),
        Experiment(8, 1.0, 1.0, 12.0),

        # Length changes
        Experiment(9, 1.0, 2.0, 9.81),
        Experiment(10, 1.0, 3.0, 9.81),
        Experiment(11, 1.0, 5.0, 9.81),
        Experiment(12, 1.0, 0.5, 9.81),

        # Mass changes
        Experiment(13, 2.0, 1.0, 9.81),
        Experiment(14, 5.0, 1.0, 9.81),
        Experiment(15, 8.0, 1.0, 9.81),
    ]

    return (
        pendulum,
        hypotheses,
        history,
        candidates
    )


# ============================================================
# Agent factory
# ============================================================

def create_agent(
    agent_name: str,
    sigma: float
):
    if agent_name == "random":
        return RandomAgent()

    if agent_name == "greedy":
        return GreedyAgent()

    if agent_name == "disagreement":
        return DisagreementAgent()

    if agent_name == "information_gain":
        return InformationGainAgent(
            sigma=sigma
        )

    raise ValueError(
        f"Unknown agent: {agent_name}"
    )


# ============================================================
# Run one agent
# ============================================================

def run_agent(
    agent,
    seed: int,
    noise_std: float,
    max_steps: int = 10
) -> dict:

    random.seed(seed)

    pendulum, hypotheses, history, candidates = (
        build_scenario(
            seed=seed,
            noise_std=noise_std
        )
    )

    # Avoid sigma = 0 in the probabilistic model.
    inference_sigma = max(
        noise_std,
        1e-6
    )

    experiment_loop = ExperimentLoop(
        environment=pendulum,
        agent=agent,
        hypotheses=hypotheses,
        history=history,
        candidates=candidates,
        sigma=inference_sigma,
        confidence_threshold=0.95
    )

    results = experiment_loop.run(
        steps=max_steps
    )

    final_scores = score_hypotheses(
        hypotheses,
        history
    )

    best_hypothesis, best_k, best_mse = (
        final_scores[0]
    )

    success = (
        best_hypothesis.name
        == "sqrt_length_over_gravity"
    )

    chosen_experiment_ids = [
        observation.experiment.id
        for observation, _, _ in results
    ]

    return {
        "success": success,
        "experiments": len(results),
        "best_hypothesis": best_hypothesis.name,
        "best_k": best_k,
        "best_mse": best_mse,
        "chosen_experiments": chosen_experiment_ids,
    }


# ============================================================
# Benchmark one agent at one noise level
# ============================================================

def benchmark_agent(
    agent_name: str,
    noise_std: float,
    n_runs: int = 100,
    max_steps: int = 10
) -> dict:

    successes = 0
    experiment_counts = []
    run_results = []

    inference_sigma = max(
        noise_std,
        1e-6
    )

    for seed in range(n_runs):

        agent = create_agent(
            agent_name=agent_name,
            sigma=inference_sigma
        )

        result = run_agent(
            agent=agent,
            seed=seed,
            noise_std=noise_std,
            max_steps=max_steps
        )

        experiment_counts.append(
            result["experiments"]
        )

        if result["success"]:
            successes += 1

        run_results.append({
            "agent": agent_name,
            "noise_std": noise_std,
            "seed": seed,
            "success": result["success"],
            "experiments": result["experiments"],
            "best_hypothesis": result["best_hypothesis"],
            "best_k": result["best_k"],
            "best_mse": result["best_mse"],
            "chosen_experiments": result["chosen_experiments"],
        })

    return {
        "agent": agent_name,
        "noise_std": noise_std,
        "n_runs": n_runs,
        "max_steps": max_steps,
        "success_rate": successes / n_runs,
        "mean_experiments": statistics.mean(
            experiment_counts
        ),
        "median_experiments": statistics.median(
            experiment_counts
        ),
        "min_experiments": min(
            experiment_counts
        ),
        "max_experiments": max(
            experiment_counts
        ),
        "run_results": run_results,
    }


# ============================================================
# Save results
# ============================================================

def save_results(
    benchmark_results: list[dict]
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Aggregate summary
    # --------------------------------------------------------

    summary_path = (
        RESULTS_DIR
        / "benchmark_summary.csv"
    )

    summary_fields = [
        "agent",
        "noise_std",
        "n_runs",
        "max_steps",
        "success_rate",
        "mean_experiments",
        "median_experiments",
        "min_experiments",
        "max_experiments",
    ]

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields
        )

        writer.writeheader()

        for result in benchmark_results:
            writer.writerow({
                field: result[field]
                for field in summary_fields
            })

    # --------------------------------------------------------
    # Individual runs
    # --------------------------------------------------------

    runs_path = (
        RESULTS_DIR
        / "benchmark_runs.csv"
    )

    run_fields = [
        "agent",
        "noise_std",
        "seed",
        "success",
        "experiments",
        "best_hypothesis",
        "best_k",
        "best_mse",
        "chosen_experiments",
    ]

    with runs_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=run_fields
        )

        writer.writeheader()

        for benchmark_result in benchmark_results:

            for run in benchmark_result["run_results"]:

                row = run.copy()

                row["chosen_experiments"] = (
                    json.dumps(
                        row["chosen_experiments"]
                    )
                )

                writer.writerow(row)

    print(
        f"\nSaved summary to:"
        f"\n{summary_path}"
    )

    print(
        f"\nSaved individual runs to:"
        f"\n{runs_path}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    agent_names = [
        "random",
        "greedy",
        "disagreement",
        "information_gain",
    ]

    noise_levels = [
        0.0,
        0.1,
        0.25,
        0.5,
        1.0,
    ]

    n_runs = 100
    max_steps = 10

    print(
        "=========================================="
    )
    print(
        "Hypothesis Lab - Agent Benchmark"
    )
    print(
        "=========================================="
    )

    print(
        f"\nRuns per condition: {n_runs}"
    )

    print(
        f"Maximum experiments per run: "
        f"{max_steps}"
    )

    benchmark_results = []

    for noise_std in noise_levels:

        print(
            "\n------------------------------------------"
        )
        print(
            f"Noise std: {noise_std}"
        )
        print(
            "------------------------------------------"
        )

        for agent_name in agent_names:

            result = benchmark_agent(
                agent_name=agent_name,
                noise_std=noise_std,
                n_runs=n_runs,
                max_steps=max_steps
            )

            benchmark_results.append(
                result
            )

            print(
                f"{agent_name:18s} | "
                f"Success: "
                f"{result['success_rate']:7.2%} | "
                f"Mean exp: "
                f"{result['mean_experiments']:5.2f}"
            )

    # --------------------------------------------------------
    # Save benchmark outputs
    # --------------------------------------------------------

    save_results(
        benchmark_results
    )

    # --------------------------------------------------------
    # Detailed console results
    # --------------------------------------------------------

    print(
        "\n\n=========================================="
    )
    print(
        "Detailed Results"
    )
    print(
        "=========================================="
    )

    for result in benchmark_results:

        print(
            f"\nAgent: {result['agent']}"
        )

        print(
            f"Noise std: "
            f"{result['noise_std']}"
        )

        print(
            "Success rate: "
            f"{result['success_rate']:.2%}"
        )

        print(
            "Mean experiments: "
            f"{result['mean_experiments']:.2f}"
        )

        print(
            "Median experiments: "
            f"{result['median_experiments']:.1f}"
        )

        print(
            "Min experiments: "
            f"{result['min_experiments']}"
        )

        print(
            "Max experiments: "
            f"{result['max_experiments']}"
        )