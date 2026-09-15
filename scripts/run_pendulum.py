import math

from hypothesis_lab.agents.greedy_agent import GreedyAgent
from hypothesis_lab.environments.pendulum import Pendulum
from hypothesis_lab.experiments.experiment import Experiment
from hypothesis_lab.experiments.observation import Observation
from hypothesis_lab.hypotheses.hypothesis import Hypothesis
from hypothesis_lab.hypotheses.scorer import score_hypotheses
from hypothesis_lab.runner.experiment_loop import ExperimentLoop
from hypothesis_lab.agents.disagreement_agent import DisagreementAgent
from hypothesis_lab.agents.information_gain_agent import InformationGainAgent



def constant(experiment: Experiment) -> float:
    return 1.0


def linear_length(experiment: Experiment) -> float:
    return experiment.length


def sqrt_length(experiment: Experiment) -> float:
    return math.sqrt(experiment.length)


def sqrt_length_over_gravity(experiment: Experiment) -> float:
    return math.sqrt(experiment.length / experiment.gravity)


if __name__ == "__main__":

    # -------------------------
    # Environment
    # -------------------------
    pendulum = Pendulum()

    # -------------------------
    # Hypotheses
    # -------------------------
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

    # -------------------------
    # Initial experiments
    # -------------------------
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

    # -------------------------
    # Build initial history
    # -------------------------
    history = []

    for experiment in initial_experiments:
        period = pendulum.run_experiment(
            experiment.mass,
            experiment.length,
            experiment.gravity
        )

        observation = Observation(
            experiment=experiment,
            period=period
        )

        history.append(observation)

    print("Initial observations:")

    for observation in history:
        print(observation)

    # -------------------------
    # Initial hypothesis scores
    # -------------------------
    initial_scores = score_hypotheses(
        hypotheses,
        history
    )

    print("\nInitial hypothesis scores:")

    for hypothesis, k, mse in initial_scores:
        print(
            hypothesis.name,
            "k =", k,
            "mse =", mse
        )

    # -------------------------
    # Candidate experiments
    # -------------------------
    candidates = [
        Experiment(
            id=4,
            mass=1.0,
            length=1.0,
            gravity=4.0
        ),
        Experiment(
            id=5,
            mass=1.0,
            length=1.0,
            gravity=2.0
        ),
        Experiment(
            id=6,
            mass=1.0,
            length=2.0,
            gravity=9.81
        ),
        Experiment(
            id=7,
            mass=10.0,
            length=1.0,
            gravity=9.81
        ),
    ]

    # -------------------------
    # Scientific agent
    # -------------------------
    agent = InformationGainAgent(sigma=0.5)

    print("\nCandidate information gain:")

    for candidate in candidates:
        information_gain = agent.information_gain(
            hypotheses=hypotheses,
            history=history,
            candidate=candidate
        )

        print(
            f"Experiment {candidate.id}: "
            f"mass={candidate.mass}, "
            f"length={candidate.length}, "
            f"gravity={candidate.gravity} "
            f"-> IG={information_gain:.4f} bits"
        )

    experiment_loop = ExperimentLoop(
        environment=pendulum,
        agent=agent,
        hypotheses=hypotheses,
        history=history,
        candidates=candidates
    )

    # -------------------------
    # Run one discovery step
    # -------------------------
    results = experiment_loop.run(steps=10)

    for step_number, result in enumerate(results, start=1):
        observation, scores_before, scores_after = result

        print(f"\n=== Discovery step {step_number} ===")

        print("\nChosen observation:")
        print(observation)

        print("\nUpdated hypothesis scores:")

        for hypothesis, k, mse in scores_after:
            print(
                hypothesis.name,
                "k =", k,
                "mse =", mse
            )