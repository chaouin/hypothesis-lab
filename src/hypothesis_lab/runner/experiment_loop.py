from hypothesis_lab.environments.pendulum import Pendulum
from hypothesis_lab.experiments.experiment import Experiment
from hypothesis_lab.experiments.observation import Observation
from hypothesis_lab.hypotheses.hypothesis import Hypothesis
from hypothesis_lab.hypotheses.scorer import score_hypotheses
from hypothesis_lab.agents.base import Agent
from hypothesis_lab.hypotheses.posterior import hypothesis_weights


class ExperimentLoop:

    def __init__(
            self,
            environment,
            agent,
            hypotheses,
            history,
            candidates,
            sigma: float = 0.1,
            confidence_threshold: float = 0.95
    ):
        self.environment = environment
        self.agent = agent
        self.hypotheses = hypotheses
        self.history = history
        self.candidates = candidates

        self.sigma = sigma
        self.confidence_threshold = confidence_threshold

    def run_one_step(self):
        scores_before = score_hypotheses(
            self.hypotheses,
            self.history
        )

        chosen_experiment = self.agent.choose_experiment(
            hypotheses=self.hypotheses,
            history=self.history,
            candidates=self.candidates
        )

        self.candidates.remove(chosen_experiment)

        chosen_period = self.environment.run_experiment(
            chosen_experiment.mass,
            chosen_experiment.length,
            chosen_experiment.gravity
        )

        chosen_observation = Observation(
            experiment=chosen_experiment,
            period=chosen_period
        )

        self.history.append(chosen_observation)

        scores_after = score_hypotheses(
            self.hypotheses,
            self.history
        )

        return chosen_observation, scores_before, scores_after

    def run(self, steps: int):
        results = []

        for _ in range(steps):
            if not self.candidates:
                break

            observation, scores_before, scores_after = (
                self.run_one_step()
            )

            results.append(
                (observation, scores_before, scores_after)
            )

            if self.should_stop(scores_after):
                break

        return results

    def should_stop(
            self,
            scores: list[tuple[Hypothesis, float, float]]
    ) -> bool:

        weights = hypothesis_weights(
            scores=scores,
            n_observations=len(self.history),
            sigma=self.sigma
        )

        best_weight = max(weights)

        return best_weight >= self.confidence_threshold