from hypothesis_lab.agents.base import Agent
from hypothesis_lab.experiments.experiment import Experiment
from hypothesis_lab.experiments.observation import Observation
from hypothesis_lab.hypotheses.hypothesis import Hypothesis
from hypothesis_lab.hypotheses.scorer import score_hypotheses
from hypothesis_lab.hypotheses.posterior import (
    hypothesis_weights,
    entropy
)


class InformationGainAgent(Agent):

    def __init__(self, sigma: float = 0.5):
        self.sigma = sigma

    def information_gain(
        self,
        hypotheses: list[Hypothesis],
        history: list[Observation],
        candidate: Experiment
    ) -> float:

        current_scores = score_hypotheses(
            hypotheses,
            history
        )

        current_weights = hypothesis_weights(
            scores=current_scores,
            n_observations=len(history),
            sigma=self.sigma
        )

        current_entropy = entropy(current_weights)

        expected_entropy = 0.0

        for (
            hypothesis,
            k,
            _
        ), hypothesis_weight in zip(
            current_scores,
            current_weights
        ):

            hypothetical_period = hypothesis.predict(
                candidate,
                k
            )

            hypothetical_observation = Observation(
                experiment=candidate,
                period=hypothetical_period
            )

            hypothetical_history = history + [
                hypothetical_observation
            ]

            hypothetical_scores = score_hypotheses(
                hypotheses,
                hypothetical_history
            )

            posterior_weights = hypothesis_weights(
                scores=hypothetical_scores,
                n_observations=len(hypothetical_history),
                sigma=self.sigma
            )

            posterior_entropy = entropy(
                posterior_weights
            )

            expected_entropy += (
                hypothesis_weight
                * posterior_entropy
            )

        return current_entropy - expected_entropy

    def choose_experiment(
        self,
        hypotheses: list[Hypothesis],
        history: list[Observation],
        candidates: list[Experiment]
    ) -> Experiment:

        available_candidates = []

        for candidate in candidates:
            already_observed = any(
                candidate.same_conditions(observation.experiment)
                for observation in history
            )

            if not already_observed:
                available_candidates.append(candidate)

        if not available_candidates:
            raise ValueError("No new candidate experiments available.")

        best_candidate = available_candidates[0]
        best_information_gain = float("-inf")

        for candidate in available_candidates:
            information_gain = self.information_gain(
                hypotheses=hypotheses,
                history=history,
                candidate=candidate
            )

            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_candidate = candidate

        return best_candidate