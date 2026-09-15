import random

from hypothesis_lab.agents.base import Agent
from hypothesis_lab.experiments.experiment import Experiment
from hypothesis_lab.experiments.observation import Observation
from hypothesis_lab.hypotheses.hypothesis import Hypothesis


class RandomAgent(Agent):

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

        return random.choice(available_candidates)