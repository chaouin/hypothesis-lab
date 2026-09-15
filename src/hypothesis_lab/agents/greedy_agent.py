from hypothesis_lab.agents.base import Agent
from hypothesis_lab.experiments.experiment import Experiment
from hypothesis_lab.hypotheses.hypothesis import Hypothesis
from hypothesis_lab.experiments.observation import Observation
from hypothesis_lab.hypotheses.scorer import score_hypotheses


class GreedyAgent(Agent):

    def choose_experiment(
        self,
        hypotheses: list[Hypothesis],
        history: list[Observation],
        candidates: list[Experiment]
    ) -> Experiment:

        scores = score_hypotheses(hypotheses, history)

        best_hypothesis, best_k, _ = scores[0]
        second_hypothesis, second_k, _ = scores[1]

        best_candidate = None
        largest_difference = -1.0

        for candidate in candidates:
            prediction1 = best_hypothesis.predict(
                candidate,
                best_k
            )

            prediction2 = second_hypothesis.predict(
                candidate,
                second_k
            )

            difference = abs(prediction1 - prediction2)

            if difference > largest_difference:
                largest_difference = difference
                best_candidate = candidate

        return best_candidate