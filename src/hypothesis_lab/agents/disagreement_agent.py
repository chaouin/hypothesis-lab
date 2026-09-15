from hypothesis_lab.agents.base import Agent
from hypothesis_lab.experiments.experiment import Experiment
from hypothesis_lab.experiments.observation import Observation
from hypothesis_lab.hypotheses.hypothesis import Hypothesis
from hypothesis_lab.hypotheses.scorer import score_hypotheses


class DisagreementAgent(Agent):

    def choose_experiment(
        self,
        hypotheses: list[Hypothesis],
        history: list[Observation],
        candidates: list[Experiment]
    ) -> Experiment:

        scores = score_hypotheses(
            hypotheses,
            history
        )

        best_candidate = None
        best_disagreement = -1.0

        for candidate in candidates:
            predictions = []

            for hypothesis, k, mse in scores:
                prediction = hypothesis.predict(
                    candidate,
                    k
                )

                predictions.append(prediction)

            disagreement = self.compute_disagreement(
                predictions
            )

            if disagreement > best_disagreement:
                best_disagreement = disagreement
                best_candidate = candidate

        return best_candidate

    def compute_disagreement(
            self,
            predictions: list[float]
    ) -> float:

        total_difference = 0.0
        pair_count = 0

        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                difference = abs(
                    predictions[i] - predictions[j]
                )

                total_difference += difference
                pair_count += 1

        if pair_count == 0:
            return 0.0

        return total_difference / pair_count