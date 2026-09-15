from hypothesis_lab.hypotheses.hypothesis import Hypothesis
from hypothesis_lab.experiments.observation import Observation


def fit_k(
        hypothesis: Hypothesis,
        history: list[Observation]
) -> float:
    numerator = 0.0
    denominator = 0.0

    for observation in history:
        fx = hypothesis.function(observation.experiment)

        numerator += fx * observation.period
        denominator += fx ** 2

    if denominator == 0:
        raise ValueError("Cannot fit k because denominator is zero.")

    return numerator / denominator


def mean_squared_error(
    hypothesis: Hypothesis,
    history: list[Observation],
    k: float
) -> float:
    total_error = 0.0

    for observation in history:
        prediction = hypothesis.predict(
            observation.experiment,
            k
        )

        error = observation.period - prediction
        total_error += error ** 2

    return total_error / len(history)


def score_hypotheses(
    hypotheses: list[Hypothesis],
    history: list[Observation]
) -> list[tuple[Hypothesis, float, float]]:

    scores = []

    for hypothesis in hypotheses:
        k = fit_k(hypothesis, history)
        mse = mean_squared_error(hypothesis, history, k)

        info = (hypothesis, k, mse)
        scores.append(info)

    scores.sort(key=lambda x: x[2])

    return scores