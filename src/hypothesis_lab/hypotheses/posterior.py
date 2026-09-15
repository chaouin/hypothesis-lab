import math

from hypothesis_lab.hypotheses.hypothesis import Hypothesis


def hypothesis_weights(
    scores: list[tuple[Hypothesis, float, float]],
    n_observations: int,
    sigma: float = 0.5
) -> list[float]:

    log_weights = []

    for hypothesis, k, mse in scores:
        sse = mse * n_observations

        log_weight = -sse / (2 * sigma ** 2)
        log_weights.append(log_weight)

    max_log_weight = max(log_weights)

    weights = [
        math.exp(log_weight - max_log_weight)
        for log_weight in log_weights
    ]

    total = sum(weights)

    return [
        weight / total
        for weight in weights
    ]

def entropy(weights: list[float]) -> float:
    result = 0.0

    for weight in weights:
        if weight > 0:
            result -= weight * math.log2(weight)

    return result