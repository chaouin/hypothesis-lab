from dataclasses import dataclass
from typing import Callable

from hypothesis_lab.experiments.experiment import Experiment


@dataclass
class Hypothesis:
    name: str
    function: Callable[[Experiment], float]

    def predict(self, experiment: Experiment, k: float) -> float:
        period = k * self.function(experiment)
        return period