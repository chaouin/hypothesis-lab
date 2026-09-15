from abc import ABC, abstractmethod

from hypothesis_lab.experiments.experiment import Experiment
from hypothesis_lab.experiments.observation import Observation
from hypothesis_lab.hypotheses.hypothesis import Hypothesis


class Agent(ABC):

    @abstractmethod
    def choose_experiment(
        self,
        hypotheses: list[Hypothesis],
        history: list[Observation],
        candidates: list[Experiment]
    ) -> Experiment:
        pass