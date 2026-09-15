from dataclasses import dataclass

from hypothesis_lab.experiments.experiment import Experiment


@dataclass
class Observation:
    experiment: Experiment
    period:float
