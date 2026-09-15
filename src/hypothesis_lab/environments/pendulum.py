import math
import random


class Pendulum:

    def __init__(
        self,
        noise_std: float = 0.0,
        seed: int | None = None
    ):
        if noise_std < 0:
            raise ValueError("noise_std cannot be negative")

        self.noise_std = noise_std
        self.rng = random.Random(seed)

    def run_experiment(
        self,
        mass: float,
        length: float,
        gravity: float
    ) -> float:

        if mass <= 0:
            raise ValueError("mass must be greater than 0")

        if length <= 0:
            raise ValueError("length must be greater than 0")

        if gravity <= 0:
            raise ValueError("gravity must be greater than 0")

        true_period = (
            2
            * math.pi
            * math.sqrt(length / gravity)
        )

        noise = self.rng.gauss(
            0.0,
            self.noise_std
        )

        return true_period + noise