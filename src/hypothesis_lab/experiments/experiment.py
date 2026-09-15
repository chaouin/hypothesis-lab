from dataclasses import dataclass

@dataclass
class Experiment:
    id: int
    mass: float
    length: float
    gravity: float

    def same_conditions(self, other: "Experiment") -> bool:
        return (
                self.mass == other.mass
                and self.length == other.length
                and self.gravity == other.gravity
        )
