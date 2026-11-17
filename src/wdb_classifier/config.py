"""A small, explicit scientific configuration; no embedded machine-specific paths."""

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    features: str
    schema: str
    output: str
    assignments: str = ""
    family: str = "random_forest"
    seed: int = 20261023
    outer_folds: int = 5
    inner_folds: int = 3
    budgets: list[float] = field(default_factory=lambda: [0.01, 0.05])
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        if (
            self.outer_folds < 2
            or self.inner_folds < 2
            or not self.budgets
            or len(set(self.budgets)) != len(self.budgets)
            or any(not 0 < b < 1 for b in self.budgets)
        ):
            raise ValueError("Use >=2 folds and distinct background budgets between zero and one")
        if self.family not in ("random_forest", "extra_trees"):
            raise ValueError("This initial release supports random_forest and extra_trees")

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        value = tomllib.loads(path.read_text())["wdb"]
        for name in ("features", "schema", "output", "assignments"):
            if value.get(name):
                value[name] = str((path.parent / Path(value[name]).expanduser()).resolve())
        return cls(**value)

    def as_dict(self):
        return asdict(self)
