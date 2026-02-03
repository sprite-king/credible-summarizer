from dataclasses import dataclass


@dataclass
class Citation:
    sentence: str
    score: float
    index: int
