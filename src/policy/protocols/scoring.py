# policy/protocols/scoring.py
from typing import Protocol

class ScoreProvider(Protocol):
    def evaluate(self, output: object, context: dict) -> dict:
        ...
