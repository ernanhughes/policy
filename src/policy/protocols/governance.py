# policy/protocols/governance.py

from typing import Protocol

class GovernanceEngine(Protocol):
    def assess(self, bundle: dict) -> dict:
        """
        Returns governance metrics:
            {
                "energy": float,
                "regime": str,
                "dominates": bool,
                ...
            }
        """
        ...
