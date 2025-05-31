"""
Trivial observation operator: H(x) = x
"""

class IdentityObservationOperator:
    def apply(self, state_tensor):
        return state_tensor
