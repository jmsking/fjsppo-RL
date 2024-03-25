from dataclasses import dataclass
import numpy as np
import torch
from project.simulator.action import Action
from project.common.memory import Memory
from project.constraints._constraint import _Constraint

class MaxContiConstraint(_Constraint):
    """最大连续约束下的动作选择
    """
    def __init__(self, eligible, state, memory):
        super().__init__(eligible, state, memory)

    def choose(self):
        return self.eligible
