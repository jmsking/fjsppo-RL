import torch
from project.simulator.state import State
from project.common.memory import Memory
from project.constraints.filter_factory import FilterFactory

class RuleFilter:
    """规则过滤器,影响候选动作的选择
    """
    @staticmethod
    def choose(eligible: torch.Tensor, state: State, memory: Memory):
        for filt in FilterFactory.get_all():
            eligible = filt(eligible, state, memory).choose()
        return eligible