from dataclasses import dataclass
from abc import abstractmethod
import torch
from project.simulator.state import State
from project.common.memory import Memory

@dataclass
class _Constraint:
    """所有约束对象的基类
    Args
    --------
    eligible: 合法的工序工位对, true: 合法, false: 不合法, shape -> (batch_size, n_oprs, n_stations)
    state: 状态
    memory: 记忆信息
    """
    eligible: torch.Tensor
    state: State
    memory: Memory

    @abstractmethod
    def choose(self):
        raise NotImplementedError('Need implement by subclass')