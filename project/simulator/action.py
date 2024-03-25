from dataclasses import dataclass
import numpy as np
import torch
import types
from project.utils.key_utils import KeyUtils
from project.common.task_type import TaskType

@dataclass
class Action:
    """动作实体
    Args
    ----------
    opr_station_pair: 工序工位对 shape -> (batch_size, 3)
        第一个特征: 工序索引
        第二个特征: 工位索引
        第三个特征: 工序所属任务索引
    """
    opr_station_pair: torch.Tensor

    def t(self):
        """张量转置
        """
        self.opr_station_pair = self.opr_station_pair.t()
        return self
        
    def __repr__(self):
        return f'action shape: {self.opr_station_pair.shape}'