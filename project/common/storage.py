from dataclasses import dataclass
from typing import List
import collections

@dataclass
class Storage:
    """存储各种字典,以便进行高效查询
    Args
    ------
    batch_oprs: 工序信息
    batch_opr_stations: 工序工位对应信息
    batch_opr_links: 工序链接信息
    """
    batch_oprs: List[dict]
    batch_opr_stations: List[dict]
    batch_opr_links: List[dict]

    def __post_init__(self):
        self.batch_size = len(self.batch_oprs)
        self._build_idx_station()
    
    def _build_idx_station(self):
        """构建索引与工位对象字典
        """
        self.batch_idx_station = [{} for _ in range(self.batch_size)]
        stations = []
        for b in range(self.batch_size):
            self.batch_idx_station[b].update({
                item.index : item for alloc_stations in self.batch_opr_stations[b].values() for item in alloc_stations
            })