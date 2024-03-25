from typing import List
from project.domain.station import Station
from project.common.storage import Storage

class StationUtils:
    """工位相关工具包
    """
    @staticmethod
    def obtain_station_by_idx(n_stations: List[int], storage: Storage):
        """根据工位索引获取工位对象
        Args
        ------
        n_stations: 每批次工位数
        station_idx: 工位索引
        """
        batch_stations = []
        batch_idx_station = storage.batch_idx_station
        for b in range(len(n_stations)):
            for s in range(n_stations):
                pass
            