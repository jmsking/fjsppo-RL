from typing import List

class CalendarUtils:
    """日历相关工具包
    """
    @staticmethod
    def align_time(stations: List[int], time: int):
        """将时间步与工位的日历时间相对应
        NOTE: 目前假定每个工位的的日历时间保持一致
        Args
        -----
        stations: 工位索引
        time: 当前时间步
        """
        pass