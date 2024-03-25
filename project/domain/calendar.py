from dataclasses import dataclass, field
from typing import List
import datetime
from .period import Period

@dataclass
class Calendar:
    """日历实体
    Args
    -------
    periods: 时间段
    time_start_index: 时间开始索引(不包含)
    time_end_index: 时间结束索引(包含)
    """
    periods: List[Period]
    time_start_index: int = -1
    time_end_index: int = -1
        
    def __post_init__(self):
        self.interval = sum(list(map(lambda x : x.interval, self.periods)))
        
    def update(self, process_time: int):
        """更新日历时间
        Args
        ---------
        process_time: 加工时长,单位: 秒
        """
        start = self.periods[0].start_dt
        for period in self.periods:
            remain = period - process_time
            # 找到可覆盖加工时长的时间段
            if remain >= 0:
                break
            process_time -= period.interval
        self.periods = list(filter(lambda x : x.interval > 0, self.periods))
        self.time_start_index = self.time_end_index
        duration = self.periods[0].start_dt - start
        self.time_end_index = duration.days * 24 * 60 * 60 + duration.seconds
        
    def to_json(self):
        return {
            'periods': [period.to_json() for period in self.periods]
        } 