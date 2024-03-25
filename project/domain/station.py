from dataclasses import dataclass
import datetime
from .calendar import Calendar

@dataclass
class Station:
    """工位实体
    Args
    ----
    station_key: 工位唯一key
    calendars: 工位日历
    index: 工位索引
    """
    station_key: str
    calendar: Calendar
    index: int
        
    def __post_init__(self):
        """构建工位特征
        """
        self.features = [
            
        ]
        self._update_progress()
        
    def _update_progress(self):
        """更新进度
        """
        if not self.calendar:
            return
        self.last_finish_time = None
        self.next_day_start = None
        if len(self.calendar.periods) > 0:
            # 工位上最后一个工序的完工时间
            self.last_finish_time = self.calendar.periods[0].start_dt
        if len(self.calendar.periods) > 1:
            # 工位上最后一个工序的下一天最开始时间
            self.next_day_start = self.calendar.periods[1].start_dt
        
    def update(self, process_time: int):
        """更新相关信息
        Args
        --------
        process_time: 处理时间
        """
        self.calendar.update(process_time)
        self._update_progress()
        
    @property
    def time_start_index(self):
        """时间开始索引
        """
        return self.calendar.time_start_index
        
    @property
    def time_end_index(self):
        """时间结束索引
        """
        return self.calendar.time_end_index
    
    def to_json(self):
        return {
            'station_key': self.station_key,
            'calendar': self.calendar.to_json(),
            'index': self.index
        }
        
    def __eq__(self, o):
        return self.station_key == o.station_key
    
    def __hash__(self):
        return hash(self.station_key)
        
    def __repr__(self):
        return f'key: {self.station_key}, index: {self.index}'