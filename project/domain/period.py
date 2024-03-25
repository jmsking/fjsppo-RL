from dataclasses import dataclass
import datetime

SECONDS_OF_DAY = 24 * 60 * 60
DT_FMT = '%Y-%m-%d %H:%M:%S'

@dataclass
class Period:
    """时间段
    Args
    ---------
    start_dt: 开始时间
    end_dt: 结束时间
    """
    start_dt: datetime.datetime
    end_dt: datetime.datetime    
    
    def __post_init__(self):
        """初始化必要信息
        """
        self.init()
        
    def init(self):
        duration = self.end_dt - self.start_dt
        self.interval = duration.days * SECONDS_OF_DAY + duration.seconds
        
    def to_json(self):
        return {
            'start_dt': self.start_dt.strftime(DT_FMT),
            'end_dt': self.end_dt.strftime(DT_FMT)
        }
        
    def __sub__(self, val: int):
        """重载运算符 '-'
        """
        remain = self.interval - val
        if remain >= 0:
            self.start_dt = self.start_dt + datetime.timedelta(seconds=val)
        else:
            self.start_dt = self.end_dt
        self.init()
        return remain
    
    def __repr__(self):
        return f'{self.start_dt.strftime(DT_FMT)} -> {self.end_dt.strftime(DT_FMT)} total: {self.interval} (s)'