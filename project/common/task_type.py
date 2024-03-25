from enum import Enum

class TaskType(Enum):
    # 正常任务
    normal_task = 0
    # 空节拍任务
    skip_beat_task = 1
    # 跨天任务
    cross_day_task = 2