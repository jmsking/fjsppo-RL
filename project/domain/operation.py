from dataclasses import dataclass, field

@dataclass
class Operation:
    """工序实体
    Args
    ----
    opr_key: 工序唯一Key
    material_no: 物料编码
    index: 工序索引
    job_index: 所属任务(订单)的索引
    line_index: 所属线(主线或部装线)的索引
    process_time: 工序到工位的加工时长(单位: 秒)
    is_mainline: 是否是主线工序
    ref_opr_key: 主线与部装线的交叉工序
    is_first: 是否是首工序
    """
    opr_key: str
    material_no: str
    index: int
    job_index: int
    line_index: int
    process_time: dict = field(default_factory=dict)
    is_mainline: bool = True
    ref_opr_key: str = None
    is_first: bool = False
        
    def to_json(self):
        return {
            'opr_key': self.opr_key,
            'process_time': self.process_time,
            'material_no': self.material_no,
            'index': self.index,
            'job_index': self.job_index,
            'line_index': self.line_index,
            'is_mainline': self.is_mainline,
            'ref_opr_key': self.ref_opr_key
        }
        
    def __eq__(self, o):
        return self.opr_key == o.opr_key
    
    def __hash__(self):
        return hash(self.opr_key)
        
    def __repr__(self):
        return f'{self.opr_key}: index: {self.index}, \
    time: {self.process_time} (s), mainline: {self.is_mainline}, line index: {self.line_index}'