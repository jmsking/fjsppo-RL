class KeyUtils:
    """生成Key值的工具包
    """
    
    @staticmethod
    def gen_action_key(opr_key: str, station_key: str):
        return f'ACT_{opr_key}_{station_key}'