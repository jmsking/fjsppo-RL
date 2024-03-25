from project.common.manager import Manager
from project.common.storage import Storage

class Memory:
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.action_indices = []
        
        self.batch_opr_station = []
        self.batch_opr_pre = []
        self.batch_opr_next = []
        self.batch_indices = []
        self.batch_opr_features = []
        self.batch_station_features = []
        self.batch_edge_features = []
        self.eligible = []

        self.temperature = None

        self.manager = Manager()
        
    def update(self, om_pairs):
        """更新 O-M pair
        """
        self.om_pairs = om_pairs

    def register_storage(self, storage: Storage):
        self.storage = storage

    def record(self, state, action):
        """记录影响动作选择的相关信息
        """
        self.manager.statistic(state, action)
        
        
    def clear_memory(self):
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_indices[:]
        
        del self.batch_opr_station[:]
        del self.batch_opr_pre[:]
        del self.batch_opr_next[:]
        del self.batch_indices[:]
        del self.batch_opr_features[:]
        del self.batch_station_features[:]
        del self.batch_edge_features[:]
        del self.eligible[:]

        self.manager = Manager()