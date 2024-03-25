from project.utils.state_utils import StateUtils
from project.utils.case_utils import CaseUtils

params = {
    'batch_size': 1,
    'n_jobs': 2,
    'n_stations': 3
}
batch_oprs, batch_opr_stations, batch_opr_links = CaseUtils.generate(**params)

source_data = StateUtils.build_features(batch_oprs, batch_opr_stations, batch_opr_links)
for i in range(len(source_data)):
    print(source_data[i].shape)
    print(source_data[i])
    print('-----------------')
