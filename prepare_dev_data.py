from project.utils.case_utils import CaseUtils


env_paras = {
    "n_jobs": 10,
    "n_stations": 8,
    "batch_size": 100,
    "is_train": True,
    "no_action": True,
    "is_single_station": True,
    "exist_subline": True,
    "is_save": True
}
batch_oprs, batch_opr_stations, batch_opr_links = CaseUtils.generate(**env_paras)