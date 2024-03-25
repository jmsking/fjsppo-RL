from project.utils.case_utils import CaseUtils
from project.common.constant import DIR

def random_produce():
        params = {
                'batch_size': 1,
                'n_jobs': 2,
                'n_stations': 3
        }
        CaseUtils.generate(**params)

def produce_from_file():
        # simulation data
        params_simulation_01 = {
            'source': 'simulation',
            'file_path': f'{DIR}/datasets/simulation/1005'
        }
        batch_oprs, batch_opr_stations, batch_opr_links = CaseUtils.generate(**params_simulation_01)
        for b in range(len(batch_oprs)):
                for opr_key, opr in batch_oprs[b].items():
                        station = batch_opr_stations[b][opr_key][0]
                        #print(station)
                        link = None
                        if opr_key in batch_opr_links[b]:
                                link = batch_opr_links[b][opr_key]
                        #print(opr.job_index)
                        if opr.job_index in (0,1,10):
                                print('========')
                                print(opr)
                                print(station)
                                print(link)

if __name__ == '__main__':
        produce_from_file()
