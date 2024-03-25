import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from project.common.memory import Memory
from project.common.constant import DIR

plt.rcParams['savefig.dpi'] = 100

class DrawUtils:
    """绘图工具包
    """
    @staticmethod
    def draw_station_gantt(batch_size: int, n_jobs: int, n_stations: int, color: list,
                batch_opr_schedule: torch.Tensor,
                n_total_opr: torch.Tensor,
                memory: Memory):
        """绘制工位甘特图
        """
        for batch_id in range(batch_size):
            schedules = batch_opr_schedule[batch_id].to('cpu')
            #plt.ylim(0, n_jobs+1)
            fig = plt.figure(figsize=(10, 6))
            fig.canvas.manager.set_window_title('Operation-Machine Gantt')
            axes = fig.add_axes([0.1, 0.1, 1.72, 1.8])
            y_ticks = []
            y_ticks_loc = []
            for i in range(n_stations):
                y_ticks.append('Machine {0}'.format(i))
                y_ticks_loc.append(i)
            labels = [''] * n_jobs
            for j in range(n_jobs):
                labels[j] = "Job {0}".format(j + 1)
            patches = [mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in range(n_jobs)]
            axes.cla()
            axes.set_title(u'FJSP')
            axes.grid(linestyle='-.', color='gray', alpha=0.2)
            axes.set_xlabel('time')
            axes.set_ylabel('machine')
            #axes.set_yticks(y_ticks_loc, y_ticks)
            plt.yticks(y_ticks_loc, y_ticks)
            axes.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=int(14 / pow(1, 0.3)))
            axes.set_ybound(1 - 1 / n_stations, n_stations + 1 / n_stations)
            for i in range(int(n_total_opr[batch_id])):
                if schedules[i, 2]+1 >= schedules[i, 3]:
                    continue
                opr_idx = i
                station_idx = int(schedules[opr_idx][1].item())
                job_idx = int(schedules[opr_idx][4].item())
                axes.barh(station_idx,
                            0.2,
                            left=schedules[opr_idx][2],
                            color=color[job_idx],
                            height=0.5)
                axes.barh(station_idx,
                            schedules[opr_idx][3] - schedules[opr_idx][2] - 0.2,
                            left=schedules[opr_idx][2]+0.2,
                            color=color[job_idx],
                            height=0.5)
            plt.savefig(f'{DIR}/results/benchmark_machine.png')
            plt.show()

    @staticmethod
    def draw_job_gantt(batch_size: int, n_jobs: int, n_stations: int, color: list,
                batch_opr_schedule: torch.Tensor,
                n_total_opr: torch.Tensor,
                memory: Memory):
        """绘制任务甘特图
        """
        #print(n_jobs)
        for batch_id in range(batch_size):
            schedules = batch_opr_schedule[batch_id].to('cpu')
            start_times = schedules[:, 2].squeeze()
            sort_times = torch.sort(start_times).indices
            #print(sort_times)
            #print(schedules[sort_times])
            part_schedules = schedules[sort_times]
            mask = torch.where(part_schedules[:, 2].squeeze()+1 >= part_schedules[:, 3].squeeze(), False, True)
            #print(mask)
            part_schedules = part_schedules[mask]
            sort_jobs = part_schedules[:, 4]
            #print(part_schedules)
            sort_jobs = torch.flip(torch.unique(sort_jobs, sorted=False), dims=[0]).numpy()
            #print(sort_jobs)
            mapping = {j : idx for idx, j in enumerate(sort_jobs)}
            #print(schedules[:, 2:4])
            #raise Exception()
            #plt.ylim(0, n_stations+1)
            fig = plt.figure(figsize=(10, 6))
            fig.canvas.manager.set_window_title('Operation-Job Gantt')
            axes = fig.add_axes([0.1, 0.1, 0.72, 0.8])
            y_ticks = []
            y_ticks_loc = []
            #for i in range(n_jobs):
            for i, v in enumerate(sort_jobs):
                y_ticks.append('Job {0}'.format(int(v+1)))
                y_ticks_loc.append(i)
            labels = [''] * n_stations
            for j in range(n_stations):
                labels[j] = "Machine {0}".format(j)
            patches = [mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in range(n_stations)]
            axes.cla()
            axes.set_title(u'FJSP')
            axes.grid(linestyle='-.', color='gray', alpha=0.2)
            axes.set_xlabel('time')
            axes.set_ylabel('job')
            #axes.set_yticks(y_ticks_loc, y_ticks)
            plt.yticks(y_ticks_loc, y_ticks)
            axes.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=int(14 / pow(1, 0.3)))
            axes.set_ybound(1 - 1 / n_stations, n_stations + 1 / n_stations)
            #cnt = 0
            for i in range(int(n_total_opr[batch_id])):
                if schedules[i, 2]+1 >= schedules[i, 3]:
                    continue
                opr_idx = i
                station_idx = int(schedules[opr_idx][1].item())
                job_idx = int(schedules[opr_idx][4].item())
                site = mapping[job_idx]
                #print(opr_idx, station_idx, job_idx, site)
                #raise Exception()
                axes.barh(site,
                            0.2,
                            left=schedules[opr_idx][2],
                            color=color[station_idx],
                            height=1.0)
                axes.barh(site,
                            schedules[opr_idx][3] - schedules[opr_idx][2] - 0.2,
                            left=schedules[opr_idx][2]+0.2,
                            color=color[station_idx],
                            height=1.0)
            plt.savefig(f'{DIR}/results/benchmark_job.png')
            plt.show()