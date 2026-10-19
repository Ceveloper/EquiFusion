import torch
from tabulate import tabulate
import numpy as np

def format_metric_time_table(metric):
    # dim 0 is time
    interval = 30 # FPS/2
    table_timesteps = [i*interval for i in range(16) if i*interval< len(metric)]
    metric = torch.stack([metric[t] for t in table_timesteps], dim=0) # + [metric.mean(0)]
    return metric

def choose_best_sample(out, y):
    kpts3d = out
    indeces = torch.linalg.norm(out - y.unsqueeze(1), dim=-1).mean(-1).mean(-1).min(axis=-1).indices.cpu().numpy()
    indeces_bool = torch.zeros(out.shape[0], out.shape[1], dtype=bool)
    for b, idx in enumerate(indeces):
        indeces_bool[b,idx] = True
    kpts3d = kpts3d[indeces_bool]
    assert kpts3d.shape == y.shape
    return kpts3d, y



def draw_table(results, if_consider_hip=False):
    n_subcolumns = 3 if if_consider_hip else 1
    metric_names_ptype = { 'ADE': [None]*n_subcolumns, 'FDE': [None]*n_subcolumns, 'MAE': [None]*n_subcolumns, 
                          'MMADE': [None]*n_subcolumns, 'MMFDE': [None]*n_subcolumns, 'APDE': [None]*n_subcolumns, 'APD': [None]*n_subcolumns, 'CMD': [None]*n_subcolumns, 'BodyR-mean': [None]*n_subcolumns, 'BodyR-RMSE': [None]*n_subcolumns, 
                          }
    table_header = list(metric_names_ptype.keys()) #['Metric'] + 
    if_hip = any(["traj" in m or "pose" in m for m in results.keys()])

    for stats in results:
        for m in table_header:
            if m==stats:
                metric_names_ptype[m][0] = f"{results[stats][-1]:.4f}" if isinstance(results[stats], list) else f"{results[stats]:.4f}"
            else:
                continue
    metric_names_ptype['BodyR-mean'][0] =  str(round(results['StretchMean'], 3)) + "  |  " + str(round(results['JitterMean'], 3))
    metric_names_ptype['BodyR-RMSE'][0] =  str(round(results['StretchRMSE'], 3)) + "  |  " + str(round(results['JitterRMSE'], 3))
    if n_subcolumns > 1:
        table = [[["total"]] +[metric_names_ptype[m] for m in table_header]]
    else:
        table = [["total"] +[metric_names_ptype[m][0] for m in table_header]]
    table_header = ['Metric'] + table_header
    return tabulate(table, headers=table_header, tablefmt="grid")
