from copy import deepcopy
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd 
from tqdm import tqdm

def get_cfg_clusters(run_lst, cluster_cfg_lst):
    run_lst_cp = deepcopy(run_lst)
    cfgs= [run_lst_cp[ix].config for ix in range(len(run_lst_cp))]


    data_dir_lst = list()
    for ix in range(len(cfgs)):
        for key in cfgs[ix].keys():
            if "data.datamodule.root_dirs" in key or "data.datamodule.device" in key or "exp_id" in key:
                data_dir_lst.append(key)

    data_dir_lst = np.unique(data_dir_lst)

    cluster_cfg_lst = cluster_cfg_lst + data_dir_lst.tolist()

    for ix in range(len(cfgs)):
        for key in cluster_cfg_lst:
            try:
                del cfgs[ix][key]
            except:
                pass
        


    selected_id = list()
    cluster_lst = list()

    for ix in range(len(cfgs)):
        if ix not in selected_id:
            cluster_lst.append([ix])
            selected_id.append(ix)
            for iy in range(ix+1,len(cfgs)):
                if cfgs[ix]==cfgs[iy]:
                    cluster_lst[-1].append(iy)
                    selected_id.append(iy)

    
    return cluster_lst


def compute_norm_integral_metric(ratio, score, benchmark_score):
    f = interp1d(ratio, score)
    metric = quad(f, ratio[0], ratio[-1])[0] \
        /((ratio[-1]-ratio[0])*benchmark_score)
    return metric

def select_results(result_lst, selection_dict):
    filtered_result_lst = list()
    for res in result_lst:
        cfg = res["cfg"]
        match = True
        for key, value in selection_dict.items():
            if cfg[key] != value:
                match = False
                break
        
        if match:
            filtered_result_lst.append(res)

    return filtered_result_lst


def t_test(random, dal, benchmark_score):
    random_metric = list()
    dal_metric = list()

    for i in range(len(dal["ratio"])):
        dal_metric.append(compute_norm_integral_metric(dal["ratio"][i], dal["score"][i], benchmark_score))
    
    for i in range(len(random["ratio"])):
        random_metric.append(compute_norm_integral_metric(random["ratio"][i], random["score"][i], benchmark_score))

    return ttest_ind(random_metric, dal_metric, alternative='less').pvalue, np.mean(random_metric), np.mean(dal_metric) 

def cluster2res_lst(cluster_lst_seed_labeled_ratio, runs, score_name):
    result_lst = list()
    for cluster in cluster_lst_seed_labeled_ratio:
        df = pd.DataFrame(columns=["ratio", "seed", "score", "time"])
        for run_index in cluster:
            run_i = runs[run_index]
            try:
                df = df.append({
                    "ratio": run_i.config["data.datamodule.arguments.initial_labeled_ratio"], 
                    "seed": run_i.config["experiment.seed"] , 
                    "score": run_i.summary[score_name], 
                    "time": run_i.summary["_runtime"]},
                    ignore_index=True)
            except:
                pass


        seeds = np.unique(df["seed"].to_numpy())

        df = df.sort_values(by=["ratio"])

        result_dict = dict()
        result_dict["ratio"] = list()
        result_dict["score"] = list()
        result_dict["time"] = list()

        for seed in seeds:
            result_dict["ratio"].append(df[df["seed"]==seed]["ratio"].to_numpy())
            result_dict["score"].append(df[df["seed"]==seed]["score"].to_numpy())
            result_dict["time"].append(np.sum(df[df["seed"]==seed]["time"].to_numpy()))

        
        result_dict["time"] = np.mean(result_dict["time"])
        result_dict["cfg"] = run_i.config
        if len(result_dict["ratio"])>0:
            result_lst.append(result_dict)

    return result_lst