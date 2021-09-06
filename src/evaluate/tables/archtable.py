import os
import glob
import math
import re
import numpy as np

from .tools import load_metrics


def valformat(val, power=3):
    p = float(pow(10, power))
    # "{:<04}".format(np.round(p*val).astype(int)/p)
    return str(np.round(p*val).astype(int)/p).ljust(4, "0")


def format_values(values, key):
    mean = np.mean(values)

    if key == "accuracy":
        mean = 100*mean
        values = 100*values
        smean = valformat(mean, 1)
    else:
        smean = valformat(mean, 2)

    interval = valformat(1.96 * np.var(values), 2)  # [1:]
    # string = rf"${mean:.4}^{{\pm{interval:.3}}}$"
    # string = rf"${smean}$"  # ^{{\pm{interval}}}$"
    string = rf"${smean}^{{\pm{interval}}}$"
    return string


def construct_table(folder):
    exppath = folder
    paths = glob.glob(f"{exppath}/**/evaluation*_all.yaml")

    keys = ["fid", "accuracy", "diversity", "multimodality"]
    
    model_metrics_dataset = {"ntu13": {},
                             "uestc": {}}

    epoch_dataset = {"ntu13": 1000,
                     "uestc": 500}
    
    model_naming = {"fc": "Fully connected",
                    "gru": "GRU",
                    # "transformer": "Old transformer",
                    "gtransformer": "Transformer"}

    ablation_naming = {"average_encoder": r"No $\mu_{a}^{token},\Sigma_{a}^{token}$",
                       "time_encoding": r"No Decoder-PE",
                       "zandtime": r"No $b_{a}^{token}$"}
    
    for i, path in enumerate(paths):
        epoch = int(path.split("evaluation_metrics_")[1].split(".")[0].split("_")[0])
        
        modelinfo = os.path.split(os.path.split(path)[0])[1]
        modelname = modelinfo.split("_")[1]
        dataset = modelinfo.split("_kl_")[1].split("_")[0]

        # Take the right epoch
        if epoch_dataset[dataset] != epoch:
            continue

        # Ablation study
        if "abl" in modelinfo:
            ablation = modelinfo.split("_abl_")[1].split("_sampling")[0]

            if ablation not in ablation_naming:
                continue

            name = ablation_naming[ablation]
        else:
            if modelname not in model_naming:
                continue
            name = model_naming[modelname]
            
        metrics = load_metrics(path)

        model_metrics = model_metrics_dataset[dataset]
        if dataset == "ntu13":
            a2m = metrics["action2motion"]

            if "GT" not in model_metrics:
                a2m["fid_gt"] = a2m["fid_gt2"]
                
                row = []
                for key in keys:
                    ckey = f"{key}_gt"
                    values = np.array([float(x) for x in a2m[ckey]])
                    string = format_values(values, key)
                    row.append(string)
                model_metrics["GT"] = row
                
            row = []
            for key in keys:
                ckey = f"{key}_gen"
                values = np.array([float(x) for x in a2m[ckey]])
                string = format_values(values, key)
                row.append(string)

            model_metrics[name] = row
        elif dataset == "uestc":
            stgcn = metrics["stgcn"]

            if "GT" not in model_metrics:
                for sets in ["train", "test"]:
                    stgcn[f"fid_gt_{sets}"] = stgcn[f"fid_gt2_{sets}"]
                stgcnkeys = ["fid_gt_train", "fid_gt_test", "accuracy_gt_train", "diversity_gt_train", "multimodality_gt_train"]
                row = []
                for ckey in stgcnkeys:
                    values = np.array([float(x) for x in stgcn[ckey]])
                    string = format_values(values, ckey.split("_")[0])
                    row.append(string)
                model_metrics["GT"] = row

            stgcnkeys = ["fid_gen_train", "fid_gen_test", "accuracy_gen_train", "diversity_gen_train", "multimodality_gen_train"]
            row = []
            for ckey in stgcnkeys:
                values = np.array([float(x) for x in stgcn[ckey]])
                string = format_values(values, ckey.split("_")[0])
                row.append(string)

            model_metrics[name] = row

    archmodels = list(model_naming.values())
    ablationmodels = list(ablation_naming.values())
    
    gtvalues = ["GT"]
    for dataset in ["uestc", "ntu13"]:
        model_metrics = model_metrics_dataset[dataset]
        gtvalues.extend(model_metrics["GT"])
    gtrow = " & ".join(gtvalues) + r"\\"
    
    groupedrows = []
    for lst in [archmodels, ablationmodels]:
        rows = []
        for model in lst:
            if model == "GT":
                continue
            values = [model]
            for dataset in ["uestc", "ntu13"]:
                model_metrics = model_metrics_dataset[dataset]
                if model in model_metrics:
                    values.extend(model_metrics[model])
                else:
                    dummy = ["" for _ in range(len(model_metrics["GT"]))]
                    values.extend(dummy)
            row = " & ".join(values) + r"\\"
            rows.append(row)
        groupedrows.append("\n".join(rows) + "\n")
        
    template = r"""\documentclass{{standalone}}
\usepackage{{booktabs}}
\usepackage[dvipsnames]{{xcolor}}
\begin{{document}}
    \begin{{tabular}}{{lccccc|cccc}}
        \toprule
        Architecture &  FID$_{{tr}}$$\downarrow$ & Acc.$\uparrow$ & Div.$\uparrow$ & Multimod.$\uparrow$ & FID$_{{tr}}$$\downarrow$ & FID$_{{test}}$$\downarrow$ & Acc.$\uparrow$ & Div.$\uparrow$ & Multimod.$\uparrow$\\
        \midrule
        {gtrow}
        \midrule
        {archrow}
        \midrule
        {ablationrow}
        \bottomrule
    \end{{tabular}}
\end{{document}}
""".format(gtrow=gtrow, archrow=groupedrows[0], ablationrow=groupedrows[1])
    return template


if __name__ == "__main__":
    import argparse

    def parse_opts():
        parser = argparse.ArgumentParser()
        parser.add_argument("exppath", help="name of the exp")
        return parser.parse_args()

    opt = parse_opts()
    exppath = opt.exppath

    folder = exppath
    
    tex = construct_table(folder)
    texpath = os.path.join(folder, "table_arch.tex")

    with open(texpath, "w") as ftex:
        ftex.write(tex)
        
    print(f"Table saved at {texpath}")
