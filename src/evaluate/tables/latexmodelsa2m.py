import os
import glob
import math
import re
import numpy as np

from .tools import load_metrics


def valformat(val, power=3):
    p = float(pow(10, power))
    # "{:<04}".format(np.round(p*val).astype(int)/p)
    return str(np.round(p*val).astype(int)/p).ljust(5, "0")


def construct_table(folder):
    exppath = folder
    paths = glob.glob(f"{exppath}/**/evaluation*_all*.yaml")

    keys = ["fid", "accuracy", "diversity", "multimodality"]

    models_results = []
    for i, path in enumerate(paths):
        metrics = load_metrics(path)
        a2m = metrics["action2motion"]
        a2m["fid_gt"] = a2m["fid_gt2"]

        modelname = os.path.split(os.path.split(path)[0])[1]

        for info in ["vibe", "rot6d", "glob", "translation", "numlayers_8",
                     "numframes_60", "sampling_conseq", "samplingstep_1", "jointstype",
                     "gelu", "kl_1e-05", "cvae", "ntu13"]:
            modelname = modelname.replace(info, "")
            
        modelname = re.sub("_{1,}", " ", modelname)

        # takin GT only for the first one
        if i == 0:
            gtrow = ["Our GT"]
            for key in keys:
                ckey = f"{key}_gt"
                values = np.array([float(x) for x in a2m[ckey]])
                mean = valformat(np.mean(values))
                interval = valformat(1.96 * np.var(values))[1:]
                # string = rf"${mean:.4}^{{\pm{interval:.3}}}$"
                string = rf"${mean}$"  # ^{{\pm{interval}}}$"
                gtrow.append(string)
            gtrow = " & ".join(gtrow) + r"\\"
                
        rows = []
        for model in ["gen"]:  # ["gt", "gen", "recons"]:
            # row = ["{} {}".format(modelname, model)]
            row = [modelname]
            for key in keys:
                ckey = f"{key}_{model}"
                values = np.array([float(x) for x in a2m[ckey]])
                mean = valformat(np.mean(values))
                interval = valformat(1.96 * np.var(values))[1:]
                # string = rf"${mean:.4}^{{\pm{interval:.3}}}$"
                string = rf"${mean}$"  # ^{{\pm{interval}}}$"
                row.append(string)
            row = " & ".join(row) + r"\\"
            rows.append(row)
        models_result = "\n        ".join(rows)
        models_results.append(models_result)

    sorting = ["former rc kl", "former rcxyz kl", "former rc rcxyz kl",
               "former rc rcxyz vel kl", "former rc rcxyz velxyz kl",
               "former rc rcxyz vel velxyz kl"]

    changing = {"rc": r"$\mathcal{L}_{R}$",
                "rcxyz": r"$\mathcal{L}_{O}$",
                "vel": r"$\mathcal{L}_{\Delta R}$",
                "velxyz": r"$\mathcal{L}_{\Delta O}$"}
    
    changing_jointstype = {"smpl": "J",
                           "vertices": "V"}
    
    sorted_models = [gtrow, "        \\midrule\n"]
    for sortkey in sorting:
        for models_result in models_results:
            if sortkey in models_result:
                modelsname = models_result.split("&")[0].rstrip()
                losses = sortkey.split(" ")[1:-1]  # remove former and kl
                wlosses = []
                for loss in losses:
                    renaming = changing[loss]
                    jtype = modelsname.split(" ")[-1]
                    if jtype in changing_jointstype:
                        renaming = renaming.replace("O", changing_jointstype[jtype])
                    wlosses.append(renaming)
                
                models_result = models_result.replace(modelsname, " + ".join(wlosses))
                sorted_models.append(models_result)
                
    # MODELS = "\n        \\midrule\n".join(sorted_models)
    MODELS = "\n".join(sorted_models) + "\n"
    
    template = r"""\documentclass{{standalone}}
\usepackage{{booktabs}}
\usepackage[dvipsnames]{{xcolor}}
\begin{{document}}
    \begin{{tabular}}{{lcccc}}
        \toprule
        Architecture &  FID$\downarrow$ & Acc.$\uparrow$ & Div.$\uparrow$ & Multimod.$\uparrow$\\
        \midrule
        action2motion ground truth   & $0.031^{{\pm.004}}$  & $0.999^{{\pm.001}}$  & $7.108^{{\pm.048}}$ & $2.194^{{\pm.025}}$ \\
        action2motion lie model & $0.330^{{\pm.008}}$  & $0.949^{{\pm.001}}$  & $7.065^{{\pm.043}}$ & $2.052^{{\pm.030}}$ \\
        \midrule
        {MODELS}
        \bottomrule
    \end{{tabular}}
\end{{document}}
""".format(MODELS=MODELS)
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
    texpath = os.path.join(folder, "table.tex")

    with open(texpath, "w") as ftex:
        ftex.write(tex)
        
    print(f"Table saved at {texpath}")
