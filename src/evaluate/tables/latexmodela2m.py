import os
import glob
import math
import numpy as np

from .tools import load_metrics


def get_gtname(mname):
    return mname + "_gt"


def get_genname(mname):
    return mname + "_gen"


def get_reconsname(mname):
    return mname + "_recons"


def construct_table(folder, evaluation):
    evalpath = os.path.join(folder, evaluation)
    metrics = load_metrics(evalpath)

    a2m = metrics["action2motion"]
    keys = ["fid", "accuracy", "diversity", "multimodality"]

    a2m["fid_gt"] = a2m["fid_gt2"]
    modelname = os.path.split(folder)[1]

    modelname = modelname.replace("_ntu13_vibe_rot6d_glob_translation_numlayers_8_numframes_60_sampling_conseq_samplingstep_1_kl_1e-05_gelu", "")
    modelname = modelname.replace("_", " ")

    def valformat(val, power=3):
        p = float(pow(10, power))
        # "{:<04}".format(np.round(p*val).astype(int)/p)
        return str(np.round(p*val).astype(int)/p).ljust(5, "0")
    
    values = []
    rows = []
    for model in ["gt", "gen", "recons"]:
        row = ["{} {}".format(modelname, model)]
        for key in keys:
            ckey = f"{key}_{model}"
            values = np.array([float(x) for x in a2m[ckey]])
            mean = valformat(np.mean(values))
            interval = valformat(1.96 * np.var(values))[1:]
            # string = rf"${mean:.4}^{{\pm{interval:.3}}}$"
            string = rf"${mean}^{{\pm{interval}}}$"
            row.append(string)
        row = " & ".join(row) + r"\\"
        rows.append(row)
        
    MODELS = "\n        ".join(rows)
    
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
        parser.add_argument("evalpath", help="name of the evaluation")
        return parser.parse_args()

    opt = parse_opts()
    evalpath = opt.evalpath
    
    folder, evaluation = os.path.split(evalpath)
    tex = construct_table(folder, evaluation)
    texpath = os.path.join(folder, os.path.splitext(evaluation)[0] + ".tex")

    with open(texpath, "w") as ftex:
        ftex.write(tex)
        
    print(f"Table saved at {texpath}")
