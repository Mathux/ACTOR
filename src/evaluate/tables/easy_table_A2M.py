import os
import glob
import math
import numpy as np

from ..tools import load_metrics


def valformat(val, power=3):
    p = float(pow(10, power))
    # "{:<04}".format(np.round(p*val).astype(int)/p)
    return str(np.round(p*val).astype(int)/p).ljust(4, "0")


def construct_table(folder, evaluation):
    evalpath = os.path.join(folder, evaluation)
    metrics = load_metrics(evalpath)

    a2m = metrics["feats"]
    keys = ["fid", "accuracy", "diversity", "multimodality"]

    a2m["fid_gt"] = a2m["fid_gt2"]

    values = []
    rows = []
    for model in ["gt", "gen", "genden"]:
        row = ["{:6}".format(model)]
        for key in keys:
            ckey = f"{key}_{model}"
            values = np.array([float(x) for x in a2m[ckey]])
            mean = np.mean(values)
            if key == "accuracy":
                mean = 100*mean
                values = 100*values
                smean = valformat(mean, 1)
            else:
                smean = valformat(mean, 2)
                mean = np.mean(values)
            interval = valformat(1.96 * np.var(values), 2)  # [1:]
            string = rf"${smean}^{{\pm{interval}}}$"
            # string = rf"{mean:.4}"  #^{{\pm{interval:.1}}}"
            row.append(string)
        rows.append(" & ".join(row) + r"\\")

    test = "\n".join(rows)
    print(test)
    import ipdb; ipdb.set_trace()
    bodylist.append(r"\bottomrule")
    body = "\n".join(bodylist)
    ncols = 5
    title = f"Evaluation TODO name"
    template = r"""\documentclass{{standalone}}
\usepackage{{booktabs}}
\usepackage[dvipsnames]{{xcolor}}
\begin{{document}}
\begin{{tabular}}{{{ncolsl}}}
\multicolumn{{{ncols}}}{{c}}{{{title}}} \\
\multicolumn{{{ncols}}}{{c}}{{}} \\
& \multicolumn{{{nbcolsxyz}}}{{c}}{{xyz}} & & \multicolumn{{{nbcolspose}}}{{c}}{{{pose_rep}}} & & \multicolumn{{{nbcolsa2m}}}{{c}}{{action2motion}} \\
{firstrow}
\midrule
{body}
\end{{tabular}}
\end{{document}}
""".format(ncolsl="l"+"c"*(ncols-1), ncols=ncols,
           pose_rep=pose_rep, title=title, firstrow=firstrow,
           nbcolsxyz=len(METRICS["joints"]),
           nbcolspose=len(METRICS[pose_rep]),
           nbcolsa2m=len(METRICS["action2motion"]),
           body=body)
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
