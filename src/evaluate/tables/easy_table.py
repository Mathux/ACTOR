import os
import glob
import math
import numpy as np

from ..tools import load_metrics


def get_gtname(mname):
    return mname + "_gt"


def get_genname(mname):
    return mname + "_gen"


def get_reconsname(mname):
    return mname + "_recons"


def valformat(val, power=3):
    p = float(pow(10, power))
    # "{:<04}".format(np.round(p*val).astype(int)/p)
    return str(np.round(p*val).astype(int)/p).ljust(4, "0")


def format_values(values, key, latex=True):
    mean = np.mean(values)

    if "accuracy" in key:
        mean = 100*mean
        values = 100*values
        smean = valformat(mean, 1)
    else:
        smean = valformat(mean, 2)

    interval = valformat(1.96 * np.var(values), 2)  # [1:]

    if latex:
        string = rf"${smean}^{{\pm{interval}}}$"
    else:
        string = rf"{smean} +/- {interval}"
    return string


def print_results(folder, evaluation):
    evalpath = os.path.join(folder, evaluation)
    metrics = load_metrics(evalpath)

    a2m = metrics["feats"]

    if "fid_gen_test" in a2m:
        keys = ["fid_{}_train", "fid_{}_test", "accuracy_{}_train", "diversity_{}_train", "multimodality_{}_train"]
    else:
        keys = ["fid_{}", "accuracy_{}", "diversity_{}", "multimodality_{}"]

    lines = ["gen", "recons"]
    # print the GT, only if it is computed with respect to "another" GT
    if "fid_gt2" in a2m:
        a2m["fid_gt"] = a2m["fid_gt2"]
        lines = ["gt"] + lines

    rows = []
    rows_latex = []

    for model in lines:
        row = ["{:6}".format(model)]
        row_latex = ["{:6}".format(model)]
        try:
            for key in keys:
                ckey = key.format(model)
                values = np.array([float(x) for x in a2m[ckey]])
                string_latex = format_values(values, key, latex=True)
                string = format_values(values, key, latex=False)
                row.append(string)
                row_latex.append(string_latex)
            rows.append(" | ".join(row))
            rows_latex.append(" & ".join(row_latex) + r"\\")
        except KeyError:
            continue

    table = "\n".join(rows)
    table_latex = "\n".join(rows_latex)
    print("Results")
    print(table)
    print()
    print("Latex table")
    print(table_latex)


if __name__ == "__main__":
    import argparse

    def parse_opts():
        parser = argparse.ArgumentParser()
        parser.add_argument("evalpath", help="name of the evaluation")
        return parser.parse_args()

    opt = parse_opts()
    evalpath = opt.evalpath

    folder, evaluation = os.path.split(evalpath)
    print_results(folder, evaluation)
