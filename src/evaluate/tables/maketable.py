import os
import glob
import math

from .tools import load_metrics

METRICS = {"joints": ["acceleration", "rc", "diversity", "multimodality"],
           "action2motion": ["accuracy", "fid", "diversity", "multimodality"]}

UP = r"$\uparrow$"
DOWN = r"$\downarrow$"
RIGHT = r"$\rightarrow$"

ARROWS = {"accuracy": UP,
          "acceleration": RIGHT,
          "rc": DOWN,
          "fid": DOWN,
          "diversity": RIGHT,
          "multimodality": RIGHT}

POSE_ORDER = ["xyz", "rotvec", "rotquat", "rotmat", "rot6d"]
for pose in POSE_ORDER:
    METRICS[pose] = METRICS["joints"]

GROUPORDER = POSE_ORDER + ["action2motion"]

GREEN = "Green"
RED = "Mahogany"


def bold(string):
    return r"\textbf{{" + string + r"}}"


def colorize_template(string, color):
    return r"\textcolor{{" + color + r"}}{{" + string + r"}}"


def colorize_bold_template(string, color):
    return bold(colorize_template(string, color))


def format_table(val, gtval, mname):
    value = float(val)
    
    try:
        exp = math.floor(math.log10(value))
    except ValueError:
        exp = 0
        value = 0
    
    if mname == "rc":
        formatter = "{:.1e}"
        if value >= 1:
            formatter = colorize_bold_template(formatter, RED)
            
    elif mname in ["diversity", "multimodality"]:
        if exp < -1:
            formatter = "{:.1e}"
        else:
            formatter = "{:.3g}"
        if gtval is not None:
            gtval = float(gtval)
            if value > 0.8*gtval:
                formatter = colorize_bold_template(formatter, GREEN)
            elif value < 0.3*gtval:
                formatter = colorize_bold_template(formatter, RED)
                    
    elif mname == "accuracy":
        formatter = "{:.1%}"
        if value > 0.65:
            formatter = colorize_bold_template(formatter, GREEN)
        elif value < 0.35:
            formatter = colorize_bold_template(formatter, RED)
        
    elif mname == "acceleration":
        formatter = "{:.1e}"
        if gtval is not None:
            gtval = float(gtval)
            diff = math.log10(value/gtval)
            # below acceleration
            if diff < 0.05:
                formatter = colorize_bold_template(formatter, GREEN)
            elif diff > 0.3:
                formatter = colorize_bold_template(formatter, RED)
                
    else:
        formatter = "{:.2f}"

    formatter = bold(formatter)
    return formatter.format(value).replace("%", r"\%")


def get_gtname(mname):
    return mname + "_gt"


def get_genname(mname):
    return mname + "_gen"


def get_reconsname(mname):
    return mname + "_recons"


def collect_tables(folder, expname, lastepoch=False, norecons=False):
    exppath = os.path.join(folder, expname)
    paths = glob.glob(f"{exppath}/**/evaluation*")

    if len(paths) == 0:
        raise ValueError("No evaluation founds.")

    pose_rep, *losses = expname.split("_")
    expname = expname.replace("_", "\\_")

    models_kl = {}
    allkls = set()
    models_epochs = {}
    for path in paths:
        metrics = load_metrics(path)
        fname = os.path.split(path)[0]
        modelname = fname.split("cvae_")[1].split("_rc")[0]
        kl_loss = float(fname.split("_kl_")[2].split("_")[0])
        epoch = os.path.split(path)[1].split("evaluation_metrics_")[1].split(".")[0]
            
        if lastepoch:
            if modelname not in models_epochs:
                models_epochs[modelname] = epoch
            else:
                if models_epochs[modelname] > epoch:
                    continue
                else:
                    models_epochs[modelname] = epoch
            modelname = rf"{modelname}"
        else:
            modelname = rf"{modelname}\_{epoch}"

        if "numlayers" in fname:
            nlayers = int(fname.split("numlayers")[1].split("_")[1])
            modelname += rf"\_nlayer\_{nlayers}"

        if "relu" in fname:
            activation = "relu"
        elif "gelu" in fname:
            activation = "gelu"
        else:
            activation = ""

        modelname += rf"\_{activation}"

        try:
            ablation = fname.split("abl_")[1].split("_sampling")[0]
            ablation = ablation.replace("_", r"\_")
            modelname += rf"\_{ablation}"
        except IndexError:
            modelname += r"\_noablation"
        
        if modelname not in models_kl:
            models_kl[modelname] = {}
        models_kl[modelname][kl_loss] = metrics
        allkls.add(kl_loss)

    lambdas_sorted = sorted(list(allkls), reverse=True)
    
    gtrowl = ["ground truth"]
    for group in GROUPORDER:
        if group in metrics:
            for mname in METRICS[group]:
                gtname = get_gtname(mname)
                if gtname in metrics[group]:
                    val = format_table(metrics[group][gtname], None, mname)
                    gtrowl.append(val)
                else:
                    gtrowl.append("")
            gtrowl.append("")
    gtrowl.pop()
    gtrow = " & ".join(gtrowl) + r"\\"

    bodylist = [gtrow]
    bodylist.append(r"\midrule")

    modelnames = sorted(list(models_kl.keys()))

    # compute first rows
    # to add a first col
    firstrow = [""]
    for group in GROUPORDER:
        if group in metrics:
            for mname in METRICS[group]:
                mname = f"{mname} {ARROWS[mname]}"
                firstrow.append(mname)
            firstrow.append("")
    firstrow.pop()
    firstrow = " & ".join(firstrow) + r"\\"
       
    for lam in lambdas_sorted:
        for modelname in modelnames:
            if lam in models_kl[modelname]:
                metrics = models_kl[modelname][lam]
                row = [f"{modelname} {lam}"]
                for group in GROUPORDER:
                    if group in metrics:
                        for mname in METRICS[group]:
                            gtname = get_gtname(mname)
                            gtval = metrics[group][gtname] if gtname in metrics[group] else None
                            genname = get_genname(mname)
                            reconsname = get_reconsname(mname)
                            if not norecons and genname in metrics[group] and reconsname in metrics[group]:
                                genval = format_table(metrics[group][genname], gtval, mname)
                                reconsval = format_table(metrics[group][reconsname], gtval, mname)
                                row.append(f"{genval}/{reconsval}")
                            elif genname in metrics[group]:
                                genval = format_table(metrics[group][genname], gtval, mname)
                                row.append(f"{genval}")
                            elif reconsname in metrics[group]:
                                reconsval = format_table(metrics[group][reconsname], gtval, mname)
                                row.append(f"{reconsval}")
                            else:
                                print(f"{mname} is not present in this evaluation")
                        row.append("")
                row.pop()
                row = " & ".join(row) + r"\\"
                bodylist.append(row)
        # bodylist.append(emptyrow)
        bodylist.append(r"\midrule")

    bodylist.append(r"\bottomrule")
    body = "\n".join(bodylist)
    ncols = len(gtrowl)
    title = f"Evaluation of {expname} experiment"
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
        parser.add_argument("exppath", help="name of the exp")
        parser.add_argument("--outpath", default="tex", help="name of the exp")
        parser.add_argument("--norecons", dest='norecons', action='store_true')
        parser.set_defaults(norecons=False)
        parser.add_argument("--lastepoch", dest='lastepoch', action='store_true')
        parser.set_defaults(lastepoch=False)
        return parser.parse_args()

    opt = parse_opts()
    exppath = opt.exppath
    norecons = opt.norecons
    lastepoch = opt.lastepoch
    
    folder, expname = os.path.split(exppath)

    template = collect_tables(folder, expname, lastepoch=lastepoch, norecons=norecons)

    # os.makedirs(opt.outpath, exist_ok=True)
    
    name = expname
    if norecons:
        name += "_norecons"
    texpath = os.path.join(exppath, name + ".tex")

    with open(texpath, "w") as ftex:
        ftex.write(template)
    print(f"Table saved at {texpath}")
