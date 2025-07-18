import pyagrum.lib.image as gumimage
import pandas as pd
import pyagrum.lib.notebook as gnb
import pyagrum as gum
import pyagrum.lib.bn_vs_bn as bnvsbn
from pyagrum.lib.bn2roc import showROC
import os
basedir=""


def data_change_columns(data):
    data = data.astype({"Age": str})
    data.loc[data["Age"] == "35-49", "Age"] = "3549"
    data.loc[data["Age"] == "50-74", "A ge"] = "5074"
    data.loc[data["Age"] == ">75", "Age"] = "75"
    data.loc[data["Age"] == "<35", "Age"] = "35"

    data.loc[data["Size"] == "<1cm", "Size"] = "small"
    data.loc[data["Size"] == ">3cm", "Size"] = "large"
    data.loc[data["Size"] == "1-3cm", "Size"] = "medium"

    data.loc[data["Margin"] == "Well-defined", "Margin"] = "WellDefined"
    data.loc[data["Margin"] == "Ill-defined", "Margin"] = "IllDefined"
    return data

def get_data(bc_binary):
    data = pd.read_csv("original/data/bc.csv")
    null_mask = data.isnull().any(axis=1)
    null_rows = data[null_mask]
    data = data.drop(null_rows.index)

    if bc_binary:
        data.loc[data["BC"] == "Invasive", "BC"] = "Yes"
        data.loc[data["BC"] == "Insitu", "BC"] = "Yes"
        print(data.BC.unique())
    else:
        data = data

    data.reset_index(inplace=True)

    data = data.astype({"Age": str})
    data.loc[data["Age"] == "35-49", "Age"] = "3549"
    data.loc[data["Age"] == "50-74", "Age"] = "5074"
    data.loc[data["Age"] == ">75", "Age"] = "75"
    data.loc[data["Age"] == "<35", "Age"] = "35"

    data.loc[data["Size"] == "<1cm", "Size"] = "small"
    data.loc[data["Size"] == ">3cm", "Size"] = "large"
    data.loc[data["Size"] == "1-3cm", "Size"] = "medium"

    data.loc[data["Margin"] == "Well-defined", "Margin"] = "WellDefined"
    data.loc[data["Margin"] == "Ill-defined", "Margin"] = "IllDefined"

    return data

def get_net(name,savename):
    bn = gum.BayesNet()
    bn = gum.loadBN("original/network/"+name)
    gumimage.export(bn, "out/"+ savename+".png")
    return bn

def export_html_string(html, name):
    os.makedirs("makedirs", exist_ok=True)
    Func = open(name+".html", "w", encoding="utf-8")
    Func.write(html)
    Func.close()

def create_dirs(dir, alg):
    if not dir == "":
        dir = dir + "/"
        os.makedirs(dir,exist_ok=True )

    network = dir + "network/" + alg + "/"
    os.makedirs(network, exist_ok=True)
    outdir = dir + "out/"+alg
    os.makedirs(outdir, exist_ok=True)
    dir += "out/" + alg
    os.makedirs(dir, exist_ok=True)
    return network, dir


def learn(data, template, smoothing, alg, dir=""):
    network, dir = create_dirs(dir, alg)
    learner = gum.BNLearner(data, template)

    if alg == "pc":
        bn2 = learner.useSmoothingPrior(smoothing).learnBN()
        bn2.saveNET(network + "learned_bn.net", allowModificationWhenSaving=False)
    if alg == "miic":
        bn2 = learner.useMIIC().addForbiddenArc("Age", "Metastasis").addForbiddenArc("SkinRetract",
                                                                                     "Metastasis").learnBN()
        bn2.saveNET(network + "learned_bn.net", allowModificationWhenSaving=False)
    if alg == "greedy":
        bn2 = learner.useGreedyHillClimbing().useNMLCorrection().useScoreBDeu().learnBN()
        bn2.saveNET(network + "learned_bn.net", allowModificationWhenSaving=False)

    bn3 = learner.learnParameters(bn2)
    inf = gnb.getInference(bn3)
    if dir == "":
        export_html_string(inf, "agr_opdracht2/out/inf")
    else:
        export_html_string(inf, dir + "/inf")
    gumimage.export(bn3, dir + "/bn-learned.png")
    gum.saveBN(bn3, network + "network.dsl")
    return bn2

def get_template(data):
    template = gum.BayesNet()

    for cname in data:
        if not cname=="index":
            labels= data[cname].unique()
            template.add(gum.LabelizedVariable(cname, cname, labels))

    gumimage.export(template, "out/null-template.png", size=30)
    return template

def difference(bn, bn1, name):
    html = gnb.getBNDiff(bn,bn1)
    export_html_string(html,name)

def hamming(bn1,bn2):
    cmp = bnvsbn.GraphicalBNComparator(bn1, bn2)
    print(cmp.hamming())

def get_gen_data(alg, number, validation=False):
    if validation:
        data = pd.read_csv("generated/data/out/" + alg + "/" + alg + ".csv")
    else:
        data = pd.read_csv("generated/data/out/" + alg + "/" + alg + "_validation.csv")

    data = data.astype({"Age": str})
    data.loc[data["Age"] == "35-49", "Age"] = "3549"
    data.loc[data["Age"] == "50-74", "Age"] = "5074"
    data.loc[data["Age"] == ">75", "Age"] = "75"
    data.loc[data["Age"] == "<35", "Age"] = "35"

    data.loc[data["Size"] == "<1cm", "Size"] = "small"
    data.loc[data["Size"] == ">3cm", "Size"] = "large"
    data.loc[data["Size"] == "1-3cm", "Size"] = "medium"

    data.loc[data["Margin"] == "Well-defined", "Margin"] = "WellDefined"
    data.loc[data["Margin"] == "Ill-defined", "Margin"] = "IllDefined"

    data.loc[data["BC"] == "Invasive", "BC"] = "Yes"
    data.loc[data["BC"] == "Insitu", "BC"] = "Yes"

    bc_data = data[data['BC'] == 'Yes']
    nobc_data = data[data['BC'] == 'No']
    bal_data = pd.concat([bc_data.sample(int(number / 2)), nobc_data.sample(int(number / 2))])

    return bal_data

def generate_sample_files(miic_learned_bn, greedy_learned_bn, gt_learned_bn):
    os.makedirs("generated/data/out/miic/", exist_ok=True)
    os.makedirs("generated/data/out/greedy/", exist_ok=True)
    os.makedirs("generated/data/out/pc/", exist_ok=True)

    gum.generateSample(miic_learned_bn, 10000, "generated/data/out/miic/miic.csv", show_progress=True, with_labels=True)
    gum.generateSample(greedy_learned_bn, 10000, "generated/data/out/greedy/greedy.csv", show_progress=True, with_labels=True)
    gum.generateSample(gt_learned_bn, 10000, "generated/data/out/pc/pc.csv", show_progress=True, with_labels=True)
    gum.generateSample(gt_learned_bn, 10000, "generated/data/out/pc/pc_validation.csv", show_progress=True, with_labels=True)

def get_samples():
    validation = get_gen_data("pc", 1000, validation=True)
    smp100 = get_gen_data("pc", 100)
    smp500 = get_gen_data("pc", 500)
    smp1000 = get_gen_data("pc", 1000)
    return validation,smp100, smp500, smp1000

def get_networks(smp100,smp500,smp1000, template):
    miic_1000_learned_bn = learn(smp1000, template, 0, "miic", "test/miic1000")
    miic_500_learned_bn = learn(smp500, template, 0, "miic", "test/miic500")
    miic_100_learned_bn = learn(smp100, template, 0, "miic", "test/miic100")
    greedy_1000_learned_bn = learn(smp1000, template, 0, "greedy", "test/greedy1000")
    greedy_500_learned_bn = learn(smp500, template, 0, "greedy", "test/greedy500")
    greedy_100_learned_bn = learn(smp100, template, 0, "greedy", "test/greedy100")
    return miic_100_learned_bn, miic_500_learned_bn, miic_1000_learned_bn, greedy_100_learned_bn, greedy_500_learned_bn, greedy_1000_learned_bn

def show_roc(network, data):
    showROC(network, data, 'BC', 'Yes', show_progress=False, show_fig=True, save_fig=False, with_labels=True, significant_digits=4)
