import pyagrum as gum
import pyagrum.lib.image as gumimage
import pandas as pd
import pyagrum.lib.notebook as gnb
import pyagrum as gum
import pyagrum.lib.bn_vs_bn as bnvsbn
from IPython.display import display, HTML
import os
basedir="opdracht2"


def get_data(bc_binary):
    data = pd.read_csv("data/bc.csv")
    print(data.head())
    print(data.columns)

    null_mask = data.isnull().any(axis=1)
    print(null_mask)
    null_rows = data[null_mask]
    data = data.drop(null_rows.index)

    if bc_binary:
        data.loc[data["BC"] == "Invasive", "BC"] = "Yes"
        data.loc[data["BC"] == "Insitu", "BC"] = "Yes"
        print(data.BC.unique())
    else:
        data = data

    data.reset_index(inplace=True)
    print(data.isna().any(axis=None))
    print(data.info())

    for col in data:
        print(col)
        print(data[col].unique())

    data.loc[data["Age"] == "35-49", "Age"] = "3549"
    data.loc[data["Age"] == "50-74", "Age"] = "5074"
    data.loc[data["Age"] == ">75", "Age"] = "75"
    data.loc[data["Age"] == "<35", "Age"] = "35"

    data.loc[data["Size"] == "<1cm", "Size"] = "small"
    data.loc[data["Size"] == ">3cm", "Size"] = "large"
    data.loc[data["Size"] == "1-3cm", "Size"] = "medium"

    data.loc[data["Margin"] == "Well-defined", "Margin"] = "WellDefined"
    data.loc[data["Margin"] == "Ill-defined", "Margin"] = "IllDefined"

    print(data.sample(20))

    return data

def get_net(name,savename):
    bn = gum.BayesNet()
    bn = gum.loadBN("network/"+name)
    gumimage.export(bn, "out/"+ savename+".png")
    return bn

def export_html_string(html, name):
    print("export html")
    print(name)
    Func = open(name+".html", "w", encoding="utf-8")
    Func.write(html)
    Func.close()


def create_dirs(dir, alg):
    if not dir == "":
        dir = dir + "/"
        try:
            os.mkdir(dir)
        except:
            pass

    try:
        if not dir == "":
            os.mkdir(basedir + "/" + dir)
            print("dir created")
    except:
        pass
    print('ok1')
    network = dir + "network/"
    try:
        os.mkdir(network)
    except:
        pass
    print('ok2')
    network += alg + "/"
    try:
        os.mkdir(network)
    except:
        pass
    print('ok3')

    ndir = dir + "out/"

    try:
        os.mkdir(ndir)
    except:
        pass

    dir += "out/" + alg

    try:
        os.mkdir(dir)
    except:
        pass
    print('ok5')
    print(network)
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

    print("is constraintbased: " + str(learner.isConstraintBased()))
    bn3 = learner.learnParameters(bn2)
    inf = gnb.getInference(bn3)
    if dir == "":
        export_html_string(inf, "opdracht2/out/inf")
    else:
        export_html_string(inf, dir + "/inf")
    gumimage.export(bn3, dir + "/bn-learned.png")
    gum.saveBN(bn3, network + "network.dsl")
    return bn2

def get_template(data):
    template = gum.BayesNet()

    for cname in data:
        print(cname)
        if not cname=="index":
            nlabels = data[cname].nunique()
            labels= data[cname].unique()
            print(labels)
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
        data = pd.read_csv("data/out/" + alg + "/" + alg + ".csv")
    else:
        data = pd.read_csv("data/out/" + alg + "/" + alg + "_validation.csv")

    print(data.head())
    print(data.columns)
    print(data.info())

    for col in data:
        print(col)
        print(data[col].unique())

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