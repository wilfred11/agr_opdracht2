from functions import get_data, get_net, get_template, learn, difference, hamming, get_gen_data
import pyagrum as gum
from pyagrum.lib.bn2roc import showROC

do=1

if do==1:
    data = get_data(True)
    gt_bn = get_net("bc.net", "ground-truth")
    template = get_template(data)
    miic_learned_bn = learn(data, template, 0, "miic")
    greedy_learned_bn = learn(data, template, 0, "greedy")
    gt_learned_bn = learn(data, gt_bn, 0, "pc")
    difference(miic_learned_bn, greedy_learned_bn, "diff_greedy_miic")
    difference(gt_bn, miic_learned_bn, "diff_miic_gt")
    difference(gt_bn, greedy_learned_bn, "diff_greedy_gt")
    hamming(greedy_learned_bn, miic_learned_bn)
    miic_learned_bn = learn(data, template, 0, "miic")
    greedy_learned_bn = learn(data, template, 0, "greedy")
    gt_learned_bn = learn(data, gt_bn, 0, "pc")
    gum.generateSample(miic_learned_bn, 10000, "data/out/miic/miic.csv", show_progress=True, with_labels=True)
    gum.generateSample(greedy_learned_bn, 10000, "data/out/greedy/greedy.csv", show_progress=True, with_labels=True)
    gum.generateSample(gt_learned_bn, 10000, "data/out/pc/pc.csv", show_progress=True, with_labels=True)
    gum.generateSample(gt_learned_bn, 10000, "data/out/pc/pc_validation.csv", show_progress=True, with_labels=True)
    validation = get_gen_data("pc", 1000, validation=True)
    smp100 = get_gen_data("pc", 100)
    smp500 = get_gen_data("pc", 500)
    smp1000 = get_gen_data("pc", 1000)
    miic_1000_learned_bn = learn(smp1000, template, 0, "miic", "miic1000")
    miic_500_learned_bn = learn(smp500, template, 0, "miic", "miic500")
    miic_100_learned_bn = learn(smp100, template, 0, "miic", "miic100")
    greedy_1000_learned_bn = learn(smp1000, template, 0, "greedy", "greedy1000")
    greedy_500_learned_bn = learn(smp500, template, 0, "greedy", "greedy500")
    greedy_100_learned_bn = learn(smp100, template, 0, "greedy", "greedy100")
    showROC(miic_1000_learned_bn, validation, 'BC', 'Yes', show_progress=False, show_fig=True, save_fig=False,
            with_labels=True, significant_digits=4)
    showROC(miic_500_learned_bn, validation, 'BC', 'Yes', show_progress=False, show_fig=True, save_fig=False,
            with_labels=True, significant_digits=4)
    showROC(miic_100_learned_bn, validation, 'BC', 'Yes', show_progress=False, show_fig=True, save_fig=False,
            with_labels=True, significant_digits=4)
    showROC(greedy_1000_learned_bn, validation, 'BC', 'Yes', show_progress=False, show_fig=True, save_fig=False,
            with_labels=True, significant_digits=4)
    showROC(greedy_500_learned_bn, validation, 'BC', 'Yes', show_progress=False, show_fig=True, save_fig=False,
            with_labels=True, significant_digits=4)
    showROC(greedy_100_learned_bn, validation, 'BC', 'Yes', show_progress=False, show_fig=True, save_fig=False,
            with_labels=True, significant_digits=4)


