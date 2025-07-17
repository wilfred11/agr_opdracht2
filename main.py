import os

from functions import get_data, get_net, get_template, learn, difference, hamming, get_gen_data, \
    generate_sample_files, get_samples, get_networks, show_roc

do=1

if do==1:
    try:
        os.mkdir("out")
    except:
        pass

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

    generate_sample_files(miic_learned_bn, greedy_learned_bn, gt_learned_bn)

    validation, smp100,smp500, smp1000=get_samples()

    miic_100_learned_bn, miic_500_learned_bn, miic_1000_learned_bn, greedy_100_learned_bn, greedy_500_learned_bn, greedy_1000_learned_bn =get_networks(smp100,smp500,smp1000, template)

    show_roc(miic_500_learned_bn, validation)
    show_roc(miic_100_learned_bn, validation)
    show_roc(greedy_500_learned_bn, validation)
    show_roc(greedy_100_learned_bn, validation)






