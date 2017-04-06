import sys

from E3_shallow.Shallow import ShallowNetBuilder, ShallowTester, ShallowLoader, ShallowNet, NetScore
from config import cfg, common, feat_dataset_n_classes

import numpy as np
# Use one of the shallow classifier trained in E3_shallow to test the training set class per class and get the
# best ~500 classes (top-1-score)

# From the big test results, seems that the best classifier OVER THE TEST SET (52 classes without noise) is the network
# H8K with 15 epochs, obviously the best H8K over the validation set is the model H8K with weights 'best'.

# So we get the H8K@15, or H8K_ft@10 + lf-0.01@0 (the best model with labelflip OVER THE TEST SET) and we make a
# prediction over the training set, order the classes by avarage top-1 score and get the first 500 classes to make a new
# dataset, let's say 'dbp3120_pruned_H8K@15' ...



# ---- NEW -----
# Instead of use a testset like I did.. for pruning, the best aproach is to use a shallow network, as shallow as possible!
# This because we simply want to understand if from a class we can learn something that make sense.
# If a lot of outliers images are present in a class, probably a shallow classifier can't fit the data, while a more deep
# classifier can. Our target is not to fit perfectly the data, because we we would fit also the noise!
# We want to try to fit the data with a shallow classifier that can learn stuff only if the data are clean, so that
# we can prune the classes that the shallow classifier couldn't fit!
#
# In one paper is described a procedure to overcome the noise for a simila problem.
# The idea is that:
#  - Dividing a clean dataset in training-set and validation-set, and then appling a labelflip nose randomly over the
#    only training set, will cause the learning to become impossible: fitting the training set, the validation set will
#    diverge! The point is: every training-set (noisy or not) can be fitted with a network (obviously).
#
#  - Starting from this idea, the most representative classifier over a noisy dataset, is the one that minimize the loss
#    difference from training and validation set, not the one that better fit the validation or the training.
#    Of course if we had a clean test/validation, the best classifier is the one that fit the test/validation (in my case
#    I made a testset by myself with clean data, but it's only for 52 classes so is not representative for the entire
#    dataset).
#
from imdataset import ImageDataset

LOSS = 'categorical_crossentropy'
METRIC = ['accuracy']
BATCH = 32


def main(args):
    cfg.init()
    prune_feat_dataset_with_shallow_classifier('resnet50')



def prune_feat_dataset_with_shallow_classifier(feat_net='resnet50', double_seeds=True, n_top_classes=500, labelflip=False):

    dataset_name = cfg.dataset
    trainset_name = cfg.dataset + '_train' + ('_ds' if double_seeds else '')
    #validset_name = cfg.dataset + '_valid'

    # testset_name = trainset_name
    # testset_name = cfg.dataset + '_test'
    testset_name = cfg.dataset
    #testset_name = cfg.dataset + '_valid'

    print("Shallow Test")
    print("Features from CNN: " + feat_net)
    print("Trained on: " + trainset_name)
    print("Testing on: " + testset_name)

    in_shape = cfg.feat_shape_dict[feat_net]
    out_shape = feat_dataset_n_classes(testset_name, feat_net)


    SNB = ShallowNetBuilder(in_shape, out_shape)
    SL = ShallowLoader(trainset_name, feat_net)
    ST =  ShallowTester(feat_net, trainset_name, testset_name, csv_class_stats=False, csv_global_stats=False)


    # Nets to test
    #shallow_nets = [SNB.H8K]
    shallow_nets = [SNB.A]

    # Weights to load on nets to test
    #shallow_weights_to_loads = [ '15'] # 'best' # best = best on validation
    shallow_weights_to_loads = ['best']  # 'best' # best = best on validation

    #  Weights to load on labelflip-finetuned nets (finetuned loading the weights  in shallow_weights_to_loads list)
    shallow_ft_lf_weights_to_load = ['00'] # 'best' # best = best on validation


    dataset_to_prune = common.feat_dataset(dataset_name, feat_net)

    for sn in shallow_nets:
        for sh_i in shallow_weights_to_loads:
            if labelflip:
                # Test some of the finetuned model that use LabelFlip noise label:
                extr_n = '_ft@' + str(sh_i)
                for lf_i in shallow_ft_lf_weights_to_load:
                    shallow_net = sn(extr_n, lf_decay=0.01).init(lf=False).load(SL, lf_i)
                    keep, prune = test_for_top_classes(shallow_net, ST,  out_on_csv="class_pruning.csv")
                    pruned = dataset_to_prune.sub_dataset_with_labels(keep)
                    pruned.save_hdf5(common.feat_path(pruned_feat_dataset_path(dataset_name, feat_net, shallow_net)))
            else:
                # Test without LabelFlip Finetune:
                shallow_net = sn().init(lf=False).load(SL, sh_i)
                keep, prune = test_for_top_classes(shallow_net, ST, out_on_csv="class_pruning.csv")
                pruned = dataset_to_prune.sub_dataset_with_labels(keep)
                pruned.save_hdf5(common.feat_path(pruned_feat_dataset_path(dataset_name, feat_net, shallow_net)))


# def test_train_valid_loss(feat_net, training_set_name, validation_set_name, shallow_network):
#     in_shape = cfg.feat_shape_dict[feat_net]
#     out_shape = feat_dataset_n_classes(training_set_name, feat_net)
#
#     SNB = ShallowNetBuilder(in_shape, out_shape)
#     SL = ShallowLoader(training_set_name, feat_net)
#     ST_tr =  ShallowTester(feat_net, training_set_name, training_set_name, csv_class_stats=False, csv_global_stats=False)
#     ST_val =  ShallowTester(feat_net, training_set_name, validation_set_name, csv_class_stats=False, csv_global_stats=False)
#


def pruned_feat_dataset(dataset_name, feat_net, shallow_net):
    return ImageDataset().load_hdf5( pruned_feat_dataset_path(dataset_name, feat_net, shallow_net))

def pruned_feat_dataset_path(dataset_name, feat_net, shallow_net):
    return common.feat_path(dataset_name, feat_net, ext=False) + '_pruned-'+shallow_net.name+'@'+shallow_net.weights_loaded_index+ '.h5'


def test_for_top_classes(shallow_net, shallow_tester, nb_selected_classes=500, ordering_score='top1', out_on_csv=None, verbose=False):
    '''

    :param shallow_net:
    :param shallow_tester:
    :param nb_selected_classes:
    :param score: top1, top5, precision, recall, f1
    :param out_on_csv:
    :param verbose:
    :return:
    '''
    # type: (ShallowNet, ShallowTester) -> (np.ndarray, NetScore)
    print("")
    print("")
    print("Testing with shallow model: " + shallow_net.name)
    print("With weights:               " + shallow_net.weights_loaded_index)
    net_score = shallow_net.test(shallow_tester)
    top_classes = np.argsort(net_score.perclass_top1)[::-1]
    print("")

    score_dict = {
        'top1'      : net_score.perclass_top1,
        'top5'      : net_score.perclass_top5,
        'precision' : net_score.perclass_precision,
        'recall'    : net_score.perclass_recall,
        'f1'        : net_score.perclass_f1,
    }
    score = score_dict[ordering_score]

    print("Class\t" + ordering_score + " score")
    top_classes_str = ""

    keep_class_list = []
    prune_class_list = []
    for i in range(0, nb_selected_classes):
        cls = top_classes[i]
        keep_class_list.append(cls)

        ts = score[cls]
        print(str(cls) + "  \t" + str(ts) )

    for i in range(nb_selected_classes, len(top_classes)):
        prune_class_list.append(nb_selected_classes)

    if out_on_csv is not None:
        of = file(out_on_csv, 'w')
        of.write('Ordering by:\t' + ordering_score)

        of.write('\n\n\nBest classes (to keep)\tordering by:\t' + ordering_score)
        of.write('\nClass Index\tTop-1\tTop-5\tPrecision\tRecall\tF1-Score')
        for cls in keep_class_list:
            of.write('\n{}\t{}\t{}\t{}\t{}\t{}'.format(cls, net_score.perclass_top1[cls], net_score.perclass_top5[cls],
                                                       net_score.perclass_precision[cls], net_score.perclass_recall[cls],
                                                       net_score.perclass_f1[cls]) )

        of.write('\n\n\nWorst classes (to prune)\tordering by:\t' + ordering_score)
        of.write('\nClass Index\tTop-1\tTop-5\tPrecision\tRecall\tF1-Score')
        for cls in keep_class_list:
            of.write('\n{}\t{}\t{}\t{}\t{}\t{}'.format(cls, net_score.perclass_top1[cls], net_score.perclass_top5[cls],
                                                       net_score.perclass_precision[cls],
                                                       net_score.perclass_recall[cls],
                                                       net_score.perclass_f1[cls]))
    print top_classes_str
    return keep_class_list, prune_class_list


if __name__ == "__main__":
    main(sys.argv)






#
# parser = ArgumentParser()
# parser.add_argument("-n", "-net", action='store', dest='net_list', required=True, type=str, nargs='+',
#                     choices=["vgg16", "vgg19", "resnet50", "inception_v3"],
#                     help="Set the net(s) to use for the training experiment.")
# # parser.add_argument("-train", action='store_true', dest='train', default=False,
# #                     help="Train experiment.")
# # parser.add_argument("-test", action='store_true', dest='test', default=False,
# #                     help="Test experiment.")
# # parser.add_argument("-class-test", action='store_true', dest='class_test', default=False,
# #                     help="Test experiment.")
# parser.add_argument("-l", "--local", action='store_true', dest='use_local', default=False,
#                     help="Test experiment.")
# parser.add_argument("-s", "--scratch", action='store_false', dest='use_local', default=False,
#                     help="Test experiment.")
# args = parser.parse_args()