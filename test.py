import os

import net_utils
from argparse import ArgumentParser
import sys
import cfg
from imdataset import ImageDataset
from imdataset import NetScore


def test_net(model, testset, verbose=True, single_class_verbose=False):
    predictions = model.predict(x=testset.data, batch_size=32, verbose=verbose)
    ns = NetScore(testset, predictions)
    if verbose:
        if single_class_verbose:
            ns.print_perclass()
        ns.print_global()
    return ns


def write_net_score(netScore, model_name, testset_name, csv_path=None, detailed_csv=False, detailed_csv_path=None, skip_no_example_test=True):
    # type: (NetScore, basestring, basestring, basestring, bool, basestring, bool) -> None
    ns = netScore
    print("Net score for model name: " +model_name)
    if csv_path is None:
        csv_path ="global_net_score.csv"

    if os.path.isfile(csv_path):
        print('csv file exists, append!')
        csv = file(csv_path, "a")
    else:
        print('csv file not exists, create!')
        csv = file(csv_path, "w")
        csv.write("MODEL NAME\tTESTSET NAME\t\tTOP-1\tTOP-5\tPRECISION\tRECALL\tF1 SCORE\tEXAMPLES(weights)"
                                          "\t\tAVG TOP-1\tAVG TOP-5\tAVG PRECISION\tAVG RECALL\tAVG F1 SCORE")

    csv.write("\n{}\t{}\t\t{}\t{}\t{}\t{}\t{}\t{}\t\t{}\t{}\t{}\t{}\t{}".
              format(model_name, testset_name, ns.global_top1, ns.global_top5,
                     ns.global_precision, ns.global_recall, ns.global_f1, ns.examples,
                     ns.avg_top1, ns.avg_top5, ns.avg_precision, ns.avg_recall, ns.avg_f1))
    csv .close()

    if detailed_csv:
        if detailed_csv_path is None:
            detailed_csv_path = model_name + ".netscore.csv"
        model_csv = file(detailed_csv_path, "w")
        model_csv.write("\t\tMODEL NAME\tTESTSET NAME\t\tTOP-1\tTOP-5\tPRECISION\tRECALL\tF1 SCORE\tEXAMPLES")
        model_csv.write("\n\t{}\t{}\t{}\t\t{}\t{}\t{}\t{}\t{}\t{}".
                        format("Weighted AVG:", model_name, testset_name, ns.global_top1, ns.global_top5,
                               ns.global_precision, ns.global_recall, ns.global_f1, ns.examples))
        model_csv.write("\n\t{}\t{}\t{}\t\t{}\t{}\t{}\t{}\t{}\t{}".
                        format("Simple AVG:", model_name, testset_name, ns.avg_top1, ns.avg_top5,
                               ns.avg_precision, ns.avg_recall, ns.avg_f1, ns.examples))

        model_csv.write("\n\n\nPER CLASS CORE\nCLASS NAME\tCLASS\tTOP-1\tTOP-5\tPRECISION\tRECALL\tF1 SCORE\tEXAMPLES")
        l = 0
        for top1, top5, prec, rec, f1, examples in zip(ns.perclass_top1, ns.perclass_top5, ns.perclass_precision,
                                                       ns.perclass_recall, ns.perclass_f1, ns.perclass_examples):
            if skip_no_example_test and examples == 0:
                pass
            else:
                model_csv.write('\n"{}"\t{}\t{}\t{}\t{}\t{}\t{}\t{}'
                                .format(ns.testset.labelIntToStr(l), l, top1, top5, prec, rec, f1, examples))
            l += 1
        model_csv.close()
    print("")





# TODO: The only way to make a MAIN is to load the net weights and model from file h5/json. Skip for now.
#
# def main(argv):
#
#     parser = ArgumentParser()
#     parser.add_argument('-n', "-net", action='store', dest='net', required=True, choices=cfg.NETS)
#     parser.add_argument('-d', "-dataset", action='store', dest='dataset_path', required=True)
#     parser.add_argument('-o', "-out", action='store', dest='out_dataset_path', required=True)
#
#
#     parser.add_argument('-l', "-layer", action='store', dest='alternative_out_layer', required=True)
#     parser.add_argument('-b', "-batch-size", action='store', dest='batch_size', default=32, type=int)
#     parser.add_argument("-v", action='store_true', dest='verbose', default=False)
#
#     # parser.add_argument("--local", action='store_true', dest='use_local', default=True)
#     # parser.add_argument("--scratch", action='store_false', dest='use_local', default=False)
#     a = parser.parse_args()
#
#     cfg.init("FEATURES EXTRACTION", is_experiment=False , use_local=a.use_local)
#
#
#
# if __name__ == "__main__":
#     main(sys.argv)
#
#
#
#

