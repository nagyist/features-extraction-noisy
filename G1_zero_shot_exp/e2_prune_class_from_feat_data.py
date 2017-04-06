import sys

from config import cfg
from config import common


def main(args):
    cfg.init()


def class_prune_features_dataest(class_list_to_keep, feat_net='resnet50', dataset_name=cfg.dataset):
    dataset = common.feat_dataset(dataset_name, feat_net)

    pruned_dataset = dataset.sub_dataset_with_labels(class_list_to_keep)




if __name__ == "__main__":
    main(sys.argv)