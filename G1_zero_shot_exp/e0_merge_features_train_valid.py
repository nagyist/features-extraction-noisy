import sys

from config import cfg, feat_dataset_n_classes, common
from imdataset import ImageDataset

def main(args):
    cfg.init()
    merge_features_train_valid('resnet50')



def merge_features_train_valid(feat_net='resnet50'):
    dataset_name = cfg.dataset
    trainset_name = dataset_name + '_train'
    validset_name = dataset_name + '_valid'

    print("Merging training and validation data-features (feature net: " + feat_net + ")")

    train_feat_set = common.feat_dataset(trainset_name, feat_net)
    valid_feat_set = common.feat_dataset(validset_name, feat_net)

    merged_feat_set = ImageDataset.merge_datasets(train_feat_set, valid_feat_set)

    merged_dataset_path = common.feat_path(dataset_name, feat_net)
    merged_feat_set.save_hdf5(merged_dataset_path)



if __name__ == "__main__":
    main(sys.argv)

