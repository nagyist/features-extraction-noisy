import cfg
from featext import features_extraction_helper
import common
from imdataset import ImageDataset

NETS = cfg.NETS
#NETS = ["inception_v3", "vgg16", "vgg19"]



BATCH = 64


def main():
    cfg.init()

    for feat_net in NETS:
        crop, size = cfg.crop_size(net=feat_net)

        # dataset_path = common.dataset_path(cfg.DATASET, crop, size)
        # train_path = dataset_path + ".train.h5"
        # train_path_ds = dataset_path + ".train.ds.h5"
        # valid_path = dataset_path + ".valid.h5"

        train_name = cfg.DATASET + '_train'
        train_ds_name = cfg.DATASET + '_train_ds'
        valid_name = cfg.DATASET + '_valid'
        test_name = cfg.DATASET + '_test' # + '_verrocchio77'
        DATASETS = [train_name, train_ds_name, valid_name, test_name]
        outliers_name = 'outliers'
        outliers_dataset = common.feat_dataset(outliers_name, feat_net)

        for dataset in DATASETS:
            feat_dataset = common.feat_dataset(dataset, feat_net)
            out_dataset_path = common.feat_path(dataset + "_outl", feat_net)
            print("")
            print("Features net:     {}".format(feat_net))
            print("Input dataset:    {}".format(common.feat_fname(dataset, feat_net)))
            print("Merging with:     {}".format(common.feat_fname(outliers_name, feat_net)))
            print("Out feat dataset: {}".format(common.feat_fname(dataset + "_outl", feat_net)))

            out = ImageDataset.merge_datasets(feat_dataset, outliers_dataset, label_mode='new')
            out.save_hdf5(out_dataset_path)
            print("Done")
        print("")

    print("")
    print("All done.")



if __name__ == "__main__":
    main()