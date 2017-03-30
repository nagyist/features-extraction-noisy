import cfg
from featext import features_extraction_helper
import common


NETS = cfg.NETS
#NETS = ["inception_v3", "vgg16", "vgg19"]



BATCH = 64


def main():
    cfg.init()

    print NETS
    for feat_net in NETS:
        crop, size = cfg.crop_size(net=feat_net)

        # dataset_path = common.dataset_path(cfg.DATASET, crop, size)
        # train_path = dataset_path + ".train.h5"
        # train_path_ds = dataset_path + ".train.ds.h5"
        # valid_path = dataset_path + ".valid.h5"

        train_name = cfg.DATASET + '_train'
        train_ds_name = cfg.DATASET + '_train_ds'
        valid_name = cfg.DATASET + '_valid'
        #test_name = cfg.DATASET + '_test' # + '_verrocchio77'

        DATASETS = [train_name, train_ds_name, valid_name]
        #DATASETS = [test_name]
        DATASETS = ['outliers']

        print DATASETS
        for dataset in DATASETS:
            in_dataset_path = common.dataset_path(dataset, crop, size)
            out_dataset_path = common.feat_path(dataset, feat_net)
            print("")
            print("Features net:     {}".format(feat_net))
            print("Input dataset:    {}".format(in_dataset_path))
            print("Out feat dataset: {}".format(out_dataset_path))

            features_extraction_helper(feat_net, in_dataset_path, out_dataset_path, alternative_out_layer=None,
                                       verbose=True, batch_size=32)
            print("Done.")
        print("")

    print("")
    print("All done.")



if __name__ == "__main__":
    main()