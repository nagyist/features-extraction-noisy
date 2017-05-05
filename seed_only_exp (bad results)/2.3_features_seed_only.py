import config
from config import common
from featext import features_extraction_helper

NETS = config.NETS
#NETS = ["inception_v3", "vgg16", "vgg19"]



BATCH = 64


def main():
    config.init()

    print NETS
    for feat_net in NETS:
        crop, size = config.crop_size(net=feat_net)

        train_name = config.DATASET + "_so"
        test_name = config.DATASET + '_so_test'


        DATASETS = [train_name, test_name]
        # DATASETS = [test_name]

        print DATASETS
        for dataset in DATASETS:
            in_dataset_path = common.dataset_path(dataset, crop, size)
            out_dataset_path = common.feat_path(dataset, feat_net)
            print("")
            print("Features net:     {}".format(feat_net))
            print("Input dataset:    {}".format(in_dataset_path))
            print("Out feat dataset: {}".format(out_dataset_path))

            features_extraction_helper(feat_net, in_dataset_path, out_dataset_path, alternative_out_layer=None, verbose = True)
            print("Done.")
        print("")

    print("")
    print("All done.")



if __name__ == "__main__":
    main()