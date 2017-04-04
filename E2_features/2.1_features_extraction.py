from config import cfg, common
from featext import features_extraction_helper

BATCH = 64


def main():
    cfg.init()
    exp_features_extraction(['_train',
                             '_train_ds',
                             '_valid',
                             '_test',
                             '_test_verrocchio77'
                             ])


def exp_features_extraction(sub_dataset=['_train', '_valid', '_test']):
    datasets = [cfg.dataset + sd for sd in sub_dataset]

    print "Executing on nets: " + str(cfg.nets)
    print "Executing on datasets: " + str(datasets)

    for feat_net in cfg.nets:
        crop, size = cfg.crop_size(net=feat_net)

        print("|- Net: " + feat_net)
        print datasets
        for ds_name in datasets:
            in_dataset_path = common.dataset_path(ds_name, crop, size)
            out_dataset_path = common.feat_path(ds_name, feat_net)
            print("|  |")
            print("|  |- Features net:     {}".format(feat_net))
            print("|  |- Input dataset:    {}".format(in_dataset_path))
            print("|  |- Out feat dataset: {}".format(out_dataset_path))

            features_extraction_helper(feat_net, in_dataset_path, out_dataset_path, alternative_out_layer=None,
                                       verbose=True, batch_size=32)
            print("|  * Done.")
        print("|")

    print("|")
    print("* All done.")


if __name__ == "__main__":
    main()