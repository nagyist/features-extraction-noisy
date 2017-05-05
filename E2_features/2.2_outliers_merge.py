from config import cfg, common
from imdataset import ImageDataset



BATCH = 64

def main():
    cfg.init()
    exp_outlier_merge(['_train', '_train_ds', '_valid', '_test'], 'outliers')





def exp_outlier_merge(sub_dataset=['_train', '_valid', '_test'], outlier_dataset_name='outliers'):

    datasets = [cfg.dataset + sd for sd in sub_dataset]

    for feat_net in cfg.nets:

        outliers_dataset = common.feat_dataset(outlier_dataset_name, feat_net)

        for dataset in datasets:
            feat_dataset = common.feat_dataset(dataset, feat_net)
            out_dataset_path = common.feat_path(dataset + "_outl", feat_net)
            print("")
            print("Features net:     {}".format(feat_net))
            print("Input dataset:    {}".format(common.feat_fname(dataset, feat_net)))
            print("Merging with:     {}".format(common.feat_fname(outlier_dataset_name, feat_net)))
            print("Out feat dataset: {}".format(common.feat_fname(dataset + "_outl", feat_net)))

            out = ImageDataset.merge_datasets(feat_dataset, outliers_dataset, label_mode='new')
            out.save_hdf5(out_dataset_path)
            print("Done")
        print("")

    print("")
    print("All done.")

if __name__ == "__main__":
    main()