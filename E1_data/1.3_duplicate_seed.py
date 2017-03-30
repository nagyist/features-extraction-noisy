from cfg import cfg
import common
from common import dataset_fname
from imdataset import ImageDataset



# PARAMS
if cfg.USE_TOY_DATASET:
    FNAME_START_WITH= "0001"
else:
    FNAME_START_WITH = "seed"


def main():
    cfg.init()
    exp_duplicate_seed()


def exp_duplicate_seed():
    dataset = cfg.DATASET
    for crop_size, crop_size_stamp in zip(cfg.ALL_CROP_SIZE, cfg.ALL_CROP_SIZE_STAMP):
        crop = crop_size['crop']
        size = crop_size['size']

        dataset_path = common.dataset_path(dataset, crop, size)
        train_path = common.dataset_path(dataset + '_train', crop, size)
        train_path_ds = common.dataset_path(dataset + '_train_ds', crop, size)

        print("")
        print("")
        print("Duplicate seed on:")
        print("Training Set: " + train_path)
        print("")
        print("Out Training Set: " + train_path_ds)

        trainingset = ImageDataset()
        print("loading hdf5 training set: {}".format(train_path))
        trainingset.load_hdf5(train_path)
        print("hdf5 file loaded.")

        print("Getting sub dataset from filter (seed files)...")
        seeds_dataset = trainingset.sub_dataset_from_filename(filename_start_with=FNAME_START_WITH)
        print("Merging seed-only sub dataset with original dataset")
        trainingset.merge_with_dataset(seeds_dataset)
        print("Saving merged dataset in: " + train_path_ds)
        trainingset.save_hdf5(train_path_ds)
        print("Done.")

    print("")
    print("All done.")

#
# def duplicate_dataset_imgs_helper(dataset_path, out_dataset_path, fname_start, fname_end):
#     print("")
#     print("")
#     print("Duplicate seed on:")
#     print("Dataset: " + dataset_path)
#     print("")
#     print("Out Training Set: " + out_dataset_path)
#
#     training_path = common.dataset_path(dataset_path, net)
#     out_training_path = common.dataset_path(out_dataset_path, net)
#
#     trainingset = ImageDataset()
#     print("loading hdf5 training set: {}".format(training_path))
#     trainingset.load_hdf5(training_path)
#     print("hdf5 file loaded.")
#
#     print("Getting sub dataset from filename filters...")
#     seeds_dataset = trainingset.sub_dataset_from_filename(filename_start_with=fname_start, filename_end_with=fname_end)
#     print("Merging seed-only sub dataset with original dataset")
#     trainingset.merge_with_dataset(seeds_dataset)
#     print("Saving merged dataset in: " + out_training_path)
#     trainingset.save_hdf5(out_training_path)
#     print("Done.")


if __name__ == "__main__":
    main()

