import config
from config import common
from imdataset import ImageDataset



# PARAMS
from imdataset.ImageDataset import SplitOptions

if config.USE_TOY_DATASET:
    FNAME_START_WITH = ["0001"]
    FNAME_END = []
else:
    FNAME_START_WITH = ["seed", "google"]
    FNAME_END = ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg',
                 '10.jpg', '11.jpg']


def main():
    config.init()

    for crop_size, crop_size_stamp in zip(config.ALL_CROP_SIZE, config.ALL_CROP_SIZE_STAMP):
        crop = crop_size['crop']
        size = crop_size['size']

        dataset_path = common.dataset_path(config.DATASET, crop, size)
        out_validset_path =  common.dataset_path(config.DATASET + '_sg_valid', crop, size)
        out_trainset_path =  common.dataset_path(config.DATASET + '_sg_train', crop, size)


        print("")
        print("")
        print("Seed + Google train/valid set.")
        print("Original dataset: " + dataset_path)
        print("Out s+g trainset: " + out_trainset_path)
        print("Out s+g validset: " + out_validset_path)
        print("")

        trainingset = ImageDataset()
        print("loading hdf5 dataset set: {}".format(dataset_path))
        trainingset.load_hdf5(dataset_path)
        print("hdf5 file loaded.")

        print("Getting sub dataset (seed dataset)")
        seeds_dataset = trainingset.sub_dataset_from_filename(filename_start_with="seed")
        print("Getting sub dataset (google dataset)")
        google_dataset = trainingset.sub_dataset_from_filename_multi(filename_start_with=["google"], filename_end_with=FNAME_END)
        print("Splitting google dataset in train/valid")
        google_train, google_valid = google_dataset.validation_per_class_split([SplitOptions("", 0.33)])

        print("Creating double_seeds_dataset")
        double_seeds_dataset = ImageDataset.merge_datasets(seeds_dataset, seeds_dataset)
        print("Creating train dataset (merge google_train with double_seeds_dataset)")
        train = ImageDataset.merge_datasets(google_train, double_seeds_dataset)
        print("Creating valid dataset (merge google_valid with seeds_dataset)")
        valid = ImageDataset.merge_datasets(google_valid, seeds_dataset)

        print("Saving train on h5")
        train.save_hdf5(out_trainset_path)
        print("Saving valid on h5")
        valid.save_hdf5(out_validset_path)
        print("Done.")

    print("")
    print("All done.")


def duplicate_dataset_imgs_helper(dataset_path, out_dataset_path, fname_start, fname_end):
    print("")
    print("")
    print("Duplicate seed on:")
    print("Dataset: " + dataset_path)
    print("")
    print("Out Training Set: " + out_dataset_path)

    training_path = common.dataset_path(dataset_path, net)
    out_training_path = common.dataset_path(out_dataset_path, net)

    trainingset = ImageDataset()
    print("loading hdf5 training set: {}".format(training_path))
    trainingset.load_hdf5(training_path)
    print("hdf5 file loaded.")

    print("Getting sub dataset from filename filters...")
    seeds_dataset = trainingset.sub_dataset_from_filename(filename_start_with=fname_start, filename_end_with=fname_end)
    print("Merging seed-only sub dataset with original dataset")
    trainingset.merge_with_dataset(seeds_dataset)
    print("Saving merged dataset in: " + out_training_path)
    trainingset.save_hdf5(out_training_path)
    print("Done.")


if __name__ == "__main__":
    main()

