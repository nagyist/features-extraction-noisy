import config
from config import common
from imdataset import ImageDataset



# PARAMS
if config.USE_TOY_DATASET:
    FNAME_START_WITH= "image_"
else:
    FNAME_START_WITH = "seed"


def main():
    config.init()

    for crop_size, crop_size_stamp in zip(config.ALL_CROP_SIZE, config.ALL_CROP_SIZE_STAMP):
        crop = crop_size['crop']
        size = crop_size['size']

        dataset_path = common.dataset_path(config.DATASET, crop, size)
        out_dataset_path =  common.dataset_path(config.DATASET + '_so', crop, size)


        print("")
        print("")
        print("Seed only dataset.")
        print("Original dataset: " + dataset_path)
        print("Out seed only dataset: " + out_dataset_path)
        print("")


        trainingset = ImageDataset()
        print("loading hdf5 dataset set: {}".format(dataset_path))
        trainingset.load_hdf5(dataset_path)
        print("hdf5 file loaded.")

        print("Getting sub dataset from filter (seed files)...")
        seeds_dataset = trainingset.sub_dataset_from_filename(filename_start_with=FNAME_START_WITH, remove_empty_classes=True)
        print("Saving merged dataset in: " + out_dataset_path)
        seeds_dataset.save_hdf5(out_dataset_path)
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

