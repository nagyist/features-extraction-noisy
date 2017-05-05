from config import cfg, common
from imdataset import ImageDataset
from imdataset.ImageDataset import SplitOptions

# EXPERIMENT PARAMETERS:
if cfg.use_toy_dataset:
    split_options = [SplitOptions("", 0.25)]
    exclude_file_starting_with = ["image"]
else:
    #split_options = [SplitOptions("flickr", 0.25), SplitOptions("google", 0.3)]
    split_options = [SplitOptions("google", 0.25)]  # FOR DATASET WITHOUT FLICKR
    exclude_file_starting_with = ["seed"]



def main():
    cfg.init()
    exp_split_dataset()








def exp_split_dataset():
    dataset = cfg.dataset
    for crop_size in cfg.all_crop_size:
        crop = crop_size['crop']
        size = crop_size['size']

        dataset_path = common.dataset_path(dataset, crop, size)
        train_path = common.dataset_path(dataset + '_train', crop, size)
        valid_path = common.dataset_path(dataset + '_valid', crop, size)
        split_dataset_helper(dataset_path, train_path, valid_path, split_options, exclude_file_starting_with)

    print("")
    print("All done.")


def split_dataset_helper(dataset_path,  # type: basestring
                         out_training_path,  # type: basestring
                         out_valid_path,  # type: basestring
                         split_options=[SplitOptions("", 0.3)],  # type: list(SplitOptions)
                         exclude_file_starting_with=[]  # type: list(basestring)
                         ):
    print("")
    print("")
    print("Split dataset -> train/valid")
    print("Dataset: " + dataset_path)
    print("")
    print("Train: " + out_training_path)
    print("Valid: " + out_valid_path)

    dataset = ImageDataset()
    print("loading dataset hdf5 file: {}".format(dataset_path))
    dataset.load_hdf5(dataset_path)
    print("hdf5 file loaded.")

    print("Splitting the dataset")
    training, validation = dataset.validation_per_class_split(split_options, exclude_file_starting_with)
    print("Dataset splitted.")

    print("Training set length: {}".format(len(training.data)))
    print("Validation set length: {}".format(len(validation.data)))
    print("")
    print("Saving trainig on hdf5 file: " + out_training_path)
    training.save_hdf5(out_training_path)
    print("Saving validation on hdf5 file: " + out_valid_path)
    validation.save_hdf5(out_valid_path)





if __name__ == "__main__":
    main()




# Google always have 20 images, choosing a percentage of 0.3 we fix 14 google images in training and 6 in validation.
# flickr max images are 12, i.e. with percentage 0.23 we have max 3 images in validation
# For all possible images we have:
#               F.val G.val   TOT val.      TOT train.
# 12 -> (9, 3)                  9               29 (+1 seed?)
# 11 -> (8, 3)--- 3 + 6         9               28 (+1 seed?)
# 10 -> (8, 2)--  2             8               28 (+1 seed?)
# 9  -> (7, 2)                  8               27 (+1 seed?)
# 8  -> (6, 2)--  2             8               26 (+1 seed?)
# 7  -> (6, 2)-   1             7               26 (+1 seed?)
# 6  -> (6, 1)                  7               25 (+1 seed?)
# 5  -> (6, 1)                  7               24 (+1 seed?)
# 4  -> (6, 1)                  7               23 (+1 seed?)
# 3  -> (6, 1)-   1             7               22 (+1 seed?)
# 2  -> (6, 0).   0             6               22 (+1 seed?)
# 1  -> (6, 0)                  6               21 (+1 seed?)

