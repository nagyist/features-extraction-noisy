import h5py
import os
import imdataset
from config import cfg

# SHALLOW CLASSIFIER MODEL WEIGHTS
from imdataset import ImageDataset




########################################################################################################################
### SHALLOW NETWORK EXTRACTED FEATURES
# def shallow_feat_name(shallow_name, dataset_name, feat_net, ext=False):
#     extracted_feat_dataset = feat_fname(dataset_name, feat_net)
#     ret = "shallow_{}__{}".format(shallow_name, extracted_feat_dataset) + ('.h5' if ext else '')
#     return ret
#
# def shallow_feat_path(shallow_net_name, dataset_name, feat_net, ext=True):
#     folder = shallow_folder_path(shallow_net_name, dataset_name, feat_net)
#     if not os.path.isdir(folder):
#         os.mkdir(folder)
#     return os.path.join(folder, shallow_name(shallow_net_name, dataset_name, feat_net, ext))


########################################################################################################################
### SHALLOW NETWORK WEIGHTS
def shallow_name(shallow_name, dataset_name, feat_net, ext=False):
    extracted_feat_dataset = feat_fname(dataset_name, feat_net)
    ret = "shallow_{}__{}".format(shallow_name, extracted_feat_dataset) + ('.h5' if ext else '')
    return ret

def shallow_path(shallow_net_name, dataset_name, feat_net, ext=True):
    folder = shallow_folder_path(shallow_net_name, dataset_name, feat_net)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    return os.path.join(folder, shallow_name(shallow_net_name, dataset_name, feat_net, ext))

def shallow_folder_path(shallow_net_name, dataset_name, feat_net):
    return os.path.join(cfg.shallow_path, shallow_name(shallow_net_name, dataset_name, feat_net, False))



########################################################################################################################
### EXTRACTED FEATURE DATASET
def feat_fname(dataset_name, net_name, ext=False):
    ret =  "feat_" + dataset_name + "__" + net_name
    if cfg.feature_layer_dict[net_name] is not None:
        ret += "__" + cfg.feature_layer_dict[net_name]
    return ret + ('.h5' if ext else '')

def feat_path(dataset_name, net_name, ext=True):
    return os.path.join(cfg.features_path, feat_fname(dataset_name, net_name, ext))

def feat_dataset(dataset_name, net_name, in_ram=True):
    path = feat_path(dataset_name, net_name, True)
    print("Loading features dataset: " + path)
    try:
        return imdataset.ImageDataset().load_hdf5(path, copy_in_ram=in_ram)
    except IOError as e:
        print("Can't open file: " + path)
        raise e
        return None

def feat_dataset_shape(dataset_name, net_name):
    path = feat_path(dataset_name, net_name, True)
    return ImageDataset.h5_data_shape(path)

def feat_dataset_n_classes(dataset_name, net_name):
    path = feat_path(dataset_name, net_name, True)
    return ImageDataset.h5_label_size(path)[0]



########################################################################################################################
###  DATASET FOLDER
def folder_dataset_path(dataset_name):
    return os.path.join("dataset",dataset_name)

########################################################################################################################
### DATASET H5
def dataset_path(dataset_name, crop=None, size=None, ext=True):
    return os.path.join("dataset", dataset_fname(dataset_name, crop, size, ext))

def dataset_fname(dataset_name, crop=None, size=None, ext=False):
    return dataset_name + ('__' + cfg.crop_size_stamp(crop, size) if crop is not None and size is not None else '') \
           + ('.h5' if ext else '')

def dataset(dataset_name, crop=None, size=None, inram=True):
    return imdataset.ImageDataset().load_hdf5(dataset_path(dataset_name, crop, size, ext=True), copy_in_ram=inram)






def summary(net, dataset_path, validation_path, validation_split, optimizer, nb_epochs,
            batch_size, additional_hidden_neurons, additional_hidden_dropout ):

    print("Trained on dataset: {}".format(dataset_path))
    if validation_path is not None:
        print("Validation set: {}".format(validation_path))
    elif validation_split is not None and validation_split > 0:
        print("Validation split from dataset: " + str(validation_split*100) + "%")
    print("-----------------------------------")
    print("FEATURE EXTRACTOR: " + net)
    print("-----------------------------------")
    print("SHALLOW NET: ")
    print("Additional hidden layer neurons: " + str(additional_hidden_neurons))
    print("Dropout on additional hidden layer: " + str(additional_hidden_dropout))
    print("-----------------------------------")
    print("OPTIMIZER PARAMETERS:")
    for x, y in optimizer.get_config().items():
         print (x + ": " + str(y))
    # print("lr: " + str(lr))
    # print("decay: " + str(decay))
    # print("momentum: " + str(momentum))
    # print("nesterov: " + str(nesterov))
    print("-----------------------------------")
    print("TRAINED WITH PARAMETERS:")
    print("nb_epochs: " + str(nb_epochs))
    print("batch_size: " + str(batch_size))
    print("-----------------------------------")