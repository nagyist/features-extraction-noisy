# Command line application (and python function) that can compute the output of a trained vgg16 network over an images
# dataset (organized as ./path/to/dataset/label/image.jpg or as ./path/to/dataset/image.jpg) or hdf5 dataset
# (see ImageDataset.saveToHDF5()) in a specific vgg16 layer.
# The computed output can be saved into an HDF5 file or as Image in another folder.
# Refactoring of vgg16out.py


import os
import h5py
import numpy as np


from keras.models import model_from_json
import json
from imdataset.__OldImageDataset import loadImageList, ImageDataset


#
# try:
#     os.chdir("/mnt/das4-fs4/")  # when executed locally
# except OSError as e:
#     os.chdir("/")  # when executed on das-4
# os.chdir("var/scratch/rdchiaro/")

DEFAULT_WEIGHTS_FILE = "/mnt/das4-fs4/var/scratch/rdchiaro/weights/vgg16_weights.h5"
DEFAULT_VGG16_INPUT_SIZE = [256, 256]
DEFAULT_VGG16_INPUT_CROP = [224, 224]










def model_predict(model,               # type: Model
                  input,               # type: np.ndarray
                  input_layer=None,    # type: basestring
                  output_layer=None,   # type: basestring
                  verbose=False        # type: bool
                  ):
    # type: () -> np.ndarray
    from keras.models import Model

    if isinstance(model, Model) == False or isinstance(input, h5py) == False:
        raise TypeError("model_predict: model must be a keras Model object, input must be a valid h5py object.")

    import net_utils
    model = net_utils.submodel(model, input_layer_name=input_layer, output_layer_name=output_layer)
    prediction = model.predict(input, batch_size=len(input), verbose=verbose)

    if verbose:
        print prediction
    return prediction



# TODO: test all the code paths
def model_predict_helper(model,
                         input,

                         input_layer_name=None,
                         output_layer_name=None,

                         model_weights = None,
                         input_crop_size = None,
                         input_img_size = None,
                         load_folder_with_label=True,
                         h5_group_name=None,
                         load_h5file_as_ImageDataset=True,

                         verbose=True):
    '''
    :param model: path to json keras model or a keras Model instance
    :param input: numpy.ndarray or ImageDataset instance or path to folder containing image or path to single image


    :param input_layer_name: basestr, name of the input layer to use, None to default input
    :param output_layer_name: basestr, name of the output layer to use, None to default output

    :param input_crop_size: [x, y] -> list(int), use only if input is a single image path or folder path
    :param input_img_size: numpy.ndarray or ImageDataset instance or path to folder containing image or path to single image
    :param load_folder_with_label: if true will load the target input folder using each sub-folder as label.
    :param h5_group_name: specify the target group/array name to load (used if the input is a path to hdf5 file)
    :param load_h5file_as_ImageDataset: specify that the target hdf5 file must be loaded as a ImageDataset.

    :param verbose:

    :return: call model_predict and return the prediction as a numpy.ndarray
    '''
    # input: an image file, a folder containing image, a folder containing labe-folders containing image
    from net_utils import submodel
    model = model_or_path(model)
    submodel(model, input_layer_name=input_layer_name, output_layer_name=output_layer_name)
    input = _load_input_helper(input,
                               crop_size=input_crop_size, img_size=input_img_size,
                               load_folder_with_label=load_folder_with_label,
                               h5_group_name=h5_group_name,
                               load_h5file_as_ImageDataset=load_h5file_as_ImageDataset)

    if model_weights is not None:
        if os.path.isfile(model_weights):
            weights = h5py.File(model_weights)
        if isinstance(weights, h5py.File):
            model.load_weights_from_hdf5_group_by_name(weights)
        else:
            raise ValueError("weights is not a valid hdf5 file or path to a valid hdf5 file.")


    prediction = model.predict(input, batch_size=len(input), verbose=True)
    if verbose:
        print prediction
    return prediction



# TODO: test all the code paths
def _load_input_helper(input, crop_size, img_size, load_folder_with_label=True, h5_group_name = None, load_h5file_as_ImageDataset = True):
    dataset = None

    if isinstance(input, str):
        if os.path.isfile(input):

            ext = _file_extension(input)

            if ext == ".h5":
                if load_h5file_as_ImageDataset:
                    dataset = ImageDataset()
                    dataset.loadHDF5(input)
                else:
                    input = h5py.File(input, 'r')

            elif ext == "":
                dataset = ImageDataset()
                if load_folder_with_label:
                    dataset.loadFolderWithLabels(input, crop_size=crop_size, img_size=img_size, sortFileNames=True)
                else:
                    dataset.loadSingleFolder(input, crop_size=crop_size, img_size=img_size, sortFileNames=True)

            else:
                dataset = ImageDataset()
                dataset.loadSingleImage(input, crop_size=crop_size, img_size=img_size)
        else:
            raise ValueError("input (basestring type) must be a valid path to a file (hdf5/image) or folder")



    ret = None

    if dataset is not None:
        ret = dataset.data

    elif isinstance(input, np.ndarray):
        ret = input
    elif isinstance(input, h5py):
        if isinstance(h5_group_name, basestring):
            ret = input[h5_group_name][:]
        else:
            ret = input[input.keys()[0]][:]  # get the first group in the hdf5 file
    else:
        raise TypeError("input must be a basestring or h5py or a numpy.ndarray instance")
    return dataset



#
#
# # TODO: ridefinire il main
# def main(argv):
#
#     parser = ArgumentParser()
#     parser.add_argument('-i', action='store', dest='input', required=True)
#
#     parser.add_argument("-is", "--img-size", action='store', dest='input_size', default=None,
#                         docs="Specify the size of the input")
#     parser.add_argument("-ic", "--img-crop", action='store', dest='input_crop', default=None,
#                         docs="Specify input crop. ")
#
#     parser.add_argument("-w", "--weights-file", action='store', dest='weights_file', default=DEFAULT_WEIGHTS_FILE,
#                         docs="Specify the hdf5 file containing the weights of the pretrained net. ")
#
#
#
#     parser.add_argument("-n", "--net", action='store', dest='net', default="vgg16",
#                         docs="Specify the network model to use from the default network model available",
#                         choices={"vgg16"})
#
#     parser.add_argument("-nj", "--net-json", action='store', dest='net_json_file', default=None,
#                         docs="Specify the json file containing the network model definition.")
#
#     parser.add_argument("-il", "--input-layer",
#                         action='store', dest='input_layer', default=None,
#                         docs="Select the output layer, using the layer name")
#
#     parser.add_argument("-ol", "--output-layer",
#                         action='store', dest='output_layer', default=None,
#                         docs="Select the output layer, using the layer name")
#
#     # parser.add_argument("-hf", "--hdf5-file", action='store', dest='hdf5_out_path', default='output.h5',
#     #                     docs="Specify the path of the output h5 file.")
#     # parser.add_argument("-oi", "--output-image", action='store', dest='img_out_path', default=None,
#     #                     docs="Specify the path in which write the output as Image. NOT YET IMPLEMENTED.")
#
#
#     args=parser.parse_args()
#
#
#
#     # if args.net_json_file is not None:
#     #     if os.path.isfile(args.net_json_file):
#     #         file = open(args.net_json_file, "r")
#     #         jsnet = file.read()
#     #         model = model_from_json(jsnet)
#     if args.net is not None:
#         model = args.net
#     elif args.net_json_file is not None:
#         model = args.net_json_file
#
#         model_predict(input=args.input,
#                  model=model,
#                  img_size=args.input_size,
#                  img_crop_size=args.input_crop,
#                  weights=args.weights_file,
#                  input_layer=args.input_layer,
#                  output_layer=args.output_layer,
#                  verbose=True)
#
#
#
# if __name__ == "__main__":
#     main(sys.argv)
#













# UTILS METHODS:



def _file_extension(path):
    name, extension = os.path.splitext(path)
    return extension


def model_or_path(model_or_path):
    from keras.models import Model

    if isinstance(model_or_path, basestring):
        if os.path.isfile(model_or_path):
            j = json.load(model_or_path)
            ret = model_from_json(j)
        else:
            raise ValueError("model_or_path contains an invalid path (file not exists?)")
    elif isinstance(model_or_path, Model):
        ret = model_or_path
    else:
        TypeError("model_or_path must be a keras Model instance or basestring path to a .json file containing a model")
    return ret

def _string_or_path(string_or_file):
    if not isinstance(string_or_file, basestring):
        raise TypeError("string_or_file must be a basestring or a basestring path to a valid text file.")
    elif os.path.isfile(string_or_file):
        ret = open(string_or_file, 'r').read()
    else:
        ret = string_or_file
    return ret


def _hdf5_or_path(hdf5_in):
    if isinstance(hdf5_in, basestring):
        if os.path.isfile(hdf5_in):
            hdf5_in = h5py.File(hdf5_in, 'r')
        else:
            raise ValueError("HDF5 must be an h5py object or a path to a valid hdf5 file or")
    elif not isinstance(hdf5_in, h5py):
        raise ValueError("HDF5 must be an h5py object or a path to a valid hdf5 file or")
    return hdf5_in
#
# def hdf5_or_path(hdf5_or_file):
#     if os.path.isfile(hdf5_or_file):
#         h5 = h5py.File(hdf5_or_file, 'r')
#     elif isinstance(hdf5_or_file, h5py):
#         h5 = hdf5_or_file
#     else:
#         raise TypeError("hdf5_or_file must be an instance of h5py or a basestring path to a valid h5 file.")
#     return h5

# def img_or_file(img):
#     if isinstance(hdf5_in, str):
#         if os.path.isfile(hdf5_in):
#             input_hdf5 = h5py.File(hdf5_in, 'r')
#         else:
#             raise ValueError("HDF5 must be an h5py object or a path to a valid hdf5 file or")
#     elif not isinstance(hdf5_in, h5py):
#         raise ValueError("HDF5 must be an h5py object or a path to a valid hdf5 file or")
#     return input_hdf5
