import os
from argparse import ArgumentParser
import sys

import config
from imdataset import imdir_label_to_hdf5_dataset





def input_conversion_help(folder_dataset_path, output_h5, crop, size, remove_empty_labels=False, verbose=True):
    # type: (basestring, basestring, int, int, basestring, boolean, boolean) -> None
    crop = [crop, crop]
    size = [size, size]

    folder_dataset_path = folder_dataset_path
    if folder_dataset_path.endswith("/"):
        folder_dataset_path = folder_dataset_path[:-1]

    imdir_label_to_hdf5_dataset(folder_dataset_path, hdf5_out=output_h5,
                                im_size=size, im_crop=crop,
                                remove_label_with_no_imgs=remove_empty_labels,
                                # chunk_size_in_ram=300,
                                # skip_big_imgs=False,
                                # big_images_pixels=10000 * 10000,
                                verbose=verbose)

#
# def main(argv):
#     parser = ArgumentParser()
#     parser.add_argument('-n', "-net",     action='store', dest='net', required=False, choices=config.NETS)
#     parser.add_argument('-c', '-crop', "--dataset", action='store', dest='crop',  required=False, type=int)
#     parser.add_argument('-s','-size', "--dataset", action='store', dest='size',  required=False, type=int)
#     parser.add_argument('-d', "-dataset", action='store', dest='dataset_path',  required=True, type=str)
#     parser.add_argument("-v",action='store_true', dest='verbose', default=False, help="Verbose mode")
#     parser.add_argument("-rel",action='store_true', dest='remove_empty_labels', default=False, help="Remove empty labels from dataset")
#     #
#     # parser.add_argument("--local", action='store_true', dest='use_local', default=True)
#     # parser.add_argument("--scratch", action='store_false', dest='use_local', default=False)
#     args = parser.parse_args()
#     input_conversion_help(args.dataset_path, args.crop, args.size, args.net, args.remove_empty_labels, args.verbose)
#
#
# if __name__ == "__main__":
#     main(sys.argv)
#
#
#


