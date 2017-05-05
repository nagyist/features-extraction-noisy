from PIL.Image import DecompressionBombWarning

import PIL
import warnings

import imdataset
import numpy as np
from scipy.misc import imread, imresize, imshow

import os










def imread_list_in_ndarray(image_paths, img_size, crop_size=None, color_mode="rgb", rgb_normalize=None,
                           return_fnames_instead_paths = False,
                           skip=0, max_images=-1,
                           return_list=False,
                           progressbar=True, progressbar_path_length=18, progressbar_skip_max_relative=True, progressbar_newline=False,
                           verbose_errors=True, ignore_IOError=True, skip_big_images=True, big_images_pixels=None):
    """
    :param image_paths: list of paths to image files.
    :param img_size: resize image, list of 2 int - [x, y]
    :param crop_size: crop image (after resize), list of 2 int dimensions - [x, y]
    :param color_mode: 'rgb' or 'bgr'
    :param rgb_normalize: list of 3 floats - [r, g, b] - subtract these values from each sub-pixel value.
    :param skip: set the number of images to skip
    :param max_images: set the maximum number of images to read
    :param return_list: if True, return a list of ndarrays instead of stacking all the ndarrays of single images in a single ndarray.
    :param progressbar: if True, show a progress bar during the reading process
    :param progressbar_path_length: set the max string lenght of the current path near to the progress bar.
    :param progressbar_skip_max_relative: if True, progress bar will start to count from the first not skipped element
    :param progressbar_newline: if True, force each percentage step to be wrote on a new line
    :param verbose_errors: show on terminal the warning and errors that occurs during the process
    :param ignore_IOError: if False, IOErrors will block the reading process
    :param skip_big_images: skip images with too much pixels (image bombs)
    :param big_images_pixels: change the default number of pixels for images considered too big (image bombs)
    :return: return [ndarray, n_skips] where ndarray contains all the images stacked with np.stack. If return_list=True,
             will return [list(ndarray), n_skips] where the list contains the ndarray of the single images.

    NB: Tested on jpg and png, works with both
    """


    if skip_big_images: warnings.filterwarnings('error')  # warnings as errors
    to_reset = _set_max_image_pixels(big_images_pixels)

    return_images = []
    return_img_paths = []
    error_img_paths = []

    first, last, n_elem = imdataset.utils.subsequence_index(skip, max_images, len(image_paths))

    if progressbar:
        if progressbar_skip_max_relative:
            progressbar_max = n_elem
            absolute_progress_skip = 0
        else:
            progressbar_max = len(image_paths)
            absolute_progress_skip = first


    for i, img_path in enumerate(image_paths[first:last]):
        if progressbar:
            imdataset.utils.progress(i + absolute_progress_skip, progressbar_max,
                                     status=" - Reading image: {}".format(img_path[-progressbar_path_length:]))
            if progressbar_newline:
                print("")
        img = _imread_ndarray(img_path, img_size, crop_size, color_mode, rgb_normalize, verbose_errors, ignore_IOError, skip_big_images, big_images_pixels)
        if img is not None:
            return_images.append(img)
            return_img_paths.append(imdataset.path.path_filename(img_path) if return_fnames_instead_paths else img_path)
        else:
            error_img_paths.append(imdataset.path.path_filename(img_path) if return_fnames_instead_paths else img_path)
    if progressbar:
        if progressbar_skip_max_relative:
            finish = progressbar_max
            imdataset.utils.progress(finish, progressbar_max, status=" - Process Finished.")
            print ("")

        else:
            finish = first + n_elem
            if finish == progressbar_max:
                imdataset.utils.progress(finish, progressbar_max, status=" - Process Finished.")

    if to_reset: _reset_max_image_pixels()
    if skip_big_images:   warnings.resetwarnings()  # reset warnings as warnings
    if len(return_images) == 0:
        return_images = None
    elif return_list == False:
        return_images = ndarray_list_merge(return_images)
    return return_images, return_img_paths, error_img_paths


def imread_in_ndarray(img_path, img_size, crop_size=None, color_mode="rgb", rgb_normalize=None,
                      verbose=True, ignore_IOError=True, skip_big_images=True, big_images_pixels=None):
    if skip_big_images: warnings.filterwarnings('error')  # warnings as errors
    to_reset = _set_max_image_pixels(big_images_pixels)

    img = _imread_ndarray(img_path, img_size, crop_size, color_mode, rgb_normalize, verbose, ignore_IOError, skip_big_images, big_images_pixels)

    if to_reset: _reset_max_image_pixels()
    if skip_big_images:   warnings.resetwarnings()  # reset warnings as warnings
    return img




def ndarray_list_merge(list_of_ndarrays):
    try:
        img_batch = np.stack(list_of_ndarrays, axis=0)
    except:
        raise ValueError('image in list_of_ndarrays must have the same shapes.')
    return img_batch



_PIL_MAX_IMAGE_PIXELS_BACKUP = PIL.Image.MAX_IMAGE_PIXELS
def _set_max_image_pixels(max_image_pixels):
    if isinstance(max_image_pixels, int) and max_image_pixels > 0:
        PIL.Image.MAX_IMAGE_PIXELS = max_image_pixels
        return True
    else:
        return False
def _reset_max_image_pixels():
    PIL.Image.MAX_IMAGE_PIXELS = _PIL_MAX_IMAGE_PIXELS_BACKUP



def _channels(color_mode="rgb"):
    if color_mode == "rgb":
        channels = 3
    elif color_mode == "gray" or color_mode == "grayscale" or color_mode == "g" or color_mode == "gr":
        channels = 1
    return channels




def _imread_ndarray(img_path, img_size, crop_size=None, color_mode="rgb", rgb_normalize=None,
                    verbose_errors=True, ignore_errors=True, skip_big_images=True, big_images_pixels=None):
    '''
    :param img_path: paths to an image file.
    :param img_size: resize image, list of 2 int - [x, y]
    :param crop_size: crop image (after resize), list of 2 int dimensions - [x, y]
    :param color_mode: 'rgb' or 'bgr'
    :param rgb_normalize: list of 3 floats - [r, g, b] - subtract these values from each sub-pixel value.
    :param skip: skip the first
    :param max_images:
    :param show_progress: show progress bar
    :param ignore_errors:
    :param skip_big_images:
    :param big_images_pixels:
    :return: ndarray contains the image, if error/skip occurred return None
    '''
    try:
        img = imread(img_path, mode='RGB')

    except DecompressionBombWarning as w:
        if verbose_errors:
            print
            print "Decompression Bomb Warning reading image: {}".format(img_path)
            print "Skipping."
            if skip_big_images:
                return None

    except IOError as e:
        if verbose_errors:
            print
            print "IOError reading image: {}".format(img_path)
            print "Skipping."
        if not ignore_errors:
            raise e
        return None

    try:
        img = imresize(img ,img_size)
        # # We normalize the colors (in RGB space) with the empirical means on the training set
        if rgb_normalize is not None:
            img[:, :, 0] -= rgb_normalize[0]
            img[:, :, 1] -= rgb_normalize[1]
            img[:, :, 2] -= rgb_normalize[2]
        if color_mode == "bgr":
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]  # We permute the colors to get them in the BGR order

        img = img.transpose((2, 0, 1))  # Traspose to get image in format [ color, x, y ]

        if crop_size:
            yoff = (img_size[0] - crop_size[0]) // 2
            xoff = (img_size[1] - crop_size[1]) // 2
            img = img[:, yoff:-yoff, xoff:-xoff]
            # img = img[:, (img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2
            # , (img_size[1] - crop_size[1]) // 2:(img_size[1] + crop_size[1]) // 2]

    except Exception as e:
        if verbose_errors:
            print
            print "IOError reading image: {}".format(img_path)
            print "Skipping."
        if not ignore_errors:
            raise e
        return None
    return img





#
#
# def load_images_in_ndarray(image_paths, img_size, crop_size=None, color_mode="rgb", rgb_normalize=None,
#                            show_progress=True, ignore_IOError=True, skip_big_images=True, big_images_pixels=None):
#     '''
#
#     :param image_paths: list of paths to image files.
#     :param img_size: resize image, list of 2 int - [x, y]
#     :param crop_size: crop image (after resize), list of 2 int dimensions - [x, y]
#     :param color_mode: 'rgb' or 'bgr'
#     :param rgb_normalize: list of 3 floats - [r, g, b] - subtract these values from each sub-pixel value.
#     :param skip: skip the first
#     :param max_images:
#     :param show_progress: show progress bar
#     :param ignore_IOError:
#     :param skip_big_images:
#     :param big_images_pixels:
#     :return:
#     '''
#     img_list = []
#     if big_images_pixels is not None:
#         import PIL
#         PIL.image.MAX_IMAGE_PIXELS = big_images_pixels
#
#     if skip_big_images:
#         import warnings
#         warnings.filterwarnings('error')  # warnings as errors
#
#     io_errors = 0
#     for i, im_path in enumerate(image_paths):
#         if show_progress:
#             imdataset.utils.progress(i, len(image_paths), status=" - Processing image in label")
#
#         try:
#             img = imread(im_path, mode='RGB')
#
#         except DecompressionBombWarning as w:
#             print
#             print "Decompression Bomb Warning on File: {}".format(im_path)
#             print "Skipping."
#             print
#             continue
#
#         except IOError as e:
#             io_errors += 1
#             if not ignore_IOError:
#                 raise e
#             continue
#
#         img = imresize(img, img_size)
#
#         img = img.astype('float32')
#
#         # # We normalize the colors (in RGB space) with the empirical means on the training set
#         if rgb_normalize is not None:
#             img[:, :, 0] -= rgb_normalize[0]
#             img[:, :, 1] -= rgb_normalize[1]
#             img[:, :, 2] -= rgb_normalize[2]
#
#         if color_mode == "bgr":
#             img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]  # We permute the colors to get them in the BGR order
#
#         img = img.transpose((2, 0, 1))  # Traspose to get image in format [ color, x, y ]
#
#         if crop_size:
#             img = img[:, (img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2
#             , (img_size[1] - crop_size[1]) // 2:(img_size[1] + crop_size[1]) // 2]
#         img_list.append(img)
#
#     img_batch = np.stack(img_list, axis=0)
#
#     if skip_big_images:
#         warnings.resetwarnings()  # restore default warning behaviour
#
#     return img_batch