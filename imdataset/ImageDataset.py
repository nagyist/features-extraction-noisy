from enum import Enum
from os.path import exists

from collections import Iterable

import imdataset
import h5py
import scipy
from scipy.misc import imread, imresize, imsave, imshow
import numpy as np
import os

DATA_DATASET_NAME = 'data'
LABELS_DATASET_NAME = 'labels'
# SECONDARY_LABEL_DATASET_NAME= 'secondary_labels'
LABELS_NAME_DATASET_NAME = 'labels_name'
FILENAMES_DATASET_NAME = 'file_names'






class PercentageOption(Enum):
    respect_to_class_data = 0
    respect_to_filtered_data = 1


class SplitOptions:
    def __init__(self, file_starting_with, percentage, percentage_option=PercentageOption.respect_to_filtered_data,
                 int_approximation=np.rint):
        """

        :param file_starting_with: filter the file starting with this string. Use an empty string to get all files.
        :param percentage:
        :param percentage_option:
        :param int_approximation: a function taking a float and returning an int (example: int, np.math.floor)
        """
        assert 0 <= percentage <= 1
        self.file_starting_with = file_starting_with
        # should be a regular expression.. for semplcity I use startswith, it's what I need right now
        self.percentage = percentage
        self.percentage_option = percentage_option
        self.int_approximation = int_approximation
        # int() function, floor() function, other lambda function taking float and return int



def imdir_label_to_hdf5_dataset(dataset_path, hdf5_out,
                                im_size, im_crop,
                                remove_label_with_no_imgs=True,
                                chunk_size_in_ram=300,
                                save_fnames=True,
                                data_dtype=np.uint8,
                                label_dtype=np.uint16,
                                rgb_normalize=None,
                                color_mode='rgb',  # rgb or bgr
                                skip_big_imgs=False,
                                big_images_pixels=None,
                                verbose=True):
    """

    :param dataset_path:
    :param hdf5_out:
    :param im_size:
    :param im_crop:
    :param chunk_size_in_ram:
    :param data_dtype: uint8, int16, int32, f, i8 ...
    :param label_dtype: uint8, int16, int32, f, i8 ...
    :param verbose:
    :return:
    """

    label_name_list = imdataset.path.lsdir(dataset_path, absolute=False, sortByName=True)

    if verbose:
        print("Label folders with images to hdf5 dataset file: sequential mode (1 thread)")

    h5f = h5py.File(hdf5_out + ".tmp", 'w')
    h5_dataset = None
    h5_labelset = None
    h5_fnameset = None

    empty_labels = []
    saved_img_files = []

    n_labels = len(label_name_list)
    label = 0

    while label < n_labels:
        label_name = label_name_list[label]
        at_least_one_example = False
        if verbose:
            print("")
            print("Processing label {} of {}, label name = '{}'".format(label + 1, n_labels, label_name))
        label_path = os.path.join(dataset_path, label_name)

        fnames = imdataset.path.lsim(label_path, absolute_path=True,
                                     # extensions=['.jpg','.jpeg'],
                                     sort_by_name=True)

        n_files = len(fnames)
        if chunk_size_in_ram < 0 or chunk_size_in_ram > n_files:
            chunk_size = n_files
        else:
            chunk_size = chunk_size_in_ram
        if n_files > 0:
            chunk_range = range(0, n_files, chunk_size)
        else:
            chunk_range = []

        for j in chunk_range:

            [data, ok_files, error_files] = \
                imdataset.image.imread_list_in_ndarray(fnames, img_size=im_size, crop_size=im_crop,
                                                       color_mode=color_mode, rgb_normalize=rgb_normalize,
                                                       skip=j, max_images=chunk_size,
                                                       return_fnames_instead_paths=True,
                                                       progressbar=verbose,
                                                       progressbar_skip_max_relative=False,
                                                       progressbar_newline=False,
                                                       verbose_errors=verbose,
                                                       skip_big_images=skip_big_imgs,
                                                       big_images_pixels=big_images_pixels)
            if save_fnames:
                saved_img_files += ok_files

            if data is not None:
                at_least_one_example = True
                if h5_dataset is None:
                    h5_dataset = h5f.create_dataset(DATA_DATASET_NAME, data=data, dtype=data_dtype,
                                                    maxshape=(None, None, None, None), chunks=data.shape)
                    labels = np.zeros([len(data), 1])
                    labels[:, :] = label
                    h5_labelset = h5f.create_dataset(LABELS_DATASET_NAME, data=labels, dtype=label_dtype,
                                                     maxshape=(None, None), chunks=labels.shape)



                else:
                    oldsize = h5_dataset.shape[0]
                    h5_dataset.resize(oldsize + data.shape[0], axis=0)
                    h5_dataset[oldsize:] = data

                    oldsize = h5_labelset.shape[0]
                    labels = np.zeros([len(data), 1])
                    labels[:, :] = label
                    h5_labelset.resize(oldsize + labels.shape[0], axis=0)
                    h5_labelset[oldsize:] = labels

        label += 1
        if not at_least_one_example:
            empty_labels.append(label_name)

            if verbose:
                print("")
                print("No valid images for this label/folder.")
            if remove_label_with_no_imgs:
                label_name_list.remove(label_name)
                n_labels -= 1
                label -= 1
                if verbose:
                    print("label removed from dataset.")

    if saved_img_files:
        fnames_array = np.asarray(saved_img_files)
        h5f.create_dataset(FILENAMES_DATASET_NAME, data=fnames_array)
    h5f.create_dataset(LABELS_NAME_DATASET_NAME, data=label_name_list)
    h5f.close()
    os.rename(hdf5_out + ".tmp", hdf5_out)
    if verbose:
        print("")
        print("")

        print("*******************************************************" + "*" * len(hdf5_out))
        print("Process Finished for all labels, hdf5 file written in {}".format(hdf5_out))
        print("*******************************************************" + "*" * len(hdf5_out))

    return empty_labels


    # TODO? Outliers
    # if outlier_label is not None:
    #     outlier_label_found = False
    #     newlabels = []
    #     for l in labels:
    #         if l == outlier_label:
    #             outlier_label_found = True
    #         else:
    #             newlabels.append(l)
    #     if outlier_label_found:
    #         newlabels.append(outlier_label)

    # parallel_threads = 0
    # if parallel_threads>=1:
    # pass
    # #TODO: parallelization
    # # if verbose:
    # #     print("ImageDataset.loadFolderWithLabels - parallel mode ({} threads)".format(parallel_threads))
    # # from joblib import Parallel, delayed
    # # import multiprocessing
    # # num_cores = parallel_threads  # multiprocessing.cpu_count()
    # # num_cores =  multiprocessing.cpu_count()
    # #
    # # results = Parallel(n_jobs=num_cores)(delayed(_job)(i, labels, dataset_path, max_img_per_label, crop_size, img_size,
    # #                                                    color_mode, imgExtensions, sortFileNames) for i in range(len(labels)))
    # #
    # # for i, result in enumerate(results):
    # #     if verbose:
    # #         _progress(i, len(results), status=" - Reducing parallel results")
    # #
    # #     [d, fname] = result
    # #     labelnames.append(l)
    # #
    # #     dataset = np.concatenate([dataset, d], 0)
    # #     l = np.zeros([len(d), len(labels)])
    # #     l[:, i] = 1
    # #     labelset = np.concatenate([labelset, l], 0)
    # #     for j in range(0, len(fname)): fname[j] = _path_leaf(fname[j])
    # #     filenames = filenames + fname
    #
    #
    # else:



class ImageDataset:
    # load and manage an image dataset.
    # Can save to hdf5 file and load from a saved hdf5 file, numpy arrays, and folder containing folders.
    # Manage data, labels, label names (and file names).
    # Can save the content again in image files format.
    # TODO: not all methods are tested (in particular loadSingleImage and loadImagesFolder).
    # TODO: second label never implemented completely, not useful for now, for this reason has been commented.

    def __init__(self):
        self.__resetState()

    def __resetState(self):
        self.data = None
        self.labels = None
        self.labelsize = None
        self.fnames = None
        self.labelnames = None

    def __setState(self, dataset, labelset, filenames, labelnames=None, labelsize=None):
        if dataset is not None:
            self.data = dataset
        if filenames is not None:
            self.fnames = filenames
            assert self.data.shape[0] == self.fnames.shape[0]

        if labelset is not None or labelnames is not None or labelsize is not None:
            # Assign labels
            if labelset is not None:
                self.labels = labelset
                if self.data is not None:
                    assert self.labels.shape[0] == self.data.shape[0]
            if labelnames is not None:
                self.labelnames = labelnames

            if labelsize is not None:
                self.labelsize = labelsize
            elif self.labelnames is not None:
                self.labelsize = self.labelnames.shape[0]
            elif self.labels is not None:
                self.labelsize = max(self.labels) + 1
            else:
                self.labelsize = None

            # Check labels
            if self.labelsize is not None:
                if self.labels is not None:
                    assert max(self.labels) <= self.labelsize
                if self.labelnames is not None:
                    assert len(self.labelnames) == self.labelsize

                    # self.second_labels = np.asarray(secondary_label) if secondary_label is not None else None

    def load_ndarray(self, data, labels=None, fnames=None, labelnames=None, labelsize=None):
        # TODO: TEST
        # TODO: What about the maps?
        dim_d = dim_f = dim_l = -1
        if data is not None: dim_d = data.shape[0]
        if labels is not None: dim_l = labels.shape[0]
        if fnames is not None: dim_f = len(fnames)

        if labels is None and (labelnames is not None):
            raise ValueError("labelnames can be used only if labels is not None")
        # if labels is not None and labelnames is not None:
        #     if len(labelnames) != dim_l:
        #         ValueError("len(labelnames) must be equal to labels.shape[0]")

        if dim_d != -1 and dim_l != -1 and dim_l != -1:
            if not (dim_d == dim_f == dim_l):
                raise ValueError("data, labels and fnames must have the same length (shape[0])")
        elif dim_d != -1 and dim_l != -1:
            if dim_d != dim_l:
                raise ValueError("data and labels must have the same length (shape[0])")

        elif dim_d != -1 and dim_f != -1:
            if dim_d != dim_f:
                raise ValueError("data and fnames must have the same length (shape[0])")

        elif dim_l != -1 and dim_f != -1:
            if dim_l != dim_f:
                raise ValueError("labels and fnames must have the same length (shape[0])")

        elif dim_d == -1 and dim_l != -1 and dim_f != -1:
            raise ValueError("At east one of data, labels or fnames array must not be None")

        self.__setState(data, labels, fnames, labelnames, labelsize)
        return self

    def load_image_folder_dataset(self, dataset_path, im_size, im_crop, hdf5_out=None, remove_labels_with_no_imgs=True,
                                  chunk_size_in_ram=300, save_fnames=True, data_dtype=np.uint8, label_dtype=np.uint16,
                                  rgb_normalize=None, color_mode='rgb', skip_big_imgs=False, big_images_pixels=None,
                                  verbose=True):
        if hdf5_out == None:
            tmp = True
            hdf5_out = "image_dataset_hdf5.tmp"
        else:
            tmp = False
        imdir_label_to_hdf5_dataset(dataset_path, hdf5_out,
                                    im_size, im_crop,
                                    remove_label_with_no_imgs=remove_labels_with_no_imgs,
                                    chunk_size_in_ram=chunk_size_in_ram,
                                    save_fnames=save_fnames,
                                    data_dtype=data_dtype,
                                    label_dtype=label_dtype,
                                    rgb_normalize=rgb_normalize,
                                    color_mode=color_mode,  # rgb or bgr
                                    skip_big_imgs=skip_big_imgs,
                                    big_images_pixels=big_images_pixels,
                                    verbose=verbose)
        self.load_hdf5(hdf5_out)
        if tmp:
            os.remove(hdf5_out)
        return self

    @staticmethod
    def h5_data_shape(path):
        h5f = h5py.File(path, 'r')
        return h5f[DATA_DATASET_NAME].shape

    @staticmethod
    def h5_label_size(path):
        h5f = h5py.File(path, 'r')
        return h5f[LABELS_NAME_DATASET_NAME].shape

    def load_hdf5(self, path, copy_in_ram=True):
        h5f = h5py.File(path, 'r')
        if DATA_DATASET_NAME not in h5f.keys() and LABELS_DATASET_NAME not in h5f.keys() and FILENAMES_DATASET_NAME not in h5f.keys():
            h5f.close()
            raise ValueError('Cannot find any dataset in this hdf5 file.')
        else:
            data = labels = fnames = labelsmap = None
            if DATA_DATASET_NAME in h5f.keys():
                #data = h5f[DATA_DATASET_NAME][:]
                data = h5f[DATA_DATASET_NAME]
                data = data[:] if copy_in_ram else data
            if LABELS_DATASET_NAME in h5f.keys():
                labels = h5f[LABELS_DATASET_NAME]
                labels = labels[:] if copy_in_ram else labels
            if FILENAMES_DATASET_NAME in h5f.keys():
                fnames = h5f[FILENAMES_DATASET_NAME]
                fnames = fnames[:] if copy_in_ram else fnames
            if LABELS_NAME_DATASET_NAME in h5f.keys():
                labelsmap = h5f[LABELS_NAME_DATASET_NAME]
                labelsmap = labelsmap[:] if copy_in_ram else labelsmap
            if copy_in_ram:
                h5f.close()

            self.__setState(data, labels, fnames, labelsmap)
        return self

    def save_hdf5(self, path, write_data=True, write_labels=True, write_fnames=True, write_labels_maps=True,
                  labelsize=None):
        h5f = h5py.File(path, 'w')
        if write_data and self.data is not None:
            h5f.create_dataset(DATA_DATASET_NAME, data=self.data)
        if write_labels and self.labels is not None:
            if self.labels.dtype != np.int16:
                self.labels = self.labels.astype(np.int16)
            h5f.create_dataset(LABELS_DATASET_NAME, data=self.labels, dtype=np.uint16)
        if write_fnames and self.fnames is not None:
            # asciiList = [n.decode("ascii", "ignore") for n in self.fnames]
            h5f.create_dataset(FILENAMES_DATASET_NAME, data=self.fnames)
        if write_labels_maps and self.labelnames is not None:
            h5f.create_dataset(LABELS_NAME_DATASET_NAME, data=self.labelnames)
        h5f.close()
        return self

    def save_dataset_folder_images(self, directory, folder_per_label=True, label_in_name=True,
                                   use_fnames_if_available=True):

        if self.data is None:
            if self.labels is None or folder_per_label == False:
                raise ValueError("Can't save to image: data is None, labels is None or folder_per_label disabled")

        use_fnames = False
        if self.fnames is not None and use_fnames_if_available:
            use_fnames = True

        if self.data is not None:

            if not exists(directory):
                os.makedirs(directory)

            d_index = 0
            for d in self.data:
                filename = str(d_index)

                if label_in_name:
                    label_index = self.getLabelInt(d_index)
                    if label_index is not None:
                        filename += '_label_' + str(label_index)
                        label_name = self.labelIntToStr(label_index)
                        if label_name is not None: filename += '_' + label_name

                if use_fnames and self.fnames is not None:
                    if folder_per_label:
                        if not exists(os.path.join(directory, label_name)):
                            os.makedirs(os.path.join(directory, label_name))
                        filename = os.path.join(label_name, self.fnames[d_index])
                    else:
                        filename += '_' + self.fnames[d_index]

                vismap_index = 0

                if len(d) == 1:
                    scipy.misc.imsave(os.path.join(directory, (filename)), d)

                elif len(d) == 3:
                    rgbArray = np.zeros((d[0].shape[0], d[0].shape[1], 3), 'uint8')
                    rgbArray[..., 0] = d[0]
                    rgbArray[..., 1] = d[1]
                    rgbArray[..., 2] = d[2]
                    #imshow(rgbArray)
                    scipy.misc.imsave(os.path.join(directory, (filename)), rgbArray)

                else:
                    for vismap in d:
                        # TODO: if use_fnames == True ...
                        scipy.misc.imsave(os.path.join(directory, (filename + '_featuremap_' + str(vismap_index) + '.jpg')), vismap)
                        vismap_index += 1
                d_index += 1

        elif folder_per_label and self.labels is not None:
            i = 0
            for l in self.labels:
                np.os.makedirs(np.os.path.join(directory, str(l)))  # TODO: use the map instad str(l)  if available

                i += 1

        return self

    def getLabelsInt(self):
        return self.labels

    def getLabelsVec(self, aslist=False):
        if self.labels is not None:
            ret = [self.labelIntToVec(x) for x in self.labels]
            if aslist:
                return ret
            else:
                return np.asarray(ret)
        else:
            return None

    def getLabelsStr(self, aslist=False):
        if self.labels is not None:
            ret = [self.labelIntToStr(x) for x in self.labels]
            if aslist:
                return ret
            else:
                return np.asarray(ret)
        else:
            return None

    def getLabelVec(self, data_index):
        return self.labelIntToVec(self.labels[data_index])

    def getLabelInt(self, data_index):
        if self.labels is not None:
            label = self.labels[data_index]
            if isinstance(label, int):
                return label
            elif isinstance(label, np.ndarray):
                return label[0]
        else:
            return None

    def getLabelStr(self, data_index):
        return self.labelIntToStr(self.getLabelInt(data_index))

    # def getSecondLabelVec(self, data_index):
    #     if self.labels is not None:
    #         return self.second_labels[data_index]
    #     else: return None
    #
    # def getSecondLabelInt(self, data_index):
    #     if self.labels is not None:
    #         return self.labelVecToInt(self.getSecondLabelVec(data_index))
    #     else: return None
    #
    # def getSecondLabelStr(self, data_index):
    #     return self.labelIntToStr(self.getSecondLabelInt(data_index))





    def labelIntToStr(self, label_int):
        if self.labelnames is not None:
            return self.labelnames[label_int]
        else:
            return None

    def labelIntToVec(self, label_int):
        if self.labels is not None and self.labelsize is not None:
            return imdataset.utils.index_vector(self.labelsize, label_int)
        else:
            return None

    def labelVecToInt(self, label_vector):
        # np.argmax(label_vector, 0)
        return np.nonzero(label_vector)[0][0]

    def labelVecToStr(self, label_vector):
        return self.labelIntToStr(self.labelVecToInt(label_vector))

    def labelStrToInt(self, label_name_str):
        if self.labelnames is not None:
            dict = {}
            for i, l in self.labelnames:
                dict[l] = i
            return dict[label_name_str]

    def labelStrToVec(self, label_name_str):
        index = self.labelStrToInt(label_name_str)
        if index is not None:
            return self.labelIntToVec(index)
        else:
            return None

    # def getSecondLabelsInt(self):
    #     return np.argmax(self.second_labels, axis=1)
    #     # label_list = []
    #     # for l in self.labels:
    #     #     label_list.append( self.labelVecToInt(l) )
    #     # return np.asarray(label_list)
    #
    # def getSecondLabelsVec(self):
    #     return self.second_labels
    #
    # def getSecondLabelsStr(self):
    #     label_list = []
    #     for l in self.second_labels:
    #         label_list.append(self.labelVecToStr(l))
    #     return np.asarray(label_list)



    def get_data_with_label(self, label):
        indices = np.argwhere(self.labels == label)
        return self.data[indices[:, 0]]

    def validation_per_class_split(self,
                                   validation_inclusion_rules=[SplitOptions("", 0.2)],  # type: list(SplitOptions)
                                   exclude_file_starting_with=[],  # type: list(basestring)
                                   ):
        """

        :param validation_inclusion_rules:
        :param exclude_file_starting_with:
        :return: training_set, validation_set
        NB: exclusion has the priority on the inclusion (first we exclude data following all the exclusion rules,
        than we include only the data that satisfy one of the inclusion rule)

        """

        training_set = None
        validation_set = None
        for label in range(0,self.labelsize):
            sub = self.sub_dataset_with_label(label)

            training_only_indices = []
            validation_legal_indices_per_rule = []  # one for each validation_inclusion_rule
            for i in range(0, len(validation_inclusion_rules)):
                validation_legal_indices_per_rule.append([])

            for index, fname in enumerate(sub.fnames):
                excluded = False
                for exclude_fname in exclude_file_starting_with:
                    if fname.startswith(exclude_fname):
                        excluded=True
                        training_only_indices.append(index)
                        break

                if not excluded:
                    included = False
                    for rule_index, rule in enumerate(validation_inclusion_rules):
                        if fname.startswith(rule.file_starting_with):
                            validation_legal_indices_per_rule[rule_index].append(index)
                            included = True
                            break
                    if not included:
                        training_only_indices.append(index)

            validation_indices = []
            legal_indices_for_a_rule = []
            for rule_index, legal_indices in enumerate(validation_legal_indices_per_rule):
                legal_indices_for_a_rule += legal_indices
                rule = validation_inclusion_rules[rule_index]
                if rule.percentage_option == PercentageOption.respect_to_filtered_data:
                    len_rule_valid = int(len(legal_indices) * rule.percentage)
                elif rule.percentage_option == PercentageOption.respect_to_class_data:
                    len_rule_valid = rule.int_approximation(len(sub.data) * rule.percentage)
                if len_rule_valid > 0:
                    if len_rule_valid > len(legal_indices):
                        validation_indices += legal_indices
                        raise UserWarning(
                            "Label {} - {} doesn't have sufficient examples to reach the percentage {} on rule {}".
                            format(label, sub.labelnames[label], rule.percentage, rule_index))
                    else:
                        nd_legal_indices = np.asarray(legal_indices, dtype=int)
                        choice = scipy.random.choice(a=legal_indices, size=len_rule_valid, replace=False)
                        validation_indices += choice.tolist()

            #validation_indices = set(validation_indices)
            #   should be a set without using set(), because in random.choice replace=False.
            training_only_indices += [x for x in legal_indices_for_a_rule if x not in validation_indices]

            if training_set is None:
                training_set = sub.sub_dataset_from_indices(training_only_indices)
            else:
                sub_train = sub.sub_dataset_from_indices(training_only_indices)
                if sub_train is not None:
                    training_set.merge_with_dataset(sub_train)

            if validation_set is None:
                validation_set = sub.sub_dataset_from_indices(validation_indices)
            else:
                sub_valid = sub.sub_dataset_from_indices(validation_indices)
                if sub_valid is not None:
                    validation_set.merge_with_dataset(sub_valid)

        return training_set, validation_set

    def merge_with_dataset(self, dataset):
        # type: (ImageDataset) -> ImageDataset
        assert self.labelsize == dataset.labelsize
        self.load_ndarray(data=np.concatenate((self.data, dataset.data), axis=0),
                          labels=np.concatenate((self.labels, dataset.labels), axis=0),
                          fnames=np.concatenate((self.fnames, dataset.fnames), axis=0))
        return self

    @staticmethod
    def merge_datasets(dataset_a, dataset_b, label_mode='same'):
        '''

        :param dataset_a:
        :param dataset_b:
        :param label_mode:
                'same': assume dataset_a and dataset_b have the same exact labels and label names.
                'name_merge': merge the two dataset classes/labels using the labelnames for comparison. Labels in b
                                 with a name that not exists in a will be considered new classes.
                'new': all the classes of dataset_b will be considered new classes.


        :return:
        '''
        # type: (ImageDataset, ImageDataset) -> ImageDataset
        # assert dataset_a.data.shape == dataset_b.data.shape
        # assert dataset_a.labels.shape == dataset_b.labels.shape
        # assert dataset_a.labelsize == dataset_b.labelsize
        # assert dataset_a.labelnames == dataset_b.labelnames

        if label_mode == 'name_merge':
            labelmap = [i for i in range(0, dataset_b.labelsize)]
            next_new_class = dataset_a.labelsize
            new_names = []

            for l, b_label_name in enumerate(dataset_b.labelnames):
                found_index = np.argwhere(dataset_a.labelnames==b_label_name)
                if len(found_index) > 0:
                    labelmap[l] = found_index
                else:
                    labelmap[l] = next_new_class
                    new_names.append(b_label_name)
                    next_new_class += 1

            new_labels = np.array(dataset_b.labels, copy=True)
            for i, l in enumerate(dataset_b.labels):
                new_labels[i,:] = labelmap[l[0]]
            new_labels = np.concatenate((dataset_a.labels, new_labels), axis=0)
            new_labelnames = np.concatenate((dataset_a.labelnames, new_names), axis=0)




        elif label_mode == 'new':
            new_labels = np.array(dataset_b.labels, copy=True)
            for i, l in enumerate(new_labels):
                new_labels[i,:] = l + dataset_a.labelsize
            new_labels = np.concatenate((dataset_a.labels, new_labels), axis=0)
            new_labelnames = np.concatenate((dataset_a.labelnames, dataset_b.labelnames), axis=0)



        elif label_mode == 'same':
            new_labels = dataset_b.labels[:]
            new_labels = np.concatenate((dataset_a.labels, new_labels), axis=0)
            new_labelnames = np.array(dataset_a.labelnames, copy=True)


        else:
            raise ValueError("label_mode value '{}' is not a legal value.".format(label_mode))
            return None


        merged = ImageDataset()
        merged.load_ndarray(data=np.concatenate((dataset_a.data, dataset_b.data), axis=0),
                            fnames=np.concatenate((dataset_a.fnames, dataset_b.fnames), axis=0),
                            labels=new_labels, labelnames=new_labelnames)
        return merged

    def sub_dataset_from_indices(self, indices, remove_empty_classes=False):
        # Type: (list(int)) -> ImageDataset
        assert isinstance(indices, Iterable)
        if len(indices) > 0:
            data = self.data[indices]
            labels = self.labels[indices]
            fnames = self.fnames[indices]
            # label_names = self.labelnames[label]
            # labels = np.array([len(data),])
            subset = ImageDataset()
            subset.load_ndarray(data, labels, fnames, self.labelnames, labelsize=self.labelsize)
            if remove_empty_classes:
                return subset.remove_empty_labels()
            else:
                return subset
        else:
            return None

    def sub_dataset_from_filename(self, filename_start_with="", filename_end_with="", remove_empty_classes=False):
        assert self.fnames is not None
        assert filename_start_with is not None or filename_end_with is not None
        indices = [i for i, fn in enumerate(self.fnames) if (fn.startswith(filename_start_with) and fn.endswith(filename_end_with))]
        return self.sub_dataset_from_indices(indices, remove_empty_classes)

    def sub_dataset_from_filename_multi(self, filename_start_with=[], filename_end_with=[], remove_empty_classes=False):
        assert self.fnames is not None
        assert filename_start_with is not None or filename_end_with is not None

        indices = []
        for i, fn in enumerate(self.fnames):
            ok=False
            for sw in filename_start_with:
                if fn.startswith(sw):
                    ok=True
                    break
            if ok or len(filename_start_with) == 0:
                ok=False
                for ew in filename_end_with:
                    if fn.endswith(ew):
                        ok=True
                        break
                if ok or len(filename_end_with) == 0:
                    indices.append(i)

        return self.sub_dataset_from_indices(indices,remove_empty_classes)

    def sub_dataset_with_label(self, label, remove_empty_classes=False):
        # type: (int, bool) -> ImageDataset
        indices = np.argwhere(self.labels == label)

        if len(indices) > 0:

            return self.sub_dataset_from_indices(indices[:, 0], remove_empty_classes)
            #
            # data = self.data[indices[:,0]]
            # labels = self.labels[indices[:,0]]
            # fnames = self.fnames[indices[:,0]]
            # #label_names = self.labelnames[label]
            # # labels = np.array([len(data),])
            # subset = ImageDataset()
            # subset.load_ndarray(data, labels, fnames, self.labelnames, labelsize=self.labelsize)
            # return subset
        else:
            return None


    def sub_dataset_with_labels(self, labels, remove_empty_classes=False):
        # type: (list[int], bool) -> ImageDataset
        all_indices = []
        for label in labels:
            indices = np.argwhere(self.labels == label)[:, 0]
            all_indices += list(indices)

        if len(indices) > 0:
            return self.sub_dataset_from_indices(all_indices, remove_empty_classes)
        else:
            return None

    def shuffle(self):
        [self.data, self.labels, self.fnames] = _unison_shuffled_copies_n([self.data, self.labels, self.fnames])

    def remove_empty_labels(self):
        self = ImageDataset.without_empty_labels(self)
        return self

    @staticmethod
    def without_empty_labels(orig_dataset):
        data = np.zeros((0,) + orig_dataset.data.shape[1:], dtype=np.int8)
        labels = np.zeros((0,) + orig_dataset.labels.shape[1:], dtype=np.int16)
        fnames = np.zeros((0,) + orig_dataset.fnames.shape[1:])
        labelnames = np.zeros((0,) + orig_dataset.labelnames.shape[1:])
        nl = 0
        for l in range(0, orig_dataset.labelsize):
            sub_set = orig_dataset.sub_dataset_with_label(l, remove_empty_classes=False)
            if sub_set is not None and len(sub_set.data) > 0:
                data=np.concatenate((data, sub_set.data), axis=0)
                new_labels=np.zeros(sub_set.labels.shape)
                new_labels[:,:] = int(nl)
                labels=np.concatenate((labels, new_labels), axis=0)
                fnames=np.concatenate((fnames, sub_set.fnames), axis=0)
                labelnames=np.concatenate((labelnames, np.asarray([sub_set.labelnames[l]])), axis=0)
                nl+=1
        return ImageDataset().load_ndarray(data, labels, fnames, labelnames)

def _unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def _unison_shuffled_copies_3(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


def _unison_shuffled_copies_n(list_of_arrays):
    # type: (list(collections.Iterable)) -> list(collections.Iterable)

    not_none_map = {}
    old_len = -1

    # Ignore Nones in list_of_arrays
    for i, array in enumerate(list_of_arrays):
        if array is not None:
            not_none_map[i] = True
            assert (old_len == len(array) or old_len == -1)
            old_len = len(array)
        else:
            not_none_map[i] = False

    p = np.random.permutation(old_len)
    ret = []
    for i, array in enumerate(list_of_arrays):
        if not_none_map[i]:
            ret.append(array[p])
        else:
            ret.append(None)

    return ret

