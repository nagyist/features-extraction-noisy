import h5py
import numpy as np
from imdataset import ImageDataset

dataset_path = '/home/nagash/workspace/zz_SCRATCH_BACKUP/E2_features-extraction/dataset/dbp3120_so__r256_c224.h5'


dataset = ImageDataset().load_hdf5(dataset_path)
# new_labels = dataset.labels.astype(np.uint16)
# dataset.labels = new_labels
# dataset.save_hdf5(dataset_path+ '.converted.h5')

dataset.save_dataset_folder_images("/home/nagash/workspace/zz_SCRATCH_BACKUP/E2_features-extraction/dataset/dbp3120_so__r256_c224")

