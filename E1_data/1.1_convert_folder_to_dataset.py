from cfg import cfg
import common
from imdataset import imdir_label_to_hdf5_dataset




def exp_convert_folder_to_dataset():
    dataset = cfg.DATASET
    for crop_size, crop_size_stamp in zip(cfg.ALL_CROP_SIZE, cfg.ALL_CROP_SIZE_STAMP):
        crop = crop_size['crop']
        size = crop_size['size']

        folder_dataset_path = common.folder_dataset_path(dataset)
        out_h5 = common.dataset_path(dataset, crop, size)

        imdir_label_to_hdf5_dataset(folder_dataset_path, out_h5,
                                    im_crop=[crop,crop], im_size=[size,size],
                                    remove_label_with_no_imgs=False,
                                    # chunk_size_in_ram=300,
                                    # skip_big_imgs=False,
                                    # big_images_pixels=10000 * 10000,
                                    verbose=True)


def main():
    cfg.init()
    exp_convert_folder_to_dataset()
    # 'dbp3120_test_verrocchio77'
    # 'dbp3120_so_test'
    # 'dbp3120_noflk'
    # 'outliers'

if __name__ == "__main__":
    main()
