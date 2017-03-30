import cfg
import common
from imdataset import imdir_label_to_hdf5_dataset
from inconv import input_conversion_help




def main():
    cfg.init()

    dataset = cfg.DATASET
    dataset = cfg.DATASET + '_test'  # + '_verrocchio77'
    dataset = cfg.DATASET + '_so_test'
    dataset = cfg.DATASET + '_noflk'
    dataset = 'outliers'

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
    print("")
    print("All done.")

if __name__ == "__main__":
    main()
