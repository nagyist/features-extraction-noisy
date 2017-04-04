from config import cfg, common
from imdataset import imdir_label_to_hdf5_dataset




def main():
    cfg.init(use_toy_dataset=True)

    exp_convert_folder_to_dataset()
    # 'dbp3120_test_verrocchio77'
    # 'dbp3120_so_test'
    # 'dbp3120_noflk'
    # 'outliers'



def exp_convert_folder_to_dataset():
    dataset = cfg.dataset
    for crop_size in cfg.all_crop_size:
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



if __name__ == "__main__":
    main()
