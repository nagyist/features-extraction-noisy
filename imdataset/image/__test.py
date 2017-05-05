from scipy.misc import imshow

import imdataset

current_path = "/mnt/das4-fs4/var/scratch/rdchiaro/E2_features-extraction/dataset/dataset_dbpedia_3120/" \
               "http:^^dbpedia.org^resource^1952_in_fine_arts_of_the_Soviet_Union"

im_paths = imdataset.path.lsim(current_path, absolute_path=True, img_skip=0, img_limit=-1)

imgs, errors = imdataset.image.imread_list_in_ndarray(image_paths=im_paths,
                                                      img_size=[255, 255],
                                                      # crop_size=[224, 224],
                                                # color_mode='bgr',
                                                # rgb_normalize=[-128, 0, 128],

                                                progressbar=True,
                                                      progressbar_path_length=20,
                                                      progressbar_skip_max_relative=True,
                                                      progressbar_newline=True,

                                                      ignore_IOError=True,
                                                      verbose_errors=True,

                                                      max_images=3,
                                                      skip=4,

                                                      skip_big_images=True,
                                                      big_images_pixels=4500* 4500,

                                                      return_list=True)

for img in imgs:
    imshow(img)

