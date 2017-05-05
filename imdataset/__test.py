


import imdataset

dataset = "/home/nagash/workspace/dbpedia-dataset/dataset"
h5_out = "/home/nagash/workspace/dbpedia-dataset/dataset_3160.h5"

dataset = "/home/nagash/workspace/vgg-16-finetune/dataset/4_ObjectCategories"
h5_out = "dataset_out.h5"
imdataset.imdir_label_to_hdf5_dataset(dataset_path=dataset,
                                      hdf5_out=h5_out,
                                      im_size=[255,255],
                                      im_crop=[224, 224],
                                      chunk_size_in_ram=-1,
                                      remove_label_with_no_imgs=False,
                                      skip_big_imgs=False,
                                      big_images_pixels=10000*10000,
                                      verbose=True
                                      )
print "h5 saved"


id = imdataset.ImageDataset()
id.load_hdf5(h5_out)
print "h5 loaded"
