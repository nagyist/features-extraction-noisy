import os
from keras.applications import InceptionV3, ResNet50, VGG16, VGG19






class cfg:
    __DATASET='dbp3120'
    __TOY_DATASET='4_ObjectCategories'
    __USE_TOY_DATASET = False

    __FEATURES_PATH = "extracted_features/"
    __SHALLOW_PATH = "shallow_classifier/"
    __DATASET_PATH = "dataset/"

    __WD_PATH = "/home/nagash/workspace/zz_SCRATCH_BACKUP/E2_features-extraction"
    __ALT_WD_PATH = "/var/scratch/rdchiaro/E2_features-extraction"  # esecute in SCRATCH
    # __ALT_WD_PATH = "/local/rdchiaro/E2_features-extraction"       # execute in LOCAL
    __ALT_WD_PATH = "/var/node436/local/rdchiaro/E2_features-extraction"  # esecute in NODE436 LOCAL!


    feature_layer_dict = {
        'vgg16':    'block5_pool',
        'vgg19':    'block5_pool',
        'vgg16_fc1': 'fc1',
        'vgg19_fc1': 'fc1',
        'resnet50': 'avg_pool',
        'inception_v3': 'avg_pool',
        'inception_v3_notop'  : None
    }

    size_dict = { ### using size as int not as list of int (all sizes have ratio 1:1)
        'vgg16': 256,
        'vgg19': 256,
        'vgg16_fc1': 256,
        'vgg19_fc1': 256,
        'resnet50': 256,
        'inception_v3': 320,
        'inception_v3_notop': 320,

    }
    crop_dict = { ### using crop as int not as list of int (all crops have ratio 1:1)
        'vgg16': 224,
        'vgg19': 224,
        'vgg16_fc1': 224,
        'vgg19_fc1': 224,
        'resnet50': 224,
        'inception_v3': 299,
        'inception_v3_notop': 299,
    }

    feat_shape_dict = {
        'vgg16':     (512, 7, 7),
        'vgg19':     (512, 7, 7),
        'vgg16_fc1': (4096,),
        'vgg19_fc1': (4096,),
        'resnet50':  (2048, 1, 1),
        'inception_v3': (2048, 1, 1),
        'inception_v3_notop': (2048, 1, 1),
    }

    PRE_WEIGHTS = 'imagenet'
    trained_net_dict = {
        'vgg16':        lambda: VGG16(include_top=True, weights=cfg.PRE_WEIGHTS, input_tensor=None),
        'vgg19':        lambda: VGG19(include_top=True, weights=cfg.PRE_WEIGHTS, input_tensor=None),
        'vgg16_fc1':    lambda: VGG16(include_top=True, weights=cfg.PRE_WEIGHTS, input_tensor=None),
        'vgg19_fc1':    lambda: VGG19(include_top=True, weights=cfg.PRE_WEIGHTS, input_tensor=None),
        'resnet50':     lambda: ResNet50(include_top=True, weights=cfg.PRE_WEIGHTS, input_tensor=None),
        'inception_v3': lambda: InceptionV3(include_top=True, weights=cfg.PRE_WEIGHTS, input_tensor=None),
        'inception_v3_notop': lambda: InceptionV3(include_top=False, weights=cfg.PRE_WEIGHTS, input_tensor=None),

    }




    @staticmethod
    def init(dataset=__DATASET,
             toy_dataset=__TOY_DATASET,
             use_toy_dataset=__USE_TOY_DATASET,
             include_nets=[],
             exclude_nets=[],
             features_path=__FEATURES_PATH,
             shallow_path=__SHALLOW_PATH,
             dataset_path=__DATASET_PATH,
             wd=__WD_PATH,
             alt_wd=__ALT_WD_PATH):

        try:
            os.chdir(wd)
        except OSError:
            os.chdir(alt_wd)
        print("")
        print("CURRENT WORKING DIRECTORY: " + os.getcwd())
        print("")

        # # TOY DATASET CONFIG:
        if use_toy_dataset:
            cfg.DATASET = toy_dataset
        else:
            cfg.DATASET = dataset  # + '_noflk'


        #cfg.INCLUDE_NETS = ['resnet50', 'vgg16_fc1', 'vgg19_fc1']
        # cfg.EXCLUDED_NETS = ['vgg16', 'vgg19', 'inception_v3']

        cfg.INCLUDE_NETS = include_nets[:]
        cfg.EXCLUDE_NETS = exclude_nets[:]

        try:
            for exc in cfg.EXCLUDED_NETS:
                del cfg.feature_layer_dict[exc]
                del cfg.size_dict[exc]
                del cfg.crop_dict[exc]
                del cfg.feat_shape_dict[exc]
                del cfg.trained_net_dict[exc]
        except NameError:
            pass

        try:
            if cfg.INCLUDE_NETS is not None and len(cfg.INCLUDE_NETS) > 0:
                # for inc in INCLUDE_NETS:
                def subdict_from_keys(dict, keys):
                    return {k: dict[k] for k in [x for x in keys] if k in dict}

                cfg.feature_layer_dict = subdict_from_keys(cfg.feature_layer_dict, cfg.INCLUDE_NETS)
                cfg.size_dict = subdict_from_keys(cfg.size_dict, cfg.INCLUDE_NETS)
                cfg.crop_dict = subdict_from_keys(cfg.crop_dict, cfg.INCLUDE_NETS)
                cfg.feat_shape_dict = subdict_from_keys(cfg.feat_shape_dict, cfg.INCLUDE_NETS)
                cfg.trained_net_dict = subdict_from_keys(cfg.trained_net_dict, cfg.INCLUDE_NETS)

        except NameError:
            pass

        assert cfg.feature_layer_dict.keys() == cfg.size_dict.keys() == cfg.crop_dict.keys()\
               == cfg.feat_shape_dict.keys() == cfg.trained_net_dict.keys()
        cfg.NETS = cfg.feature_layer_dict.keys()


        cfg.ALL_CROP_SIZE_STAMP = []
        for net in cfg.NETS:
            cfg.ALL_CROP_SIZE_STAMP.append(str(cfg.crop_dict[net]) + "_" + str(cfg.size_dict[net]))
        cfg.ALL_CROP_SIZE_STAMP = set(cfg.ALL_CROP_SIZE_STAMP)
        cfg.ALL_CROP_SIZE = []
        for c_s in cfg.ALL_CROP_SIZE_STAMP:
            c_s = c_s.split('_')
            cfg.ALL_CROP_SIZE.append({'crop': int(c_s[0]), 'size': int(c_s[1])})



    @staticmethod
    def crop_size(net):
        return cfg.crop_dict[net], cfg.size_dict[net]

    @staticmethod
    def net_crop_size_stamp(net):
        return cfg.crop_size_stamp(cfg.crop_dict[net], cfg.size_dict[net])

    @staticmethod
    def crop_size_stamp(crop, size):
        ret = "r{}".format(
            size)  # if size[0]==size[1] else "r{}x{}".format(size[0],size[1])  ### using size as int not as list of int (all sizes have ratio 1:1)
        ret += "_c{}".format(crop)  # if crop[0]==crop[1] else "_c{}x{}".format(crop[0],crop[1])
        return ret

