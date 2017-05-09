import os
from keras.applications import InceptionV3, ResNet50, VGG16, VGG19



class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


_DATASET='dbp3120'
_TOY_DATASET='4_ObjectCategories'
_USE_TOY_DATASET = False
_FEATURES_PATH = "extracted_features/"
_SHALLOW_PATH = "shallow_classifier/"
_DATASET_PATH = "dataset/"

_WD_PATH = "/mnt/bulldozer/delchiaro/features-extraction"
_ALT_WD_PATH = "/home/delchiaro/mnt/bulldozer/equilibrium/delchiaro/features-extraction"


_PRE_WEIGHTS = 'imagenet'



class Cfg:
    __metaclass__ = Singleton

    def __init__(self):


        self.feature_layer_dict = {
            'vgg16':    'block5_pool',
            'vgg19':    'block5_pool',
            'vgg16_fc1': 'fc1',
            'vgg19_fc1': 'fc1',
            'resnet50': 'avg_pool',
            'inception_v3': 'avg_pool',
            'inception_v3_notop'  : None
        }

        self.size_dict = { ### using size as int not as list of int (all sizes have ratio 1:1)
            'vgg16': 256,
            'vgg19': 256,
            'vgg16_fc1': 256,
            'vgg19_fc1': 256,
            'resnet50': 256,
            'inception_v3': 320,
            'inception_v3_notop': 320,

        }
        self.crop_dict = { ### using crop as int not as list of int (all crops have ratio 1:1)
            'vgg16': 224,
            'vgg19': 224,
            'vgg16_fc1': 224,
            'vgg19_fc1': 224,
            'resnet50': 224,
            'inception_v3': 299,
            'inception_v3_notop': 299,
        }

        self.feat_shape_dict = {
            'vgg16':     (512, 7, 7),
            'vgg19':     (512, 7, 7),
            'vgg16_fc1': (4096,),
            'vgg19_fc1': (4096,),
            'resnet50':  (2048, 1, 1),
            'inception_v3': (2048, 1, 1),
            'inception_v3_notop': (2048, 1, 1),
        }

        self.trained_net_dict = {
            'vgg16':        lambda: VGG16(include_top=True, weights=Cfg.PRE_WEIGHTS, input_tensor=None),
            'vgg19':        lambda: VGG19(include_top=True, weights=Cfg.PRE_WEIGHTS, input_tensor=None),
            'vgg16_fc1':    lambda: VGG16(include_top=True, weights=Cfg.PRE_WEIGHTS, input_tensor=None),
            'vgg19_fc1':    lambda: VGG19(include_top=True, weights=Cfg.PRE_WEIGHTS, input_tensor=None),
            'resnet50':     lambda: ResNet50(include_top=True, weights=Cfg.PRE_WEIGHTS, input_tensor=None),
            'inception_v3': lambda: InceptionV3(include_top=True, weights=Cfg.PRE_WEIGHTS, input_tensor=None),
            'inception_v3_notop': lambda: InceptionV3(include_top=False, weights=Cfg.PRE_WEIGHTS, input_tensor=None),

        }


    def init(self, dataset=_DATASET,
             toy_dataset=_TOY_DATASET,
             use_toy_dataset=_USE_TOY_DATASET,
             include_nets=[],
             exclude_nets=[],
             features_path=_FEATURES_PATH,
             shallow_path=_SHALLOW_PATH,
             dataset_path=_DATASET_PATH,
             wd=_WD_PATH,
             alt_wd=_ALT_WD_PATH):
        self.dataset = dataset
        self.use_toy_dataset = use_toy_dataset

        self.included_nets = include_nets[:]
        self.excluded_nets = exclude_nets[:]

        self.features_path = features_path
        self.shallow_path = shallow_path
        self.dataset_path = dataset_path
        self.wd = wd
        self.alt_wd = alt_wd

        try:
            os.chdir(self.wd)
        except OSError:
            os.chdir(self.alt_wd)
        print("")
        print("CURRENT WORKING DIRECTORY: " + os.getcwd())
        print("")

        # # TOY DATASET CONFIG:
        if use_toy_dataset:
            self.dataset = toy_dataset
        else:
            self.dataset = dataset  # + '_noflk'


        try:
            for exc in self.excluded_nets:
                del self.feature_layer_dict[exc]
                del self.size_dict[exc]
                del self.crop_dict[exc]
                del self.feat_shape_dict[exc]
                del self.trained_net_dict[exc]
        except NameError:
            pass

        try:
            if self.included_nets is not None and len(self.included_nets) > 0:
                # for inc in INCLUDE_NETS:
                def subdict_from_keys(dict, keys):
                    return {k: dict[k] for k in [x for x in keys] if k in dict}

                self.feature_layer_dict = subdict_from_keys(self.feature_layer_dict, self.included_nets)
                self.size_dict = subdict_from_keys(self.size_dict, self.included_nets)
                self.crop_dict = subdict_from_keys(self.crop_dict, self.included_nets)
                self.feat_shape_dict = subdict_from_keys(self.feat_shape_dict, self.included_nets)
                self.trained_net_dict = subdict_from_keys(self.trained_net_dict, self.included_nets)

        except NameError:
            pass

        assert self.feature_layer_dict.keys() == self.size_dict.keys() == self.crop_dict.keys()\
               == self.feat_shape_dict.keys() == self.trained_net_dict.keys()
        self.nets = self.feature_layer_dict.keys()

        self.all_crop_size_stamp = []
        for net in self.nets:
            self.all_crop_size_stamp.append(str(self.crop_dict[net]) + "_" + str(self.size_dict[net]))
        self.all_crop_size_stamp = set(self.all_crop_size_stamp)
        self.all_crop_size = []
        for c_s in self.all_crop_size_stamp:
            c_s = c_s.split('_')
            self.all_crop_size.append({'crop': int(c_s[0]), 'size': int(c_s[1])})



    def crop_size(self, net):
        return self.crop_dict[net], self.size_dict[net]

    def net_crop_size_stamp(self, net):
        return self.crop_size_stamp(self.crop_dict[net], self.size_dict[net])

    @staticmethod
    def crop_size_stamp( crop, size):
        ret = "r{}".format(size)  # if size[0]==size[1] else "r{}x{}".format(size[0],size[1])  ### using size as int not as list of int (all sizes have ratio 1:1)
        ret += "_c{}".format(crop)  # if crop[0]==crop[1] else "_c{}x{}".format(crop[0],crop[1])
        return ret


cfg = Cfg()
