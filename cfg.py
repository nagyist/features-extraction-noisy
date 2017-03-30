import os
from keras.applications import InceptionV3, ResNet50, VGG16, VGG19

USE_TOY_DATASET = False


DAS_PATH = "/var/scratch/rdchiaro/features-extraction"  # esecute in SCRATCH
#DAS_PATH = "/local/rdchiaro/features-extraction"       # execute in LOCAL
DAS_PATH = "/var/node436/local/rdchiaro/features-extraction"  # esecute in NODE436 LOCAL!

HOME_PATH = "/home/nagash/workspace/zz_SCRATCH_BACKUP/features-extraction"



# # TOY DATASET CONFIG:
if USE_TOY_DATASET:
    DATASET = "4_ObjectCategories"
else:
    DATASET = "dbp3120"# + '_noflk'

# HOME_PATH="/var/scratch/rdchiaro/features-extraction"

FEATURES_PATH = "extracted_features/"
SHALLOW_PATH = "shallow_classifier/"
DATASET_PATH = "dataset/"




# USE THIS LIST TO EXCLUDE NETS FROM EXPERIMENTS THAT YOU WANT TO RUN
# EXCLUDED_NETS = ['vgg16', 'vgg19', 'inception_v3']

#INCLUDE_NETS = ['inception_v3_notop', 'vgg19_fc1']
INCLUDE_NETS = ['resnet50', 'vgg16_fc1', 'vgg19_fc1']


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
    'vgg16':        lambda: VGG16(include_top=True, weights=PRE_WEIGHTS, input_tensor=None),
    'vgg19':        lambda: VGG19(include_top=True, weights=PRE_WEIGHTS, input_tensor=None),
    'vgg16_fc1':    lambda: VGG16(include_top=True, weights=PRE_WEIGHTS, input_tensor=None),
    'vgg19_fc1':    lambda: VGG19(include_top=True, weights=PRE_WEIGHTS, input_tensor=None),
    'resnet50':     lambda: ResNet50(include_top=True, weights=PRE_WEIGHTS, input_tensor=None),
    'inception_v3': lambda: InceptionV3(include_top=True, weights=PRE_WEIGHTS, input_tensor=None),
    'inception_v3_notop': lambda: InceptionV3(include_top=False, weights=PRE_WEIGHTS, input_tensor=None),

}

try:
    for exc in EXCLUDED_NETS:
        del feature_layer_dict[exc]
        del size_dict[exc]
        del crop_dict[exc]
        del feat_shape_dict[exc]
        del trained_net_dict[exc]
except NameError:
    pass

try:
    #for inc in INCLUDE_NETS:
    def subdict_from_keys(dict, keys):
        return  {k: dict[k] for k in [x for x in keys] if k in dict}
    feature_layer_dict = subdict_from_keys(feature_layer_dict, INCLUDE_NETS)
    size_dict = subdict_from_keys(size_dict, INCLUDE_NETS)
    crop_dict = subdict_from_keys(crop_dict, INCLUDE_NETS)
    feat_shape_dict = subdict_from_keys(feat_shape_dict, INCLUDE_NETS)
    trained_net_dict = subdict_from_keys(trained_net_dict, INCLUDE_NETS)
except NameError:
    pass

assert feature_layer_dict.keys() == size_dict.keys() == crop_dict.keys() == feat_shape_dict.keys() == trained_net_dict.keys()
NETS = feature_layer_dict.keys()



def net_crop_size_stamp(net):
    return crop_size_stamp(crop_dict[net], size_dict[net])

def crop_size_stamp(crop, size):
    ret = "r{}".format(size)   #  if size[0]==size[1] else "r{}x{}".format(size[0],size[1])  ### using size as int not as list of int (all sizes have ratio 1:1)
    ret += "_c{}".format(crop)   #  if crop[0]==crop[1] else "_c{}x{}".format(crop[0],crop[1])
    return ret


ALL_CROP_SIZE_STAMP = []
for net in NETS:
    ALL_CROP_SIZE_STAMP.append( str(crop_dict[net])  + "_" + str(size_dict[net]) )
ALL_CROP_SIZE_STAMP = set(ALL_CROP_SIZE_STAMP)
ALL_CROP_SIZE = []
for c_s in ALL_CROP_SIZE_STAMP:
    c_s = c_s.split('_')
    ALL_CROP_SIZE.append({'crop': int(c_s[0]), 'size': int(c_s[1])})



def crop_size(net):
    return crop_dict[net], size_dict[net]

def init():
    try:
        os.chdir(HOME_PATH)
    except OSError:
        os.chdir(DAS_PATH)
    print("")
    print("CURRENT WORKING DIRECTORY: " + os.getcwd())
    print("")
