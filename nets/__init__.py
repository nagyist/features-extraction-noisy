from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

from nets.shallow_net import Hidden, new_model, save_model_json


def TrainedVGG16():
    return VGG16(include_top=True, weights='imagenet', input_tensor=None)

def TrainedInceptionV3():
    return InceptionV3(include_top=True, weights='imagenet', input_tensor=None)

def TrainedResNet50():
    return ResNet50(include_top=True, weights='imagenet', input_tensor=None)

def TrainedVGG19():
    return VGG19(include_top=True, weights='imagenet', input_tensor=None)


