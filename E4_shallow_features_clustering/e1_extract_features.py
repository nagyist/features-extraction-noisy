import net_utils
from E3_shallow.Shallow import ShallowNetBuilder, ShallowLoader, ShallowTester
from config import cfg, feat_dataset_n_classes
from config import common
from imdataset import ImageDataset



batch_size = 32

def main(args):
    extract_shallow_features()


def extract_shallow_features():
    feat_net = 'resnet50'
    cfg.init(include_nets=[feat_net])

    old_trainset_name = cfg.dataset + '_train_ds'
    #old_testset_name = cfg.dataset + '_test'
    dataset_name =  cfg.dataset + '_train_ds'
    #crop, size = cfg.crop_size(net=feat_net)


    print("\nloading dataset: " + dataset_name)
    try:
        dataset = common.feat_dataset(dataset_name, feat_net)
    except IOError:
        print("Can't open dataset.")
        return
    print("dataset loaded.")

    in_shape = cfg.feat_shape_dict[feat_net]
    out_shape = feat_dataset_n_classes(dataset_name, feat_net)


    B = ShallowNetBuilder(in_shape, out_shape)
    SL = ShallowLoader(old_trainset_name, feat_net)

    pretrain_weight_epoch = '10'
    labelflip_finetune_epoch = '00'
    out_layer = 'additional_hidden_0'


    extr_n = '_ft@' + pretrain_weight_epoch
    model = B.H8K(extr_n, lf_decay=0.01).init(lf=False).load(SL, labelflip_finetune_epoch).model()
    #model.summary()
    feature_vectors = net_utils.extract_features(model, dataset, out_layer, batch_size, True)
    feature_vectors.save_hdf5("shallow_extracted_features/shallow_classifier_feat_vector.h5")

if __name__ == "__main__":
    main(sys.argv)


