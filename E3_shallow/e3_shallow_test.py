import sys

from E3_shallow.Shallow import ShallowNetBuilder, ShallowTester, ShallowLoader
from config import cfg, common, feat_dataset_n_classes

LOSS = 'categorical_crossentropy'
METRIC = ['accuracy']
BATCH = 32


def main(args):
    cfg.init()
    shallow_test('resnet50')



def shallow_test(feat_net='resnet50', double_seeds=True, outlier_model=False):

    if cfg.use_toy_dataset:
        validset_name = None
        valid_split = 0.3
    else:
        trainset_name = cfg.dataset + '_train' + ('_ds' if double_seeds else '')
        validset_name = cfg.dataset + '_valid'
        valid_split = 0
    testset_name = cfg.dataset + '_test'

    print("Shallow Test")
    print("Features from CNN: " + feat_net)
    print("Trained on: " + trainset_name)
    print("Testing on: " + testset_name)

    in_shape = cfg.feat_shape_dict[feat_net]
    out_shape = feat_dataset_n_classes(testset_name, feat_net)


    SNB = ShallowNetBuilder(in_shape, out_shape)
    SL = ShallowLoader(trainset_name, feat_net)
    ST =  ShallowTester(feat_net, trainset_name, testset_name, save_csv_dir='current')


    # Nets to test
    shallow_nets = [SNB.H8K, SNB.H4K, SNB.A, SNB.H4K_H4K]

    # Weights to load on nets to test
    shallow_weights_to_loads = [ '05', '10', '15', 'best', 'last']

    #  Weights to load on labelflip-finetuned nets (finetuned loading the weights  in shallow_weights_to_loads list)
    shallow_ft_lf_weights_to_load = ['00', '02', '04', '08', '12', '16', 'best', 'last']

    for sn in shallow_nets:
        for sh_i in shallow_weights_to_loads:

            # Test without LabelFlip Finetune:
            sn().init(lf=False).load(SL, sh_i).test(ST)

            # Test some of the finetuned model that use LabelFlip noise label:
            extr_n = '_ft@' + str(sh_i)
            for lf_i in shallow_ft_lf_weights_to_load:
                sn(extr_n, lf_decay=0.01).init(lf=False).load(SL, lf_i).test(ST)






if __name__ == "__main__":
    main(sys.argv)






#
# parser = ArgumentParser()
# parser.add_argument("-n", "-net", action='store', dest='net_list', required=True, type=str, nargs='+',
#                     choices=["vgg16", "vgg19", "resnet50", "inception_v3"],
#                     help="Set the net(s) to use for the training experiment.")
# # parser.add_argument("-train", action='store_true', dest='train', default=False,
# #                     help="Train experiment.")
# # parser.add_argument("-test", action='store_true', dest='test', default=False,
# #                     help="Test experiment.")
# # parser.add_argument("-class-test", action='store_true', dest='class_test', default=False,
# #                     help="Test experiment.")
# parser.add_argument("-l", "--local", action='store_true', dest='use_local', default=False,
#                     help="Test experiment.")
# parser.add_argument("-s", "--scratch", action='store_false', dest='use_local', default=False,
#                     help="Test experiment.")
# args = parser.parse_args()