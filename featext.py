import net_utils
from argparse import ArgumentParser
import sys
import cfg
from imdataset import ImageDataset





def features_extraction_helper(net, dataset_path, out_dataset_path, alternative_out_layer=None, batch_size=32, verbose=True):
    def printv(s):
        if verbose: print(s)
    dataset = ImageDataset()
    printv("\nloading dataset: " + dataset_path)
    try:
        dataset.load_hdf5(dataset_path)
    except IOError:
        print("Can't open selected file.")
        return
    printv("dataset loaded.")

    model = cfg.trained_net_dict[net]()
    if alternative_out_layer is None:
        alternative_out_layer = cfg.feature_layer_dict[net]
    feature_vectors = net_utils.extract_features(model, dataset, alternative_out_layer, batch_size, verbose)
    feature_vectors.save_hdf5(out_dataset_path)
    printv("Feature extracted dataset saved in: " + out_dataset_path)



def main(argv):

    parser = ArgumentParser()
    parser.add_argument('-n', "-net", action='store', dest='net', required=True, choices=cfg.NETS)
    parser.add_argument('-d', "-dataset", action='store', dest='dataset_path', required=True)
    parser.add_argument('-o', "-out", action='store', dest='out_dataset_path', required=True)


    parser.add_argument('-l', "-layer", action='store', dest='alternative_out_layer', required=True)
    parser.add_argument('-b', "-batch-size", action='store', dest='batch_size', default=32, type=int)
    parser.add_argument("-v", action='store_true', dest='verbose', default=False)

    # parser.add_argument("--local", action='store_true', dest='use_local', default=True)
    # parser.add_argument("--scratch", action='store_false', dest='use_local', default=False)
    a = parser.parse_args()


    cfg.init("FEATURES EXTRACTION", is_experiment=False , use_local=a.use_local)


    features_extraction_helper(a.net, a.dataset_path, a.out_dataset_path, a.alternative_out_layer, a.batch_size, a.verbose)

if __name__ == "__main__":
    main(sys.argv)





