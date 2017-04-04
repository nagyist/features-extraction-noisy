import os

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten
from keras.models import Sequential
import copy

from Layers import LabelFlipNoise
from Layers import OutlierNoise
from config import common
from imdataset import NetScore
from nets import Hidden


class ShallowNetBuilder():
    def __init__(self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape


    def A(self, extra_name="", lf_decay=None, outl_alpha=None):
        return self.__builder('A', extra_name, [Hidden(8000, 0.5)], lf_decay, outl_alpha)

    def H4K(self, extra_name="", lf_decay=None, outl_alpha=None):
        return self.__builder('H4K', extra_name, [Hidden(4000, 0.5)], lf_decay, outl_alpha)

    def H8K(self, extra_name="", lf_decay=None, outl_alpha=None):
        return self.__builder('H8K', extra_name,  [Hidden(8000, 0.5)], lf_decay, outl_alpha)

    def H4K_H4K(self, extra_name="", lf_decay=None, outl_alpha=None):
        return self.__builder('H4K_H4K', extra_name, [Hidden(4000, 0.5), Hidden(4000, 0.5)], lf_decay, outl_alpha)


    def __builder(self, name, extra_name, hiddens, lf_decay, outl_alpha ):
        return ShallowNet(name+extra_name, self.in_shape, self.out_shape, hiddens, lf_decay, outl_alpha)


class ShallowNet:

    def __init__(self, name, in_shape, out_shape, hiddens=[], lf_decay=None, ol_alpha=None, flatten=True):
        self.name = self.build_name(name, lf_decay, ol_alpha)
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hiddens = hiddens
        self.labelflip_decay = lf_decay
        self.outlier_alpha = ol_alpha
        self.flatten = flatten
        self._model = None
        self.weights_loaded_from=None
        self.weights_loaded_index=None

    def init(self, lf=False, ol=False):
        model = Sequential()
        input_done = False
        if self.flatten:
            model.add(Flatten(name="flatten", input_shape=self.in_shape))
            input_done = True
        for i, h in enumerate(self.hiddens):
            if input_done == False:
                model.add(Dense(h.neurons, activation='relu', name="additional_hidden_" + str(i), input_shape=self.in_shape))
            else:
                model.add(Dense(h.neurons, activation='relu', name="additional_hidden_" + str(i)))
                if h.dropout > 0:
                    model.add(Dropout(h.dropout, name="additional_dropout_" + str(i)))
        if input_done == False:
            model.add(Dense(self.out_shape, name="dense", input_shape=self.in_shape))
        else:
            model.add(Dense(self.out_shape, name="dense"))

        model.add(Activation("softmax", name="softmax"))
        if lf and self.labelflip_decay is not None:
            model.add(LabelFlipNoise(weight_decay=self.labelflip_decay, trainable=True))
        if ol and self.outlier_alpha is not None:
            model.add(OutlierNoise(alpha=self.outlier_alpha))
        self._model = model
        return self

    def model(self):
            return self._model

    def load(self, shallowLoader, weight_index,  alt_net_load=None, alt_name_load=None, extra_name=''):
        shallowLoader.load(self, weight_index, alt_net_load=alt_net_load, alt_name_load=alt_name_load, extra_name=extra_name)
        return self

    def test(self, shallowTester):
        shallowTester.test(self)


    @staticmethod
    def build_name(name, lf_decay=None, outl_alpha=None):
        ret = name
        if lf_decay is not None:
            ret += '_lf-' + str(lf_decay)
        if outl_alpha is not None:
            ret += '_outl-' + str(outl_alpha)
        return ret

class ShallowLoader:

    def __init__(self, trainset_name, feat_net_name):
        self.trainset_name = trainset_name
        self.feat_net_name = feat_net_name

    def load(self, shallow,  weight_index, alt_net_load=None, alt_name_load=None, extra_name=''):
        if alt_net_load is not None:
            name = alt_net_load.name + extra_name
        elif alt_name_load is not None:
            name = alt_name_load + extra_name
        else:
            name = shallow.name + extra_name
        shallow_path = common.shallow_path(name, self.trainset_name, self.feat_net_name, ext=False)
        model = shallow.model()
        if model is None:
            RuntimeError("You have to initialize the model with init() before loading the weights")
        model.load_weights(shallow_path + '.weights.' + str(weight_index) + '.h5', by_name=True)
        shallow.weights_loaded_from=name
        shallow.weights_loaded_index=weight_index
        return self

class ShallowTrainer:

    def __init__(self, feat_net_name, trainset_name, validset_name=None, validsplit=0, shuffle_trainset=False,
                 batch_size=32, loss='categorical_crossentropy', metric=['ACCURACY'], checkpoint_monitor=None):
        self.trainset_name = trainset_name
        self.trainset =  common.feat_dataset(trainset_name, feat_net_name)
        if shuffle_trainset:
            self.trainset.shuffle()
        #self.validset = validset
        if validset_name is None:
            self.valdata = None
        else:
            validset = common.feat_dataset(validset_name, feat_net_name)
            self.valdata = [validset.data, validset.getLabelsVec()]
        self.validsplit = validsplit
        self.batch_size = batch_size
        self.loss=loss
        self.metric=metric
        self.feat_net_name = feat_net_name
        self.chk_mon = checkpoint_monitor

        if self.chk_mon is None and (self.valdata is not None or self.validsplit > 0):
            self.chk_mon = 'val_loss'
        else:
            self.chk_mon = 'loss'

    def continue_train(self, shallow, optimizer, epochs, callbacks, weight_index, alt_shallow_name_to_load=None,
                       chk_period=-1, chk_monitor=None):
        shallow.load(weight_index, alt_shallow_name_to_load, True)
        return self.train(shallow, optimizer, epochs, callbacks, chk_period, chk_monitor)

    def train(self, shallow, optimizer, epochs, callbacks, chk_period=-1, chk_monitor=None):
        if chk_monitor == None:
            chk_monitor = self.chk_mon
        shallow_path = common.shallow_path(shallow.name, self.trainset_name, self.feat_net_name, ext=False)

        my_callbacks = callbacks[:]
        if chk_period > 0:
            name = shallow_path + '.weights.{epoch:02d}.h5'
            checkpoint = ModelCheckpoint(name, monitor=chk_monitor, save_weights_only=True, period=chk_period)
            my_callbacks.append(checkpoint)

        bestpoint = ModelCheckpoint(shallow_path + '.weights.best.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)
        my_callbacks.append(bestpoint)
        model = shallow.model()
        if model is None:
            RuntimeError("You have to initialize the model with init() before training the network")
        model.compile(optimizer=copy.deepcopy(optimizer), loss=self.loss, metrics=self.metric)
        #model.summary()
        #print("Valid split: " + str(valid_split))
        model.fit(self.trainset.data, self.trainset.getLabelsVec(),
                  validation_data=self.valdata, validation_split=self.validsplit,
                  nb_epoch=epochs, batch_size=self.batch_size,
                  callbacks=my_callbacks,
                  shuffle=True)

        save_model_json(model, shallow_path + '.json')
        model.save_weights(shallow_path + '.weights.last.h5')




class ShallowTester:
    def __init__(self, feat_net_name, trainset_name, testset_name, verbose=True, csv_global_stats=True, csv_class_stats=True,
                 single_class_verbose=False, batch_size=32, save_csv_dir='weights'):
        '''
        :param save_csv_dir: 'weights': save the csv files in the directory of the weights of the shallow network
                         'current': save the csv files in the current working directory
                          others_strings: save the csv in the specified directory

        '''
        self.trainset_name = trainset_name
        self.feat_net_name = feat_net_name
        self.testset = common.feat_dataset(testset_name, feat_net_name)
        self.testset_name = testset_name
        self.verbose = verbose
        self.single_class_verbose = single_class_verbose
        self.csv_global_stats = csv_global_stats
        self.csv_class_stats = csv_class_stats
        self.batch_size = batch_size
        self.save_csv_mode = save_csv_dir


    def test(self, shallow):

        # type: (ShallowNet) -> NetScore

        if self.verbose:
            print("Testing net: " + shallow.name)
            if shallow.weights_loaded_from is not None:
                print("Weights from:  " + shallow.weights_loaded_from)
                print("Weights index: " + str(shallow.weights_loaded_index))

        model = shallow.model()
        if model is None:
            RuntimeError("You have to initialize the model with init() before testing the network")
        predictions = model.predict(x=self.testset.data, batch_size=self.batch_size, verbose=self.verbose)
        ns = NetScore(self.testset, predictions)
        if self.verbose:
            if self.single_class_verbose:
                ns.print_perclass()
            ns.print_global()
        self._write_net_score(shallow, ns)
        return ns



    def _write_net_score(self, shallow, netScore, skip_no_example_test=True):
        ns = netScore
        model_name = shallow.name
        model_weights = str(shallow.weights_loaded_index)
        if self.csv_global_stats:
            if self.save_csv_mode=='current':
                csv_path = "global_net_score.csv"
            elif self.save_csv_mode=='weights':
                csv_path = common.shallow_folder_path(shallow.name, self.trainset_name, self.feat_net_name)
                csv_path = os.path.join(csv_path, "global_net_score.csv")
            else:
                csv_path = os.path.join(self.save_csv_mode, "global_net_score.csv")

            if os.path.isfile(csv_path):
                print('csv file exists, append!')
                csv = file(csv_path, "a")
            else:
                print('csv file not exists, create!')
                csv = file(csv_path, "w")
                csv.write("MODEL NAME\tWEIGHTS\tTESTSET NAME\t\tTOP-1\tTOP-5\tPRECISION\tRECALL\tF1 SCORE\tEXAMPLES(weights)"
                                                  "\t\tAVG TOP-1\tAVG TOP-5\tAVG PRECISION\tAVG RECALL\tAVG F1 SCORE")



            csv.write("\n{}\t{}\t{}\t\t{}\t{}\t{}\t{}\t{}\t{}\t\t{}\t{}\t{}\t{}\t{}".
                      format(model_name, model_weights, self.testset_name, ns.global_top1, ns.global_top5,
                             ns.global_precision, ns.global_recall, ns.global_f1, ns.examples,
                             ns.avg_top1, ns.avg_top5, ns.avg_precision, ns.avg_recall, ns.avg_f1))
            csv .close()

        if self.csv_class_stats:
            fname = model_name + ".weight.{}.netscore.csv".format(model_weights)
            if self.save_csv_mode == 'current':
                detailed_csv_path = fname
            elif self.save_csv_mode == 'weights':
                detailed_csv_path = common.shallow_folder_path(shallow.name, self.trainset_name, self.feat_net_name)
                detailed_csv_path = os.path.join(detailed_csv_path, fname)
            else:
                detailed_csv_path = os.path.join(self.save_csv_mode, fname)

            model_csv = file(detailed_csv_path, "w")
            model_csv.write("\t\tMODEL NAME\tWEIGHTS\tTESTSET NAME\t\tTOP-1\tTOP-5\tPRECISION\tRECALL\tF1 SCORE\tEXAMPLES")
            model_csv.write("\n\t{}\t{}\t{}\t{}\t\t{}\t{}\t{}\t{}\t{}".
                            format("Weighted AVG:", model_name, model_weights, self.testset_name, ns.global_top1, ns.global_top5,
                                   ns.global_precision, ns.global_recall, ns.global_f1, ns.examples))
            model_csv.write("\n\t{}\t{}\t{}\t{}\t\t{}\t{}\t{}\t{}\t{}".
                            format("Simple AVG:", model_name, model_weights, self.testset_name, ns.avg_top1, ns.avg_top5,
                                   ns.avg_precision, ns.avg_recall, ns.avg_f1, ns.examples))

            model_csv.write("\n\n\nPER CLASS CORE\nCLASS NAME\tCLASS\tTOP-1\tTOP-5\tPRECISION\tRECALL\tF1 SCORE\tEXAMPLES")

            l = 0
            test_example=0
            for top1, top5, prec, rec, f1, examples in zip(ns.perclass_top1, ns.perclass_top5, ns.perclass_precision,
                                                           ns.perclass_recall,ns.perclass_f1, ns.perclass_examples):
                if skip_no_example_test and examples == 0:
                    pass
                else:
                    model_csv.write('\n"{}"\t{}\t{}\t{}\t{}\t{}\t{}\t{}'
                                    .format(ns.testset.labelIntToStr(l), l, top1, top5, prec, rec, f1, examples))

                l += 1

            model_csv.write("\n\n\nEXAMPLE FILE NAME\tTRUE CLASS"
                            "\t\tPred\tPred2\tPred3\tPred4\tPred5"
                            "\t\tPr\tPr2\tPr3\tPr4\tPr5"
                            "\t\tCLASSNAME/FILENAME (LINK)")
            i=0
            for true_c, tops, probs in zip(ns.true_class_per_example, ns.top5_classes_per_example, ns.top5_probs_per_example):
                model_csv.write('\n"{}"\t{}'.format(self.testset.fnames[i], true_c))
                model_csv.write('\t\t{}\t{}\t{}\t{}\t{}'.format(tops[0], tops[1], tops[2], tops[3], tops[4]))
                model_csv.write('\t\t{}\t{}\t{}\t{}\t{}'.format(probs[0], probs[1], probs[2], probs[3], probs[4]))
                model_csv.write('\t\t"{}/{}"'.format(self.testset.labelIntToStr(true_c), self.testset.fnames[i]))



                i+=1


def save_model_json(self, out_file):
    # type: (keras.models.Model, str) -> None
    json = self.model.to_json()
    fjson = file(out_file, "w")
    fjson.write(json)
    fjson.close()
