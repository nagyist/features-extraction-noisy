import os

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten
from keras.models import Sequential

import common
from Layers import LabelFlipNoise
from Layers import OutlierNoise
from imdataset import NetScore
from nets import Hidden


class ShallowNetBuilder():
    def __init__(self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape


    def A(self, lf_decay=None, outl_alpha=None):
        return self.__builder('A', [Hidden(8000, 0.5)], lf_decay, outl_alpha)

    def H4K(self, lf_decay=None, outl_alpha=None):
        return self.__builder('H4K', [Hidden(4000, 0.5)], lf_decay, outl_alpha)

    def H8K(self, lf_decay=None, outl_alpha=None):
        return self.__builder('H8K', [Hidden(8000, 0.5)], lf_decay, outl_alpha)

    def __builder(self, name, hiddens, lf_decay, outl_alpha):
        if lf_decay is not None:
            name += '_lf-' + lf_decay
        if outl_alpha is not None:
            name += '_outl-' + outl_alpha
        return ShallowNet(name, self.in_shape, self.out_shape, hiddens, lf_decay, outl_alpha)


class ShallowNet:

    def __init__(self, name, in_shape, out_shape, hiddens=[], labelflip_decay=None, outlair_alpha=None, flatten=True):
        self.name = name
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hiddens = hiddens
        self.labelflip_decay = labelflip_decay
        self.outlier_alpha = outlair_alpha
        self.flatten = flatten
        self._model = None
        self.weights_loaded_from=None
        self.weights_loaded_index=None

    def init_model(self):
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
        if self.labelflip_decay is not None:
            model.add(LabelFlipNoise(weight_decay=self.labelflip_decay, trainable=True))
        if self.outlier_alpha is not None:
            model.add(OutlierNoise(alpha=self.outlier_alpha))

        self._model = model
        return model

    def get_model(self):
        if self._model is None:
            return self.init_model()
        else:
            return self._model

    def load(self, weight_index, alt_shallow_name_to_load=None, init_model=True):
        name = self.name
        if alt_shallow_name_to_load is not None:
            name = alt_shallow_name_to_load
        shallow_path = common.shallow_path(name, self.trainset_name, self.feat_net_name, ext=False)
        if init_model:
            model = self.init_model()
        else:
            model = self.get_model()
        model.load_weights(shallow_path + '.weights.' + str(weight_index) + '.h5', by_name=True)
        self.weights_loaded_from=name
        self.weights_loaded_index=weight_index
        return self


class ShallowTrainer:

    def __init__(self, feat_net_name, trainset_name, validset=None, validsplit=0, batch_size=32,
                 loss='categorical_crossentropy', metric=['ACCURACY'], checkpoint_monitor=None):
        self.trainset_name = trainset_name
        self.trainset =  common.feat_dataset(trainset_name, feat_net_name)
        #self.validset = validset
        self.valdata = None if validset is None else [validset.data, validset.getLabelsVec()]
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
        model = shallow.get_model()
        model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metric)
        #model.summary()
        #print("Valid split: " + str(valid_split))
        model.fit(self.trainset.data, self.trainset.getLabelsVec(),
                  validation_data=self.valdata, validation_split=self.valid,
                  nb_epoch=epochs, batch_size=self.batch_size,
                  callbacks=my_callbacks,
                  shuffle=True)

        save_model_json(model, shallow_path + '.json')
        model.save_weights(shallow_path + '.weights.last.h5')




class ShallowTester:
    def __init__(self, testset, testset_name='<unknown>', verbose=True, csv_global_stats=True, csv_class_stats=True,
                 single_class_verbose=False, batch_size=32, save_csv_dir='weights'):
        '''
        :param save_csv_dir: 'weights': save the csv files in the directory of the weights of the shallow network
                         'current': save the csv files in the current working directory
                          others_strings: save the csv in the specified directory

        '''
        self.testset = testset
        self.testset_name = testset_name
        self.verbose = verbose
        self.single_class_verbose = single_class_verbose
        self.csv_global_stats = csv_global_stats
        self.csv_class_stats = csv_class_stats
        self.batch_size = batch_size
        self.save_csv_mode = save_csv_dir


    def test_net(self, shallow):

        # type: (ShallowNet) -> NetScore

        if self.verbose:
            print("Testing net: " + shallow.get_model)
            if shallow.weights_loaded_from is not None:
                print("Weights from:  " + shallow.weights_loaded_from)
                print("Weights index: " + str(shallow.weights_loaded_index))

        model = shallow.get_model()
        predictions = model.predict(x=self.testset.data, batch_size=self.batch_size, verbose=self.verbose)
        ns = NetScore(self.testset, predictions)
        if self.verbose:
            if self.single_class_verbose:
                ns.print_perclass()
            ns.print_global()
        return ns

        self._write_net_score(score, shallow.name, testset_name, "test_results.csv", detailed_csv=True)



    def _write_net_score(self, shallow, netScore, skip_no_example_test=True):
        ns = netScore
        model_name = shallow.name
        model_weights = str(shallow.weights_loaded_index)
        if self.csv_global_stats:
            if self.save_csv_mode=='current':
                csv_path = "global_net_score.csv"
            elif self.save_csv_mode=='weights':
                csv_path = common.shallow_folder_path(shallow.name, self.trainset_name, self.feat_net_name, ext=False)
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
                detailed_csv_path = common.shallow_folder_path(shallow.name, self.trainset_name, self.feat_net_name, ext=False)
                detailed_csv_path = os.path.join(detailed_csv_path, fname)
            else:
                detailed_csv_path = os.path.join(self.save_csv_mode, fname)

            model_csv = file(detailed_csv_path, "w")
            model_csv.write("\t\tMODEL NAME\tWEIGHTS\tTESTSET NAME\t\tTOP-1\tTOP-5\tPRECISION\tRECALL\tF1 SCORE\tEXAMPLES")
            model_csv.write("\n\t{}\t{}\t\t{}\t\t{}\t{}\t{}\t{}\t{}\t{}".
                            format("Weighted AVG:", model_name, model_weights, self.testset_name, ns.global_top1, ns.global_top5,
                                   ns.global_precision, ns.global_recall, ns.global_f1, ns.examples))
            model_csv.write("\n\t{}\t{}\t{}\t\t{}\t{}\t{}\t{}\t{}\t{}".
                            format("Simple AVG:", model_name, self.testset_name, ns.avg_top1, ns.avg_top5,
                                   ns.avg_precision, ns.avg_recall, ns.avg_f1, ns.examples))

            model_csv.write("\n\n\nPER CLASS CORE\nCLASS NAME\tCLASS\tTOP-1\tTOP-5\tPRECISION\tRECALL\tF1 SCORE\tEXAMPLES")
            model_csv.write("\t1c\tpr(1c)\t2c\tpr(2c)\t3c\tpr(3c)\t4c\tpr(4c)\t5c\tpr(5c)")
            l = 0
            for top1, top5, prec, rec, f1, examples, top5_classes, top5_probs \
                    in zip(ns.perclass_top1, ns.perclass_top5, ns.perclass_precision, ns.perclass_recall,
                           ns.perclass_f1, ns.perclass_examples, ns.top5_classes_per_class, ns.top5_probs_per_class):
                if skip_no_example_test and examples == 0:
                    pass
                else:
                    model_csv.write('\n"{}"\t{}\t{}\t{}\t{}\t{}\t{}\t{}'
                                    .format(ns.testset.labelIntToStr(l), l, top1, top5, prec, rec, f1, examples))
                    for tfc, tfp in zip(top5_classes, top5_probs):
                        model_csv.write("\t{}\t{}".format(tfc, tfp))
                l += 1
            model_csv.close()



def save_model_json(self, out_file):
    # type: (keras.models.Model, str) -> None
    json = self.model.to_json()
    fjson = file(out_file, "w")
    fjson.write(json)
    fjson.close()
