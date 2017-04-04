import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import imdataset
from imdataset import ImageDataset

class NetScore():
    def __init__(self, testset, predictions):
        # type: ( ImageDataset, list(int)) -> None

        self.testset = testset
        self.predictions = predictions
        self.__compute()

    def __compute(self):
        testset = self.testset
        predictions  = self.predictions
        top_labels = []
        global_top_5 = 0
        global_top_1 = 0
        per_class_top_5 = np.zeros([testset.labelsize, ], dtype=np.uint16)
        per_class_top_1 = np.zeros([testset.labelsize, ], dtype=np.uint16)
        per_class_counter = np.zeros([testset.labelsize, ], dtype=np.uint16)
        at_least_one_in_class = np.zeros([testset.labelsize, ])

        top5_args_per_example = np.zeros([0, 5], dtype=np.uint16)
        top5_probs_per_example = np.zeros([0, 5])
        true_class_per_example = []

        for i, prediction in enumerate(predictions):
            true_class = testset.getLabelInt(i)

            top5_args = prediction.argsort()[-5:][::-1]
            top5_probs = prediction[top5_args]
            top5_args_per_example = np.vstack((top5_args_per_example, top5_args))
            top5_probs_per_example = np.vstack((top5_probs_per_example, top5_probs))
            true_class_per_example.append(true_class)

            arg_max = np.argmax(prediction)

            per_class_counter[true_class] += 1
            at_least_one_in_class[true_class] = 1
            if true_class in top5_args:
                per_class_top_5[true_class] += 1
                global_top_5 += 1
            if arg_max == true_class:
                per_class_top_1[true_class] += 1
                global_top_1 += 1
            top_labels.append(arg_max)

        for l, counter in enumerate(per_class_counter):
            if counter > 0:
                per_class_top_1[l] /= counter
                per_class_top_5[l] /= counter

        global_top_5 = float(global_top_5) / len(predictions)
        global_top_1 = float(global_top_1) / len(predictions)

        all_labels = []
        for l in range(0, testset.labelsize):
            all_labels.append(l)
        per_class_precision = precision_score(testset.getLabelsInt(), top_labels, labels=all_labels, average=None)
        per_class_recall = recall_score(testset.getLabelsInt(), top_labels, labels=all_labels, average=None)
        per_class_f1 = f1_score(testset.getLabelsInt(), top_labels, labels=all_labels, average=None)

        avg_precision = np.average(per_class_precision)
        avg_recall = np.average(per_class_recall)
        avg_f1 = np.average(per_class_f1)

        weighted_avg_precision = precision_score(testset.getLabelsInt(), top_labels, average='weighted')
        weighted_avg_recall = recall_score(testset.getLabelsInt(), top_labels, average='weighted')
        weighted_avg_f1 = f1_score(testset.getLabelsInt(),  top_labels, average='weighted')


        # Weighted AVG of the per-class statistics
        self.global_top5 = global_top_5
        self.global_top1 = global_top_1
        self.global_precision = weighted_avg_precision
        self.global_recall = weighted_avg_recall
        self.global_f1 = weighted_avg_f1

        # AVG of the per-class statistics. NB: we exclude all the classes that aren't in the test set with at_least_one_in_class weight.
        self.avg_top5 = np.average(per_class_top_5, weights=at_least_one_in_class)
        self.avg_top1 = np.average(per_class_top_1, weights=at_least_one_in_class)
        self.avg_precision = np.average(per_class_precision, weights=at_least_one_in_class)
        self.avg_recall = np.average(per_class_recall, weights=at_least_one_in_class)
        self.avg_f1 = np.average(per_class_f1, weights=at_least_one_in_class)


        self.perclass_top5 = per_class_top_5
        self.perclass_top1 = per_class_top_1
        self.perclass_precision = per_class_precision
        self.perclass_recall = per_class_recall
        self.perclass_f1 = per_class_f1

        self.top_labels = top_labels
        self.perclass_examples = per_class_counter
        self.examples = int(np.sum(per_class_counter))

        self.top5_classes_per_example = top5_args_per_example
        self.top5_probs_per_example = top5_probs_per_example
        self.true_class_per_example = np.asarray(true_class_per_example)

    def print_global(self):
        print("\n\n")
        print("GLOBAL SCORES")
        print("top1 score: " + str(self.global_top1))
        print("top5 score: " + str(self.global_top5))
        print("precision:  " + str(self.global_precision))
        print("recall:     " + str(self.global_recall))
        print("f1 score :  " + str(self.global_f1))
        print("examples:   " + str(self.examples))
        print("\n\n\n")

    def print_perclass(self):
        print("PER-CLASS SCORES")
        for l, f1 in enumerate(self.perclass_f1):
            print("\nLabel {} - {}\nprecision={}\nrecall={}\nf1_score={}\nexamples={}"
                  .format(l, self.testset.labelnames[l], self.perclass_precision[l], self.perclass_recall[l], f1,
                          self.perclass_examples[l]))
