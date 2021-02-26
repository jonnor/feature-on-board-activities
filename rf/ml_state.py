import os
import numpy as np

import utils

PAMAP2 = "/home/atis/work/feature-group-selection/datasets/PAMAP2"
#PAMAP2 = "/home/atis/work/on-board/PAMAP2_Dataset/"


NUM_CLASSES = 12

FEATURE_NAMES_GREEDY_OLD = [
    ['tTotalAcc-energy()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-energy()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-energy()', 'tTotalAcc-q75()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-energy()', 'tTotalAcc-q75()', 'tTotalAccL1Norm-iqr()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-energy()', 'tTotalAcc-q75()', 'tTotalAccL1Norm-iqr()', 'tTotalAcc-q25()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-energy()', 'tTotalAcc-q75()', 'tTotalAccL1Norm-iqr()', 'tTotalAcc-q25()', 'tTotalAccJerkMagSq-iqr()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-energy()', 'tTotalAcc-q75()', 'tTotalAccL1Norm-iqr()', 'tTotalAcc-q25()', 'tTotalAccJerkMagSq-iqr()', 'tTotalAccL1Norm-energy()']
]

FEATURE_NAMES_OLD = ['tTotalAcc-energy()', 'tTotalAccJerk-energy()', 'tTotalAcc-q75()', 'tTotalAccL1Norm-iqr()', 'tTotalAcc-q25()', 'tTotalAccJerkMagSq-iqr()']

# for the new partitioning

FEATURE_NAMES_GREEDY_NEW = [
    ['tTotalAcc-energy()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-entropy()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-entropy()', 'tTotalAccJerkMagSq-iqr()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-entropy()', 'tTotalAccJerkMagSq-iqr()', 'tTotalAcc-max()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-entropy()', 'tTotalAccJerkMagSq-iqr()', 'tTotalAcc-max()', 'tTotalAccL1Norm-iqr()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-entropy()', 'tTotalAccJerkMagSq-iqr()', 'tTotalAcc-max()', 'tTotalAccL1Norm-iqr()', 'tTotalAccJerkL1Norm-iqr()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-entropy()', 'tTotalAccJerkMagSq-iqr()', 'tTotalAcc-max()', 'tTotalAccL1Norm-iqr()', 'tTotalAccJerkL1Norm-iqr()', 'tTotalAccL1Norm-mean()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-entropy()', 'tTotalAccJerkMagSq-iqr()', 'tTotalAcc-max()', 'tTotalAccL1Norm-iqr()', 'tTotalAccJerkL1Norm-iqr()', 'tTotalAccL1Norm-mean()', 'tTotalAccJerkL1Norm-mean()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-entropy()', 'tTotalAccJerkMagSq-iqr()', 'tTotalAcc-max()', 'tTotalAccL1Norm-iqr()', 'tTotalAccJerkL1Norm-iqr()', 'tTotalAccL1Norm-mean()', 'tTotalAccJerkL1Norm-mean()', 'tTotalAccL1Norm-max()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-entropy()', 'tTotalAccJerkMagSq-iqr()', 'tTotalAcc-max()', 'tTotalAccL1Norm-iqr()', 'tTotalAccJerkL1Norm-iqr()', 'tTotalAccL1Norm-mean()', 'tTotalAccJerkL1Norm-mean()', 'tTotalAccL1Norm-max()', 'tTotalAccJerkL1Norm-median()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-entropy()', 'tTotalAccJerkMagSq-iqr()', 'tTotalAcc-max()', 'tTotalAccL1Norm-iqr()', 'tTotalAccJerkL1Norm-iqr()', 'tTotalAccL1Norm-mean()', 'tTotalAccJerkL1Norm-mean()', 'tTotalAccL1Norm-max()', 'tTotalAccJerkL1Norm-median()', 'tTotalAccJerkL1Norm-std()'],
    ['tTotalAcc-energy()', 'tTotalAccJerk-entropy()', 'tTotalAccJerkMagSq-iqr()', 'tTotalAcc-max()', 'tTotalAccL1Norm-iqr()', 'tTotalAccJerkL1Norm-iqr()', 'tTotalAccL1Norm-mean()', 'tTotalAccJerkL1Norm-mean()', 'tTotalAccL1Norm-max()', 'tTotalAccJerkL1Norm-median()', 'tTotalAccJerkL1Norm-std()', 'tTotalAcc-mean()']
]

FEATURE_NAMES_NEW = ['tTotalAcc-energy()', 'tTotalAccJerk-entropy()', 'tTotalAccJerkMagSq-iqr()', 'tTotalAcc-max()', 'tTotalAccL1Norm-iqr()', 'tTotalAccJerkL1Norm-iqr()', 'tTotalAccL1Norm-mean()', 'tTotalAccJerkL1Norm-mean()', 'tTotalAccL1Norm-max()']

FEATURE_NAMES_NEW = ['tTotalAcc-energy()', 'tTotalAccJerk-entropy()', 'tTotalAccJerkMagSq-iqr()', 'tTotalAcc-max()', 'tTotalAccJerk-energy()', 'tTotalAccL1Norm-iqr()', 'tTotalAccMagSq-iqr()', 'tTotalAccJerk-q75()', 'tTotalAccJerk-mean()', 'tTotalAccJerkL1Norm-entropy()']


FEATURE_NAMES_GREEDY = FEATURE_NAMES_GREEDY_OLD
FEATURE_NAMES = FEATURE_NAMES_OLD

class State:
    def load_subset(self, name):
        filename = os.path.join(PAMAP2, name, "features.csv")
        data = np.asarray(utils.load_csv(filename, skiprows=1))
        filename = os.path.join(PAMAP2, name, "y_{}.txt".format(name))
        activities = np.asarray(utils.load_csv(filename)).ravel()
        return data, activities

    def load(self):
        self.train, self.train_y = self.load_subset("train")
        self.validation, self.validation_y = self.load_subset("validation")
        self.test, self.test_y = self.load_subset("test")

        filename = os.path.join("..", "feature_names.csv")
        self.names = utils.read_list_of_features(filename)
        self.simple_names = [u[1] for u in self.names]

        # need to preserve order, so cannot uniquify via the usual way (via a set)
        self.groups = []
        for n in self.names:
            if n[2] not in self.groups:
                self.groups.append(n[2])

        self.num_classes = NUM_CLASSES
        self.num_features = len(self.groups) # number of features
        self.feature_names = FEATURE_NAMES
        self.selector = None

    def get_test_features(self):
        return self.test[:,self.selector]
