#!/usr/bin/python3

import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.tree import _tree

# import the local copy of `sklearn_porter`
import sys

from ml_state import State
import utils

INDENT = 4
#OUTPUT_FILE = "/home/atis/work/pull/contiki-ng/examples/on-board-rf/exported-rf.c"
OUTPUT_FILE = "./exported-rf.c"

# TODO: run for the different options of trees, using a grid-search
NUM_TREES = 50

class RfState(State):
    def train_rf(self, indexes):
        self.selector = utils.select(self.names, self.groups, indexes, False)
        print("num_features=", len(self.selector))

        # simply train and then evaluate
        features_train = self.train[:,self.selector]
        features_validation = self.validation[:,self.selector]
        features_test = self.test[:,self.selector]
      
        n_iter = 100

        import scipy.stats
        # Spaces to search for hyperparameters
        parameter_distributions = {
            #'min_samples_leaf': scipy.stats.loguniform(0.0000001, 0.001),
            #'max_depth': scipy.stats.randint(5, 20),
            'max_depth': scipy.stats.randint(5, 30),
        }

        # use balanced weigths to account for class imbalance
        # (we're trying to optimize f1 score, not accuracy)
        base = RandomForestClassifier(n_estimators=NUM_TREES,
                                     random_state = 0, n_jobs=1,
                                     class_weight = "balanced")

        from emlearn.evaluate.trees import model_size_bytes
        from sklearn.model_selection import RandomizedSearchCV
        f1_micro = make_scorer(f1_score, average="micro")


        search = RandomizedSearchCV(
            base,
            param_distributions=parameter_distributions,
            scoring={
                # our predictive model metric
                'f1_micro': f1_micro,
                # metrics for the model costs
                #'size': model_size_bytes,
            },
            refit='f1_micro',
            n_iter=n_iter,
            cv=5,
            return_train_score=True,
            n_jobs=4,
            verbose=1,
        )


        search.fit(features_train, self.train_y)
        clf = search.best_estimator_

        import pandas
        results = pandas.DataFrame(search.cv_results_)
        results.to_parquet('hyperparam_results.parquet')

        # check the results on the validation set
        hypothesis = clf.predict(features_validation)
        validation_score = f1_score(self.validation_y, hypothesis, average="micro")

        # check also the results on the test set
        hypothesis = clf.predict(features_test)
        test_score = f1_score(self.test_y, hypothesis, average="micro")

        print("validation={:.2f} test={:.2f}".format(validation_score, test_score))

#        # Export as dot file
#       tree = clf.estimators_[0]
#        export_graphviz(tree, out_file='tree.dot')
#                        feature_names = iris.feature_names,
#                        class_names = iris.target_names,
#                        rounded = True, proportion = False, 
#                        precision = 2, filled = True)

#        porter = Porter(clf, language='c')
#        print("C:")
#        integrity = porter.integrity_score(features_train[:100])

#        y = clf.predict(features_train[:100])
#        print("python:")
#        for yy in y:
#            print(yy)
#        print("integrity=", integrity)
#        output = porter.export(embed_data=True)
#        with open("rf-export.c", "w") as f:
#            f.write(output)

        return clf, validation_score, test_score

    def export(self, clf):
        features = self.test[:,self.selector]

        num_data = len(features)
        num_classes = self.num_classes
        num_trees = len(clf.estimators_)
        num_features = len(self.selector)

        output = ""
        output += '#include <zephyr.h>\n'
        output += "\n"
        output += "#define NUM_CLASSES {}\n".format(num_classes)
        output += "#define NUM_FEATURES {}\n".format(num_features)
        output += "#define NUM_DATA {}\n".format(num_data)
        output += "#define NUM_TREES {}\n".format(num_trees)
        output += "\n"
        output += "const float data[NUM_DATA][NUM_FEATURES] = {\n"
        for element in features:
            lst = []
            for feature in element:
                lst.append("{:.6f}".format(feature))
            output += get_indent(1) + "{" + ", ".join(lst) + "},\n"
        output += "};\n"
        output += "\n"
        
        for idx, estimator in enumerate(clf.estimators_):
            output += "int predict_{}(const float features[])\n".format(idx)
            output += "{\n"
            output += get_indent(1) + "int classification = -1;\n"
            output += "\n"
            output += export_tree(estimator)
            output += get_indent(1) + "return classification;\n"
            output += "}\n"
            output += "\n"

        output += "int rf_classify_single(const float features[])\n"
        output += "{\n"
        output += get_indent(1) + "int i;\n"
        output += get_indent(1) + "int best_class = -1;\n"
        output += get_indent(1) + "int classes[NUM_CLASSES] = {0};\n"
        for i in range(num_trees):
            output += get_indent(1) + "classes[predict_{}(features)]++;\n".format(i)
        output += get_indent(1) + "for (i = 0; i < NUM_CLASSES; ++i) {\n"
        output += get_indent(2) + "if (best_class == -1 || classes[i] < classes[best_class]) {\n"
        output += get_indent(3) + "best_class = i;\n"
        output += get_indent(2) + "}\n"
        output += get_indent(1) + "}\n"
        output += get_indent(1) + "return best_class;\n"
        output += "}\n"
        output += "\n"

        output += "int rf_classify(void)\n"
        output += "{\n"
        output += get_indent(1) + "int i;\n"
        output += get_indent(1) + "int dummy = 0;\n"
        output += get_indent(1) + "for (i = 0; i < NUM_DATA; ++i) {\n"
        output += get_indent(2) + "dummy += rf_classify_single(data[i]);\n"
        output += get_indent(1) + "}\n"
        output += get_indent(1) + "return dummy;\n"
        output += "}\n"
        output += "\n"

        return output


###########################################

def get_indent(depth):
    return (" " * INDENT) * depth

def export_tree(decision_tree, decimals=6):
    """Build a text report showing the rules of a decision tree.

    Note that backwards compatibility may not be supported.

    Parameters
    ----------
    decision_tree : object
        The decision tree estimator to be exported.
        It can be an instance of
        DecisionTreeClassifier or DecisionTreeRegressor.

    decimals : int, optional (default=6)
        Number of decimal digits to display. If equal to or less than 0, printed as integers

    Returns
    -------
    report : string
        Text summary of all the rules in the decision tree.

    """
    tree_ = decision_tree.tree_

    right_child_fmt = "if ({} <= {}) {}\n"
    left_child_fmt = "} else {\n"
    feature_names_ = ["features[{}]".format(i) for i in tree_.feature]

    export_tree.report = ""

    def print_tree_recurse(node, depth):
        indent = get_indent(depth)

        info_fmt = ""
        info_fmt_left = info_fmt
        info_fmt_right = info_fmt

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names_[node]
            threshold = tree_.threshold[node]
            if decimals > 0:
                threshold = "{1:.{0}f}".format(decimals, threshold)
            else:
                threshold = "{}".format(int(round(threshold)))
            export_tree.report += indent
            export_tree.report += right_child_fmt.format(name,
                                                         threshold, "{")
            export_tree.report += info_fmt_left
            print_tree_recurse(tree_.children_left[node], depth+1)

            export_tree.report += indent
            export_tree.report += left_child_fmt

            export_tree.report += info_fmt_right
            print_tree_recurse(tree_.children_right[node], depth+1)

            export_tree.report += indent
            export_tree.report += "}\n"
        else:  # leaf
            export_tree.report += indent
            classification = np.argmax(tree_.value[node])
            val = "classification = {};\n".format(int(classification))
            export_tree.report += val

    print_tree_recurse(0, 1)
    return export_tree.report


###########################################

def main():
    print("Main...")
    s = RfState()
    print("Loading...")
    s.load()

    print("Training RF...")
    if 1:
        feature_indexes = utils.get_indexes_by_names(s.groups, s.feature_names)
    else:
        feature_indexes = range(len(s.groups))
    clf, _, _ = s.train_rf(feature_indexes)

    with open(OUTPUT_FILE, "w") as f:
        f.write(s.export(clf))

###########################################

if __name__ == '__main__':
    main()
    print("all done!")

###########################################
