import glob
import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical


def create_windows(X, y, window, overlap):
    output_X = []
    output_y = []
    i = 0

    skipped = 0
    while True:
        t = []
        y_maj_check = []
        for j in range(0, window):
            t.append(X[[(i + j)], :])
            y_maj_check.append(np.argmax(y[(i + j)]))

        y_maj_check = np.array(y_maj_check)

        frac = np.count_nonzero(y_maj_check[y_maj_check == np.argmax(y[i + window])]) / y_maj_check.size
        if frac >= .66 and len(t) == window and y_maj_check[-1] != 0:
            output_X.append(t)
            output_y.append(y[i + window])
        else:
            skipped += 1
            if len(t) < window:
                print("NOT ADDING")
                print(len(t))
                print(frac)

        i += int(window - overlap)
        if i > (len(X) - window - 1):
            break
    print("The number of windows skipped as they did not meet the criteria", skipped)
    return np.squeeze(np.array(output_X)), np.array(output_y)


def create_dataset(raw_data='PAMAP2_Dataset/Protocol/', window=128, overlap=127, test_overlap=0, processed_dir='processed/'):
    print(raw_data)
    all_files = glob.glob(raw_data + "/*.dat")
    print(all_files)

    os.makedirs(processed_dir)
    for subject in all_files:
        print('creating file for', str(subject))
        df = pd.read_csv(subject, sep=' ', header=None)
        df.dropna(subset=[df.columns[7], df.columns[8], df.columns[9]], inplace=True)
        x_train = df[df.columns[[7, 8, 9]]].values
        y_train = df[df.columns[1]].values

        y_train = np.where(y_train == 12, 8, y_train)
        y_train = np.where(y_train == 13, 9, y_train)
        y_train = np.where(y_train == 16, 10, y_train)
        y_train = np.where(y_train == 17, 11, y_train)
        y_train = np.where(y_train == 24, 0, y_train)

        y_train = to_categorical(y_train, num_classes=12)
        x_train = x_train / 9.80665

        x_train_win, y_train_win = create_windows(x_train, y_train, window, overlap)
        sub_id = subject.split('/')[-1]
        np.save(processed_dir+'x_win_' + str(window) + '_' + str(overlap) + '_' + str(sub_id) + '.npy', x_train_win)
        np.save(processed_dir+'y_win_' + str(window) + '_' + str(overlap) + '_' + str(sub_id) + '.npy', y_train_win)

        if test_overlap != overlap:
            x_train_win, y_train_win = create_windows(x_train, y_train, window, test_overlap)
            sub_id = subject.split('/')[-1]
            np.save(processed_dir+'x_win_' + str(window) + '_' + str(test_overlap) + '_' + str(sub_id) + '.npy', x_train_win)
            np.save(processed_dir+'y_win_' + str(window) + '_' + str(test_overlap) + '_' + str(sub_id) + '.npy', y_train_win)


def load_pamap2(train_subjects=None, test_subjects=None, window=128, overlap=127, test_overlap=0, processed_dir='processed/'):
    for i, sub_id in enumerate(train_subjects):
        if i == 0:
            x_train = np.load(processed_dir+'x_win_' + str(window) + '_' + str(overlap) + '_' + str(sub_id) + '.dat.npy')
            y_train = np.load(processed_dir+'y_win_' + str(window) + '_' + str(overlap) + '_' + str(sub_id) + '.dat.npy')
        x_train = np.vstack(
            (x_train, np.load(processed_dir+'x_win_' + str(window) + '_' + str(overlap) + '_' + str(sub_id) + '.dat.npy')))
        y_train = np.vstack(
            (y_train, np.load(processed_dir+'y_win_' + str(window) + '_' + str(overlap) + '_' + str(sub_id) + '.dat.npy')))

    for i, sub_id in enumerate(test_subjects):
        if i == 0:
            x_test = np.load(processed_dir+'x_win_' + str(window) + '_' + str(test_overlap) + '_' + str(sub_id) + '.dat.npy')
            y_test = np.load(processed_dir+'y_win_' + str(window) + '_' + str(test_overlap) + '_' + str(sub_id) + '.dat.npy')
        x_test = np.vstack(
            (x_test, np.load(processed_dir+'x_win_' + str(window) + '_' + str(test_overlap) + '_' + str(sub_id) + '.dat.npy')))
        y_test = np.vstack(
            (y_test, np.load(processed_dir+'y_win_' + str(window) + '_' + str(test_overlap) + '_' + str(sub_id) + '.dat.npy')))

    return x_train, y_train, x_test, y_test
