import sys
import os
import re
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from subprocess import check_call

import torch
import torch.nn as nn


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']


# turn off the word2id - define a named function here to allow for pickling
def return_unk():
    return UNK


def get_length(x):
    return x.shape[1] - (np.sum(x, axis=-1) == 0).sum(1)


class MOSI:
    def __init__(self, config):
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = None, None

        except:
            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

            # load pickle file for unaligned acoustic and visual source
            pickle_filename = '../datasets/MOSI/mosi_data_noalign.pkl'
            csv_filename = '../datasets/MOSI/MOSI-label.csv'

            with open(pickle_filename, 'rb') as f:
                d = pickle.load(f)

            # read csv file for label and text
            df = pd.read_csv(csv_filename)
            text = df['text']
            vid = df['video_id']
            cid = df['clip_id']

            train_split_noalign = d['train']
            dev_split_noalign = d['valid']
            test_split_noalign = d['test']

            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            # define a regular expression to extract the video ID out of the keys
            pattern = re.compile('(.*)_(.*)')
            num_drop = 0  # a counter to count how many data points went into some processing issues

            if True:
                v = np.concatenate(
                    (train_split_noalign['vision'], dev_split_noalign['vision'], test_split_noalign['vision']), axis=0)
                vlens = get_length(v)

                a = np.concatenate(
                    (train_split_noalign['audio'], dev_split_noalign['audio'], test_split_noalign['audio']), axis=0)
                alens = get_length(a)

                label = np.concatenate(
                    (train_split_noalign['labels'], dev_split_noalign['labels'], test_split_noalign['labels']), axis=0)

                L_V = v.shape[1]
                L_A = a.shape[1]

            all_id = np.concatenate((train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']),
                                    axis=0)[:, 0]
            all_id_list = list(map(lambda x: x.decode('utf-8'), all_id.tolist()))

            train_size = len(train_split_noalign['id'])
            dev_size = len(dev_split_noalign['id'])
            test_size = len(test_split_noalign['id'])

            dev_start = train_size
            test_start = train_size + dev_size

            all_csv_id = [(vid[i], str(cid[i])) for i in range(len(vid))]

            for i, idd in enumerate(all_id_list):
                # get the video ID and the features out of the aligned dataset
                idd1, idd2 = re.search(pattern, idd).group(1, 2)

                # matching process
                try:
                    index = all_csv_id.index((idd1, idd2))
                except:
                    exit()
                """
                    Retrive noalign data from pickle file 
                """
                _words = text[index].split()
                _label = label[i].astype(np.float32)
                _visual = v[i]
                _acoustic = a[i]
                _vlen = vlens[i]
                _alen = alens[i]
                _id = all_id[i]

                # remove nan values
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                # remove speech pause tokens - this is in general helpful
                # we should remove speech pauses and corresponding visual/acoustic features together
                # otherwise modalities would no longer be aligned
                actual_words = []
                words = []
                visual = []
                acoustic = []

                # For non-align setting
                # we also need to record sequence lengths
                """TODO: Add length counting for other datasets 
                """
                for word in _words:
                    actual_words.append(word)

                visual = _visual[L_V - _vlen:, :]
                acoustic = _acoustic[L_A - _alen:, :]

                # z-normalization per instance and remove nan/infs
                # visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
                # acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))
                if i < dev_start:
                    train.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= dev_start and i < test_start:
                    dev.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= test_start:
                    test.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                else:
                    print(f"Found video that doesn't belong to any splits: {idd}")

            print(f"Total number of {num_drop} datapoints have been dropped.")
            print("Dataset split")
            print("Train Set: {}".format(len(train)))
            print("Validation Set: {}".format(len(dev)))
            print("Test Set: {}".format(len(test)))
            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            # self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
            # torch.save((pretrained_emb, word2id), CACHE_PATH)

            # Save pickles
            to_pickle(train, DATA_PATH + '/train.pkl')
            to_pickle(dev, DATA_PATH + '/dev.pkl')
            to_pickle(test, DATA_PATH + '/test.pkl')

    def get_data(self, mode):
        if mode == "train":
            return self.train, self.word2id, None
        elif mode == "valid":
            return self.dev, self.word2id, None
        elif mode == "test":
            return self.test, self.word2id, None
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()


class MOSEI:
    def __init__(self, config):

        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = None, None

        except:
            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

            # first we align to words with averaging, collapse_function receives a list of functions
            # dataset.align(text_field, collapse_functions=[avg])
            # load pickle file for unaligned acoustic and visual source
            pickle_filename = '../datasets/MOSEI/mosei_senti_data_noalign.pkl'
            csv_filename = '../datasets/MOSEI/MOSEI-label.csv'

            with open(pickle_filename, 'rb') as f:
                d = pickle.load(f)

            # read csv file for label and text
            df = pd.read_csv(csv_filename)
            text = df['text']
            vid = df['video_id']
            cid = df['clip_id']

            train_split_noalign = d['train']
            dev_split_noalign = d['valid']
            test_split_noalign = d['test']

            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            # define a regular expression to extract the video ID out of the keys
            # pattern = re.compile('(.*)\[.*\]')
            pattern = re.compile('(.*)_([.*])')
            num_drop = 0  # a counter to count how many data points went into some processing issues

            v = np.concatenate(
                (train_split_noalign['vision'], dev_split_noalign['vision'], test_split_noalign['vision']), axis=0)
            vlens = get_length(v)

            a = np.concatenate((train_split_noalign['audio'], dev_split_noalign['audio'], test_split_noalign['audio']),
                               axis=0)
            alens = get_length(a)

            label = np.concatenate(
                (train_split_noalign['labels'], dev_split_noalign['labels'], test_split_noalign['labels']), axis=0)

            L_V = v.shape[1]
            L_A = a.shape[1]

            all_id = np.concatenate((train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']),
                                    axis=0)[:, 0]
            all_id_list = all_id.tolist()

            train_size = len(train_split_noalign['id'])
            dev_size = len(dev_split_noalign['id'])
            test_size = len(test_split_noalign['id'])

            dev_start = train_size
            test_start = train_size + dev_size

            all_csv_id = [(vid[i], str(cid[i])) for i in range(len(vid))]

            for i, idd in enumerate(all_id_list):
                # get the video ID and the features out of the aligned dataset

                # matching process
                try:
                    index = i
                except:
                    import ipdb;
                    ipdb.set_trace()

                _words = text[index].split()
                _label = label[i].astype(np.float32)
                _visual = v[i]
                _acoustic = a[i]
                _vlen = vlens[i]
                _alen = alens[i]
                _id = '{}[{}]'.format(all_csv_id[0], all_csv_id[1])

                # remove nan values
                # label = np.nan_to_num(label)
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                # remove speech pause tokens - this is in general helpful
                # we should remove speech pauses and corresponding visual/acoustic features together
                # otherwise modalities would no longer be aligned
                actual_words = []
                words = []
                visual = []
                acoustic = []

                for word in _words:
                    actual_words.append(word)

                visual = _visual[L_V - _vlen:, :]
                acoustic = _acoustic[L_A - _alen:, :]

                if i < dev_start:
                    train.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= dev_start and i < test_start:
                    dev.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                elif i >= test_start:
                    test.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                else:
                    print(f"Found video that doesn't belong to any splits: {idd}")

            # print(f"Total number of {num_drop} datapoints have been dropped.")
            print(f"Total number of {num_drop} datapoints have been dropped.")
            print("Dataset split")
            print("Train Set: {}".format(len(train)))
            print("Validation Set: {}".format(len(dev)))
            print("Test Set: {}".format(len(test)))
            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            # self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
            # torch.save((pretrained_emb, word2id), CACHE_PATH)
            self.pretrained_emb = None

            # Save pickles
            to_pickle(train, DATA_PATH + '/train.pkl')
            to_pickle(dev, DATA_PATH + '/dev.pkl')
            to_pickle(test, DATA_PATH + '/test.pkl')

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "valid":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()

class SIMS:
    def __init__(self, config):

        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = None, None
        except:
            self.pretrained_emb= None
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

            # first we align to words with averaging, collapse_function receives a list of functions
            # dataset.align(text_field, collapse_functions=[avg])
            # load pickle file for unaligned acoustic and visual source
            pickle_filename = '../datasets/SIMS/unaligned_39.pkl'
            csv_filename = '../datasets/SIMS/label.csv'

            with open(pickle_filename, 'rb') as f:
                d = pickle.load(f)

                # read csv file for label and text
                df = pd.read_csv(csv_filename)
                text = df['text']
                vid = df['video_id']
                cid = df['clip_id']

                label_T, label_A, label_V = np.array(df["label_T"].values).reshape(-1,1),np.array(df["label_A"].values).reshape(-1,1),np.array(df["label_V"].values).reshape(-1,1)

                train_split_noalign = d['train']
                dev_split_noalign = d['valid']
                test_split_noalign = d['test']

                # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
                EPS = 1e-6

                # place holders for the final train/dev/test dataset
                self.train = train = []
                self.dev = dev = []
                self.test = test = []
                self.word2id = word2id

                # define a regular expression to extract the video ID out of the keys
                pattern = re.compile('(.*?)\$_\$(.*)')
                num_drop = 0  # a counter to count how many data points went into some processing issues

                if True:
                    v = np.concatenate(
                        (train_split_noalign['vision'], dev_split_noalign['vision'], test_split_noalign['vision']),
                        axis=0)
                    vlens = get_length(v)

                    a = np.concatenate(
                        (train_split_noalign['audio'], dev_split_noalign['audio'], test_split_noalign['audio']), axis=0)
                    alens = get_length(a)

                    label = np.concatenate(
                        (train_split_noalign['regression_labels'], dev_split_noalign['regression_labels'], test_split_noalign['regression_labels']),
                        axis=0)[:,np.newaxis]


                    L_V = v.shape[1]
                    L_A = a.shape[1]

                all_id = np.concatenate((train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']),axis=0)
                all_id_list = all_id.tolist()

                train_size = len(train_split_noalign['id'])
                dev_size = len(dev_split_noalign['id'])
                test_size = len(test_split_noalign['id'])

                dev_start = train_size
                test_start = train_size + dev_size

                all_csv_id = [(vid[i], str(cid[i])) for i in range(len(vid))]

                for i, idd in enumerate(all_id_list):
                    # get the video ID and the features out of the aligned dataset
                    idd1, idd2 = re.search(pattern, idd).group(1, 2)

                    # matching process
                    try:
                        index = all_csv_id.index((idd1, str(int(idd2))))
                    except:
                        exit()
                    """
                        Retrive noalign data from pickle file 
                    """
                    _words = text[index].split()
                    _label = label[i].astype(np.float32)
                    _label_T = label_T[index].astype(np.float32)
                    _label_A = label_A[index].astype(np.float32)
                    _label_V = label_V[index].astype(np.float32)
                    _visual = v[i]
                    _acoustic = a[i]
                    _vlen = vlens[i]
                    _alen = alens[i]
                    _id = all_id[i]

                    # remove nan values
                    _visual = np.nan_to_num(_visual)
                    _acoustic = np.nan_to_num(_acoustic)

                    # remove speech pause tokens - this is in general helpful
                    # we should remove speech pauses and corresponding visual/acoustic features together
                    # otherwise modalities would no longer be aligned
                    actual_words = []
                    words = []
                    visual = []
                    acoustic = []

                    # For non-align setting
                    # we also need to record sequence lengths
                    """TODO: Add length counting for other datasets 
                    """
                    for word in _words:
                        actual_words.append(word)

                    visual = _visual[L_V - _vlen:, :]
                    acoustic = _acoustic[L_A - _alen:, :]

                    # z-normalization per instance and remove nan/infs
                    # visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
                    # acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))
                    if i < dev_start:
                        train.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label,_label_T,_label_A,_label_V, idd))
                    elif i >= dev_start and i < test_start:
                        dev.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label,_label_T,_label_A,_label_V, idd))
                    elif i >= test_start:
                        test.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label,_label_T,_label_A,_label_V, idd))
                    else:
                        print(f"Found video that doesn't belong to any splits: {idd}")

                print(f"Total number of {num_drop} datapoints have been dropped.")
                print("Dataset split")
                print("Train Set: {}".format(len(train)))
                print("Validation Set: {}".format(len(dev)))
                print("Test Set: {}".format(len(test)))
                word2id.default_factory = return_unk

                # Save glove embeddings cache too
                # self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
                # torch.save((pretrained_emb, word2id), CACHE_PATH)

                # Save pickles
                to_pickle(train, DATA_PATH + '/train.pkl')
                to_pickle(dev, DATA_PATH + '/dev.pkl')
                to_pickle(test, DATA_PATH + '/test.pkl')


    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "valid":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()


class UR_FUNNY:
    def __init__(self, config):
        if config.sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(config.sdk_dir))

        DATA_PATH = str(config.dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = None, None

        except:
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

            data_folds = load_pickle(DATA_PATH + '/data_folds.pkl')
            train_split = data_folds['train']
            dev_split = data_folds['dev']
            test_split = data_folds['test']
            word_aligned_openface_sdk = load_pickle(DATA_PATH + "/openface_features_sdk.pkl")
            word_aligned_covarep_sdk = load_pickle(DATA_PATH + "/covarep_features_sdk.pkl")
            word_embedding_idx_sdk = load_pickle(DATA_PATH + "/language_sdk.pkl")
            word_list_sdk = load_pickle(DATA_PATH + "/word_embedding_list.pkl")
            humor_label_sdk = load_pickle(DATA_PATH + '/humor_label_sdk.pkl')

            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            num_drop = 0  # a counter to count how many data points went into some processing issues

            for key in humor_label_sdk.keys():
                label = np.array(humor_label_sdk[key], dtype=int)
                _word_id = np.array(word_embedding_idx_sdk[key]['punchline_embedding_indexes'])
                _acoustic = np.array(word_aligned_covarep_sdk[key]['punchline_features'])
                _visual = np.array(word_aligned_openface_sdk[key]['punchline_features'])

                if not _word_id.shape[0] == _acoustic.shape[0] == _visual.shape[0]:
                    num_drop += 1
                    continue

                # remove nan values
                label = np.array([np.nan_to_num(label)])[:, np.newaxis]
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                actual_words = []
                words = []
                visual = []
                acoustic = []
                for i, word in enumerate(_word_id):
                    words.append(word)
                    visual.append(_visual[i, :])
                    acoustic.append(_acoustic[i, :])

                words = np.asarray(words)
                visual = np.asarray(visual)
                acoustic = np.asarray(acoustic)
                actual_words=np.array(word_embedding_idx_sdk[key]["punchline_features"])

                visual = np.nan_to_num(
                    (visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
                acoustic = np.nan_to_num(
                    (acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

                if key in train_split:
                    train.append(((words, visual, acoustic,actual_words,len(visual),len(acoustic)), label, key))
                elif key in dev_split:
                    dev.append(((words, visual, acoustic,actual_words,len(visual),len(acoustic)), label, key))
                elif key in test_split:
                    test.append(((words, visual, acoustic,actual_words,len(visual),len(acoustic)), label, key))#test.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
                else:
                    print(f"Found video that doesn't belong to any splits: {key}")


            to_pickle(train, DATA_PATH + '/train.pkl')
            to_pickle(dev, DATA_PATH + '/dev.pkl')
            to_pickle(test, DATA_PATH + '/test.pkl')

            print(f"# of Train {len(train)}")
            print(f"# of dev {len(dev)}")
            print(f"# of test {len(test)}")
            print(f"Total number of {num_drop} datapoints have been dropped.")

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, None
        elif mode == "valid":
            return self.dev, self.word2id,None
        elif mode == "test":
            return self.test, self.word2id, None
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()

