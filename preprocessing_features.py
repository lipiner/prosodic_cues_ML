from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from os import path
from typing import Dict, Tuple
from collections import Counter

BASE_DIR = 'C:\\Uni\\Lab\\EA\\Yasmin'
# BASE_DIR = 'C:\\Users\\yasminln\\ML_Audio_EA'
# BASE_DIR = "/cs/usr/lipiner/www/LAB"
FEATURES_PREFIX = "features_"
FEATURES_DIR = 'features'
RESULTS_DIR = 'results'
OUTPUT_PATH = path.join(BASE_DIR, RESULTS_DIR)
LABEL_FILE = 'ratings/all_together.csv'
IGNORED_COLS_NUM = 6
SAMPLE_RATE = 2
VID_NAME_COL = "vid_name"
TIME_COL = "Time"
SUBJECT_COL = "subject"
LABEL_COL = "Scale-10"
MEAN_RATE = 5
# LABEL_COL = "Rating"
HUMAN_EA_VIDEOS = [
    "53_vid2.wav", "101_vid2.wav", "107_vid4.wav",
    "112_vid4.wav", "114_vid4.wav", "117_vid4.wav",
    "119_vid5.wav", "120_vid8.wav", "127_vid4.wav"
]
HUMAN_EA_SUBJECTS = [video.split("_")[0] for video in HUMAN_EA_VIDEOS]
TEST_SIZE = 0.2


def perform_pca(X, features_num=None):
    # # PCA to choose best features
    # pca_model = PCA(svd_solver='full')
    #
    # features_reduced = pca_model.fit_transform(X, y)
    # print("done first PCA")

    if np.shape(X)[1] == features_num:
        return X

    # PCA to choose best features
    pca_model = PCA(n_components=features_num)

    features_reduced = pca_model.fit_transform(X)
    print(X[0])
    return features_reduced


def split_within_sub_train_test(videos_features: Dict[str, pd.DataFrame], videos_labels: Dict[str, pd.DataFrame]) -> \
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Tuple[str, np.ndarray]]]:
    train_features = {}
    test_features = {}
    train_labels = {}
    test_labels = {}
    videos = list(videos_features.keys())
    np.random.shuffle(videos)
    for video in videos:
        subject = video.split("_")[0]
        features = videos_features[video].drop([TIME_COL, VID_NAME_COL, SUBJECT_COL], axis=1)
        labels = videos_labels[video][LABEL_COL]
        features, labels = (features.values, labels.values)
        if subject in test_features or (subject in HUMAN_EA_SUBJECTS and video not in HUMAN_EA_VIDEOS):
            if subject in train_features:
                # there is already data for this subjects -> appending the new data to previous one
                curr_features = train_features[subject]
                curr_labels = train_labels[subject]
                features, labels = append_features_labels(curr_features, curr_labels, features, labels)
            # insert the data to the relevant subject
            train_features[subject] = features
            train_labels[subject] = labels
        else:
            # no test data to this subject yet
            test_features[subject] = features
            test_labels[subject] = (video, labels)

    return train_features, test_features, train_labels, test_labels


def split_general_train_test(videos_features, videos_labels):
    train_features = []
    test_features = []
    train_labels = []
    test_labels = []
    human_ea_ref = {}

    # creates the train-test sets based on the videos from the human EA experiment
    # chooses val+test videos
    val_videos = []
    test_videos = []
    for video in videos_features:
        if video.split("_")[0] in HUMAN_EA_SUBJECTS:
            if video in HUMAN_EA_VIDEOS:
                test_videos.append(video)
            else:
                val_videos.append(video)
    test_videos_num = int((len(test_videos) + len(val_videos)) / 2)
    np.random.shuffle(val_videos)
    test_videos.extend(val_videos[:test_videos_num - len(test_videos)])
    val_videos = val_videos[test_videos_num - len(test_videos):]

    # creates train data
    for video in videos_features:
        if video.split("_")[0] in HUMAN_EA_SUBJECTS:
            continue
        features = videos_features[video].drop([TIME_COL, VID_NAME_COL, SUBJECT_COL], axis=1)
        labels = videos_labels[video][LABEL_COL]
        # if the subject is not in the test data, adds it to the train data. Otherwise, drops this video
        train_features, train_labels = append_features_labels(train_features, train_labels, features, labels)

    test_index = 0
    # creates val/test data
    for i, video in enumerate(val_videos + test_videos):
        features = videos_features[video].drop([TIME_COL, VID_NAME_COL, SUBJECT_COL], axis=1)
        labels = videos_labels[video][LABEL_COL]
        test_features, test_labels = append_features_labels(test_features, test_labels, features, labels)
        # takes the indices of the current video in the sets
        human_ea_ref[video] = (len(test_features) - len(features), len(test_features))
        if i == len(val_videos):
            test_index = len(test_labels)

    # adds additional samples to the test set so it will be 20%
    test_remain_size = TEST_SIZE - len(test_labels) / (len(train_labels) + len(test_labels))
    if test_remain_size < 0:
        print("humans EA videos are more than 0.2 (%f)" % test_remain_size)
    else:
        X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=test_remain_size)
        train_features, train_labels = X_train, y_train
        test_features, test_labels = append_features_labels(test_features, test_labels, X_test, y_test)

    return train_features, test_features, train_labels, test_labels, human_ea_ref, test_index, test_videos


# def export_results(results: Dict[str, list], targets, feature_set, paradigm, model):
#     test_videos = results.keys()
#     ea_results = dict()
#     for key in test_videos:
#         ea_results[key] = max(np.correlate(targets[key], results[key], 'full'))  # todo: fix that
#     test_targets = {key.replace(".wav", "_target"): targets[key] for key in test_videos}
#     results = {key.replace(".wav", "_prediction"): results[key] for key in test_videos}
#
#     results.update(test_targets)
#     # create a transposed DataFrame because the videos are not in the same length (and then transpose back)
#     output_results = pd.DataFrame.from_dict(results, orient='index').transpose()
#     output_results.sort_index(axis=1, inplace=True)
#     output_filename = "_".join([paradigm, model, feature_set]) + '.csv'
#     output_results.to_csv(path.join(OUTPUT_PATH, "full_predictions", output_filename), index=False, sep=',')
#
#     return pd.DataFrame(ea_results, index=[[feature_set], [model]])


def append_features_labels(curr_features, curr_labels, new_features, new_labels):
    try:
        X, y = (new_features.values, new_labels.values)
    except AttributeError:
        X, y = (new_features, new_labels)

    if len(curr_features) == 0:
        return X, y
    else:
        return np.append(curr_features, X, axis=0), \
               np.append(curr_labels, y)


def get_videos_features_labels(feature_set, features_num=None, transform=(lambda x: x)) -> \
        Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Process the features and labels of each video
    :param feature_set: a tuple of the features set file name and the number of rows to skip.
    :param features_num: number of features. default=None.
    :param transform: a function to apply on the labels. Default: do nothing.
    :return: 2 dicts that contain the video name as key and the features or the labels as values.
    """
    labels = get_labels(transform)
    features = get_features(feature_set, features_num)

    video_features = {}
    video_labels = {}
    for key, group in features.groupby(VID_NAME_COL):
        video_features[key] = group.reset_index(drop=True)
    # checks that the meta data of each video is the same for the features and the labels
    same_meta = True
    for key, group in labels.groupby(VID_NAME_COL):
        try:
            vid_labels = group[:len(video_features[key])]  # get same number of rows as the features
            features_info = video_features[key][[VID_NAME_COL, TIME_COL]]
            label_info = vid_labels[[VID_NAME_COL, TIME_COL]]
            if not np.equal(features_info, label_info).all().all():
                print(key + ":", np.all(np.equal(features_info, label_info)))
                same_meta = False
            else:
                video_labels[key] = vid_labels.reset_index(drop=True)
        except ValueError:
            print(key + ":", np.shape(video_features[key][[VID_NAME_COL, TIME_COL]]),
                  np.shape(group[[VID_NAME_COL, TIME_COL]]))
            same_meta = False

    # checks that the features contains all the videos in the ratings file
    for key in list(video_features.keys()):
        if video_labels.get(key) is None:
            print("missing file in ratings:", key)
            same_meta = False
            del video_features[key]

    # if not same_meta:
    #     return exit(1)

    return video_features, video_labels


def get_features(feature_set: Tuple[str, int], features_num=None):
    """
    The given features set
    :param feature_set: a tuple of the features set file name and the number of rows to skip.
    :param features_num: number of features. default=None.
    :return: the features set.
    """
    all_features = np.loadtxt(path.join(BASE_DIR, FEATURES_DIR, FEATURES_PREFIX + feature_set[0]), delimiter=',',
                              skiprows=feature_set[1], usecols=range(2, feature_set[1] - IGNORED_COLS_NUM))
    # performing pca
    print("before pca:", np.shape(all_features))
    features_reduced = perform_pca(all_features, features_num)
    print("after pca:", np.shape(features_reduced))

    meta = pd.read_csv(path.join(BASE_DIR, FEATURES_DIR, FEATURES_PREFIX + feature_set[0]), delimiter=',',
                       names=[VID_NAME_COL, TIME_COL], skiprows=feature_set[1], usecols=(0, 1))
    meta[VID_NAME_COL] = meta[VID_NAME_COL].str.strip('\'')  # fix the video name format
    # adds the subject id column
    meta[SUBJECT_COL] = meta.apply(lambda row: str(row[VID_NAME_COL]).split('_')[0], axis=1)

    features = meta.join(pd.DataFrame(features_reduced))

    subjects_gender = pd.read_csv(path.join(BASE_DIR, FEATURES_DIR, "targets_gender.csv"), delimiter=',',
                                  index_col=0).to_dict()['gender']
    features["gender"] = meta.apply(lambda row: subjects_gender[int(row[SUBJECT_COL])], axis=1)

    return features


def get_labels(transform=(lambda x: x), label_col=LABEL_COL):
    """
    :return: The labels after fixing their sample rate
    """
    ratings = pd.read_csv(path.join(BASE_DIR, LABEL_FILE), delimiter=',')

    # separate the ratings to each video and average the ratings according to the sample rate
    averaged_ratings = pd.DataFrame()
    for name, group in ratings.groupby(VID_NAME_COL):
        temp = average_labels(group)
        temp[TIME_COL] = group[::4].reset_index()[TIME_COL]
        # change the name of the video to be consistent with the audio file name
        temp[VID_NAME_COL] = name.replace(".MP4", ".wav").replace(".mp4", ".wav")
        # # adds the subject id
        # temp[SUBJECT_COL] = name.split("_")[0]
        # append the current video to all the ratings
        averaged_ratings = averaged_ratings.append(temp, ignore_index=True)

    # round 1 digit after the point to be consistent with the features times
    averaged_ratings[TIME_COL] = averaged_ratings[TIME_COL].round(1)
    averaged_ratings[LABEL_COL] = transform(averaged_ratings[label_col]).astype(np.int)

    return averaged_ratings[[SUBJECT_COL, VID_NAME_COL, TIME_COL, LABEL_COL]]


def average_labels(df):
    return df.groupby(np.arange(len(df)) // (2 * SAMPLE_RATE)).mean()


def intensity_labels_transform(labels):
    return (labels - MEAN_RATE).abs()


def binary_labels_transform(labels):
    return np.sign(labels - MEAN_RATE)


def diff_labels_transform(labels):
    labels = labels.round()
    return np.append([0], labels[1:].values - labels[:-1].values)


def binary_intensity_labels_transform(labels):
    return ((labels - MEAN_RATE).abs() > 3).astype(np.int)
