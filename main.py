from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree, linear_model
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import export_graphviz
from os import path
from subprocess import call
import numpy as np
from pandas import DataFrame, ExcelWriter, concat, read_csv, read_excel
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pylab
from scipy.stats import pearsonr
from collections import Counter
from typing import Dict
from code_files.preprocessing_features import get_videos_features_labels, split_general_train_test, \
    split_within_sub_train_test, OUTPUT_PATH, IGNORED_COLS_NUM, intensity_labels_transform, binary_labels_transform, \
    BASE_DIR, FEATURES_DIR, FEATURES_PREFIX, HUMAN_EA_VIDEOS, diff_labels_transform, binary_intensity_labels_transform

FEATURES_FILES = [
    ('eGeMAPSv01a.csv', 96),
    ('ComParE_2016.csv', 6381)
]
FIGS_DIR = 'graphs'
HUMAN_EA_RESULTS = "general_learning_predictions"
WITHIN_SUBJECT_RESULTS = "within_subject_predictions"
GENERAL_PARADIGM_NAME = "General learning"
DEBUG = False
POL_MODEL_NAME = "Pol"
POL_CAT_MODEL_NAME = "_cat"
SVM_MODEL_NAME = "SVM"
RF_MODEL_NAME = "RF"
TITLE_SIZE = 32
LABEL_SIZE = 27
TICKS_SIZE = 19
WIN_WIDTH = 17
WIDE_WIN = 27
COLORS = ['gray', 'rosybrown', 'red', 'coral', 'gold', 'olive', 'lawngreen', 'seagreen', 'lightblue',
          'deepskyblue', 'steelblue', 'navy', 'blue', 'blueviolet', 'plum', 'magenta', 'brown', 'lightpink',
          'yellowgreen', 'black', 'aquamarine', 'sandybrown', 'mediumslateblue', 'deeppink', 'c']

ALL_MODELS_GENERAL_TRAIN = list()
ALL_MODELS_GENERAL_RESULTS = list()
ALL_MODELS_GENERAL_TEST = list()
ALL_MODELS_WITHIN_RESULTS = list()
ALL_MODELS_GENERAL_EA = list()
ALL_MODELS_GENERAL_EA_TEST = list()
ALL_MODELS_WITHIN_EA = list()
EA_GENERAL_RESULTS = list()
EA_WITHIN_RESULTS = list()
WITHIN_TEST_SIZE = list()
WITHIN_TEST_VIDS = list()
FULL_GENERAL_RESULTS = list()
FULL_WITHIN_RESULTS = list()
GENERAL_CHANCE = dict()
WITHIN_CHANCE = dict()


def get_ea_score(vector1, vector2):
    if len(np.unique(vector2)) == 1 or len(np.unique(vector1)) == 1:
        # The model is constant so correlation can't be computed (The covariance is 0)
        corr = 0
    else:
        corr = pearsonr(vector1, vector2)[0]
    return corr


def export_results(results: Dict[str, list], targets, feature_set, paradigm, model, subs_ea, models_ea, test_videos=None):
    if test_videos is None:
        test_videos = results.keys()
    ea_results_test = dict()
    ea_results_val = dict()
    for key in results.keys():
        ea_score = get_ea_score(targets[key], results[key])
        subs_ea[key][model] = ea_score
        if key in test_videos:
            ea_results_test[key] = ea_score
        else:
            ea_results_val[key] = ea_score
    models_ea["test"][model] = np.mean(list(ea_results_test.values()))
    if ea_results_val:
        models_ea["val"][model] = np.mean(list(ea_results_val.values()))

    test_targets = {key.replace(".wav", "_target"): targets[key] for key in test_videos}
    results = {key.replace(".wav", "_prediction"): results[key] for key in test_videos}

    results.update(test_targets)
    # create a transposed DataFrame because the videos are not in the same length (and then transpose back)
    output_results = DataFrame.from_dict(results, orient='index').transpose()
    output_results.sort_index(axis=1, inplace=True)
    output_filename = "_".join([paradigm, model, feature_set]) + '.csv'
    output_results.to_csv(path.join(OUTPUT_PATH, "full_predictions", output_filename), index=False, sep=',')

    ea_results_test.update(ea_results_val)
    df = DataFrame(ea_results_test, index=[feature_set])
    df.index = [[feature_set], [model]]
    return df


def export_general_ea_results(predictions, true_labels, human_ea_ref, feature_set, model, subs_scores, subs_ea, models_ea, test_videos):
    results = dict()
    targets = dict()
    for video in human_ea_ref:
        start, end = human_ea_ref[video]
        results[video] = predictions[start:end]
        targets[video] = true_labels[start:end]
        try:
            subs_scores[video][model] = accuracy_score(true_labels[start:end], predictions[start:end])
        except ValueError:
            subs_scores[video][model] = r2_score(true_labels[start:end], predictions[start:end])

    ea_results = export_results(results, targets, feature_set, HUMAN_EA_RESULTS, model, subs_ea, models_ea, test_videos)
    EA_GENERAL_RESULTS.append(ea_results)


def split_keys_for_plotting(results):
    if results is None:
        return results

    new_keys_results = dict()
    for old_key in results.keys():
        parts = old_key.split('_')
        if len(parts) > 4:
            new_key = '\n'.join(['_'.join(parts[i:i + 3]) for i in range(0, len(parts), 3)])
        else:
            new_key = old_key
        new_keys_results[new_key] = results[old_key]
    return new_keys_results


def plot_models_result(results, feature_set, paradigm, chance, y_label, error=None):
    # saving the results
    index_name = feature_set  # " ".join([paradigm, feature_set, "score"])
    results_df = DataFrame(results, index=[index_name])
    if GENERAL_PARADIGM_NAME in paradigm:
        if "EA" in y_label:
            ALL_MODELS_GENERAL_EA.append(results_df)
        else:
            ALL_MODELS_GENERAL_RESULTS.append(results_df)
    else:
        if "EA" in y_label:
            ALL_MODELS_WITHIN_EA.append(results_df)
        else:
            ALL_MODELS_WITHIN_RESULTS.append(results_df)

    results = split_keys_for_plotting(results)
    error = split_keys_for_plotting(error)
    # analyzing all other models with the best RandomForest Model
    final_results = {key: results[key] for key in results if RF_MODEL_NAME not in key and POL_MODEL_NAME not in key}

    plot_model_performances(results, paradigm, feature_set, chance, y_label, RF_MODEL_NAME, error=error,
                            final_results=final_results, width=WIDE_WIN)
    plot_model_performances(results, paradigm, feature_set, chance, y_label, POL_MODEL_NAME, error=error,
                            final_results=final_results)
    plot_model_performances(final_results, paradigm, feature_set, chance, y_label, error=error)
    return final_results.keys()


def plot_model_performances(results, paradigm, feature_set, chance, y_label, model_name="", error=None,
                            final_results=None, width=WIN_WIDTH):
    if len(model_name) == 0:
        model_name = "all models"
        model_results = results
    else:
        # analyzing specific model
        model_results = {key: results[key] for key in results if model_name in key}
    model_results_fixed_name = {key.replace(model_name + '_', ""): model_results[key] for key in model_results}
    print(model_results)
    if len(model_results) > 1:
        df = DataFrame.from_dict(model_results_fixed_name, orient='index')
        if error is not None:
            error = {key.replace(model_name + '_', ""): error[key] for key in error}
            try:
                error = [error[key] for key in df.index]
            except KeyError:
                print("Invalid error (STD) in", model_name, paradigm, feature_set)
                error = None

        title = y_label + " - " + paradigm + " - " + model_name + " - %s" % feature_set
        ax = df.plot(kind='bar', yerr=error, legend=False, rot=0)
        set_plt_params(ax, title, "Model", y_label, False, chance, width)

        if DEBUG:
            plt.show()
        else:
            plt.savefig(path.join(OUTPUT_PATH, FIGS_DIR, title + '.png'))
            plt.close()  # close the figure

    if len(model_results) != 0 and final_results is not None:
        # adding the best model to final results
        best = max(model_results, key=model_results.get)
        final_results[best] = results[best]


def set_plt_params(ax, title, x_label, y_label, legend=True, chance=None, width=WIN_WIDTH):
    ax.set_ylabel(y_label, fontsize=LABEL_SIZE)
    ax.set_xlabel(x_label, fontsize=LABEL_SIZE)
    if legend:
        plt.legend(fontsize=TICKS_SIZE)
    curr_bottom, curr_top = ax.get_ylim()
    plt.grid(axis='y')
    if isinstance(chance, int) or isinstance(chance, float):
        # title += " (chance level: %.2f)" % chance
        plt.axhline(y=chance)
        plt.text(plt.xlim()[0], chance, "chance level")
        chance_max = chance + 0.05
    elif isinstance(chance, dict):
        locs, labels = plt.xticks()
        for i, label in enumerate(labels):
            curr_chance = chance.get(label.get_text().replace("\n", "_"))
            plt.text(locs[i]-0.25, curr_chance, "-chance: %.2f-" % curr_chance)
        chance_max = max(chance.values()) + 0.05
    else:
        chance_max = curr_top
    plt.title(title, fontsize=TITLE_SIZE, y=1.03)
    # set y axis range
    plt.ylim(bottom=max(curr_bottom, -1), top=min(max(chance_max, curr_top), 1.2))
    if not DEBUG:
        plt.xticks(fontsize=TICKS_SIZE)
        plt.yticks(fontsize=TICKS_SIZE)
        fig = plt.gcf()
        fig.set_size_inches((width, 10), forward=False)


def perform_general_learning(videos_features, videos_labels, models, feature_set, mode=""):
    X_train, X_test, y_train, y_test, human_ea_ref, test_index, test_videos = split_general_train_test(videos_features, videos_labels)
    most_common_val = Counter(y_train).most_common(1)[0][0]
    val_chance = np.round(Counter(y_test[:test_index])[most_common_val] / len(y_test[:test_index]), 2)
    print("validation", len(y_test[:test_index]), "chance level:", val_chance, Counter(y_test[:test_index]))
    test_chance = np.round(Counter(y_test[test_index:])[most_common_val] / len(y_test[test_index:]), 2)
    print("test", len(y_test[test_index:]), "chance level:", test_chance, Counter(y_test[test_index:]))
    chance = np.round(Counter(y_test)[most_common_val] / len(y_test), 2)
    print("overall all test", len(y_test), "chance level:", chance, Counter(y_test))
    X_train, y_train = balance_set(X_train, y_train)
    # calculates chance levels for each video
    videos_chances = dict()
    for video in human_ea_ref:
        start, end = human_ea_ref[video]
        videos_chances[video] = np.round(Counter(y_test[start:end])[most_common_val] / len(y_test[start:end]), 2)
    videos_chances["overall"] = chance
    GENERAL_CHANCE["chance"] = videos_chances

    if mode != "":
        paradigm = " - ".join([GENERAL_PARADIGM_NAME, mode])
    else:
        paradigm = GENERAL_PARADIGM_NAME

    results = dict()
    test_results = dict()
    train_results = dict()
    subs_scores = {key: dict() for key in human_ea_ref}
    subs_ea = {key: dict() for key in human_ea_ref}
    models_ea = {"test": dict(), "val": dict()}
    for clf, name in models:
        try:
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            print(name)
            print(Counter(predictions))
            score = clf.score(X_test[:test_index], y_test[:test_index])
            results[name] = score
            test_results[name] = clf.score(X_test[test_index:], y_test[test_index:])
            train_results[name] = clf.score(X_train, y_train)
            export_general_ea_results(predictions, y_test, human_ea_ref, feature_set, name, subs_scores, subs_ea, models_ea, test_videos)
            if POL_MODEL_NAME in name:
                predictions = np.round(predictions)
                name += POL_CAT_MODEL_NAME
                results[name] = accuracy_score(y_test, predictions)
                train_results[name] = accuracy_score(y_train, np.round(clf.predict(X_train)))
                export_general_ea_results(predictions, y_test, human_ea_ref, feature_set, name, subs_scores, subs_ea, models_ea, test_videos)
        except ValueError as e:
            print("could not perform on model: %s, features: %s. Error: %s" % (name, feature_set, str(e)))
        except MemoryError:
            print("could not perform on model: %s, features: %s. Error: MemoryError" % (name, feature_set))

    subs_results_df = DataFrame.from_dict(subs_scores, orient='index').transpose()
    subs_results_df.index = [[feature_set] * len(subs_results_df.index), subs_results_df.index]
    FULL_GENERAL_RESULTS.append(subs_results_df)

    plot_RF_train_test(results, train_results, feature_set, mode)

    subs_scores = {key: subs_scores[key] for key in subs_scores.keys() if key in HUMAN_EA_VIDEOS}
    subs_ea = {key: subs_ea[key] for key in subs_ea.keys() if key in HUMAN_EA_VIDEOS}
    general_learning_plots(results, subs_scores, feature_set, paradigm, "Accuracy Score", chance, videos_chances)
    general_learning_plots(models_ea["val"], subs_ea, feature_set, paradigm, "EA Score")

    train_results_df = DataFrame(train_results, index=[feature_set])
    ALL_MODELS_GENERAL_TRAIN.append(train_results_df)

    test_results_df = DataFrame(test_results, index=[feature_set])
    ALL_MODELS_GENERAL_TEST.append(test_results_df)
    test_ea_df = DataFrame(models_ea["test"], index=[feature_set])
    ALL_MODELS_GENERAL_EA_TEST.append(test_ea_df)


def general_learning_plots(models_results, subs_results, feature_set, paradigm, y_label, chance=None, videos_chances=None):
    plot_models_result(models_results, feature_set, paradigm, chance, y_label)
    # # plots for each human_ea recording all its scores
    # plot_levels_bars(subs_scores, feature_set, GENERAL_PARADIGM_NAME)

    # gets best models for each human_ea stimuli
    subs_best_scores = get_subs_best_scores(subs_results)
    # plots for each human_ea recordings all its best scores
    plot_levels_bars(subs_best_scores, feature_set, paradigm, videos_chances, y_label=y_label)

    # saves for each subject its best results
    # if "EA" in y_label:
    #     save_subjects_results(subs_results, feature_set, ALL_SUBJECTS_GENERAL_EA)
    # else:
    #     save_subjects_results(subs_results, feature_set, ALL_SUBJECTS_GENERAL_RESULTS)


def get_subs_best_scores(scores):
    best_scores = {key: max(scores[key].items(), key=lambda x: x[1])[1] for key in scores}
    print(best_scores)
    # list of all best models for each subjects
    best_keys = [[item[0] for item in scores[key].items() if item[1] == best_scores[key]] for key in scores]

    best_unique = []  # best models that are unique for at least 1 subject
    to_fulfill = []  # list of all models that are mot unique (list of list)
    for item in best_keys:
        if len(item) == 1:
            best_unique.append(item[0])
        else:
            to_fulfill.append(item)

    # removing from to_fulfill the lists that are fulfilled with a model from best_unique
    left_keys = list()
    to_fulfill_temp = list()
    for item in to_fulfill:
        for model in item:
            if model in best_keys:
                break
        else:
            to_fulfill_temp.append(item)
            left_keys.extend(item)
    to_fulfill = to_fulfill_temp

    while to_fulfill:
        # repeat the procedure with a new model that appears the most
        new = Counter(left_keys).most_common(1)[0][0]
        left_keys = list()
        to_fulfill_temp = list()
        for item in to_fulfill:
            if new not in item:
                to_fulfill_temp.append(item)
                left_keys.extend(item)
        to_fulfill = to_fulfill_temp
        best_unique.append(new)

    subs_best_scores = {key: {model: scores[key][model] for model in best_unique} for key in scores.keys()}
    return subs_best_scores


def plot_RF_train_test(test, train, feature_set, mode):
    test_df = RF_depth_df(test, "test")
    train_df = RF_depth_df(train, "train")

    if mode != "":
        mode += " - "
    title = "%sRF train-test accuracy Vs. depth - %s" % (mode, feature_set)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # setting the amount of colors in the plot to be the amount of models
    cm = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax.set_color_cycle(cm[:test_df.shape[1]])

    test_df.plot(ax=ax)
    train_df.plot(ax=ax, style="--")
    set_plt_params(ax, title, "Tree's depth", "Accuracy score")

    if DEBUG:
        plt.show()
    else:
        plt.savefig(path.join(OUTPUT_PATH, FIGS_DIR, title + '.png'))
        plt.close()  # close the figure


def RF_depth_df(results, legend):
    results = {key: results[key] for key in results if "RF" in key}
    depths = {key.split("_")[-2] for key in results.keys()}
    results = {int(depth): {"_".join(key.split("_")[1:3]) + " " + legend: results[key]
                            for key in results if depth in key} for depth in depths}
    return DataFrame(results).transpose().sort_index()


def perform_within_sub_learning(videos_features, videos_labels, models, feature_set, mode):
    X_train, X_test, y_train, y_test = split_within_sub_train_test(videos_features, videos_labels)

    if mode != "":
        paradigm = " - ".join(["Within subject", mode])
    else:
        paradigm = "Within subject"

    # saving the test size for each subject
    test_size = dict()
    for subject in X_train.keys():
        test_size[subject] = len(X_test[subject]) / (len(X_test[subject]) + len(X_train[subject]))
        # print("subject %s" % subject, end=' ')
        # X_train[subject], y_train[subject] = balance_set(X_train[subject], y_train[subject])
    WITHIN_TEST_SIZE.append(DataFrame(test_size, index=[feature_set]))
    # saves the test videos for this feature set
    WITHIN_TEST_VIDS.append(DataFrame({subject: y_test[subject][0] for subject in y_test.keys()}, index=[feature_set]))

    # calculates chance levels for each subject
    chance_levels = dict()
    for subject in y_train.keys():
        most_common_val = Counter(y_train[subject]).most_common(1)[0][0]
        chance_levels[subject] = np.round(Counter(y_test[subject][1])[most_common_val] / len(y_test[subject][1]), 2)
    averaged_chance = np.mean(list(chance_levels.values()))
    chance_levels["overall"] = averaged_chance
    WITHIN_CHANCE[feature_set] = chance_levels

    models_results = dict()
    subjects_results = {key: dict() for key in X_train.keys()}
    subs_ea = {key[0]: dict() for key in y_test.values()}
    models_ea = {"test": dict(), "val": dict()}
    for clf, name in models:
        models_results[name] = dict()
        if POL_MODEL_NAME in name:
            models_results[name + POL_CAT_MODEL_NAME] = dict()
            cat_name = name + POL_CAT_MODEL_NAME
        predictions = dict()
        cat_pol_predictions = dict()
        true_labels = dict()
        for subject in X_train.keys():
            try:
                clf.fit(X_train[subject], y_train[subject])
                sub_predictions = clf.predict(X_test[subject])
                sub_y_test = y_test[subject][1]
                predictions[y_test[subject][0]] = sub_predictions
                true_labels[y_test[subject][0]] = sub_y_test

                score = clf.score(X_test[subject], sub_y_test)
                models_results[name][subject] = score
                subjects_results[subject][name] = score

                if POL_MODEL_NAME in name:
                    sub_predictions = np.round(sub_predictions)
                    cat_pol_predictions[y_test[subject][0]] = sub_predictions
                    score = accuracy_score(sub_y_test, sub_predictions)
                    models_results[cat_name][subject] = score
                    subjects_results[subject][cat_name] = score
            except ValueError as e:
                models_results[name][subject] = np.nan
                subjects_results[subject][name] = np.nan
                if POL_MODEL_NAME in name:
                    models_results[cat_name][subject] = np.nan
                    subjects_results[subject][cat_name] = np.nan
                print("could not perform on model: %s, features: %s, subject: %s. Error: %s" % (name, feature_set,
                                                                                                subject, str(e)))
            except MemoryError:
                models_results[name][subject] = np.nan
                subjects_results[subject][name] = np.nan
                if POL_MODEL_NAME in name:
                    models_results[cat_name][subject] = np.nan
                    subjects_results[subject][cat_name] = np.nan
                print("could not perform on model: %s, features: %s, subject: %s. Error: MemoryError" % (name, feature_set, subject))
        ea_results = export_results(predictions, true_labels, feature_set, WITHIN_SUBJECT_RESULTS, name, subs_ea,
                                    models_ea)
        # changes the names to the subject's name without file extension
        ea_results.columns = [col.split("_")[0] for col in ea_results.columns]
        EA_WITHIN_RESULTS.append(ea_results)
        if cat_pol_predictions:
            print("catt", cat_name)
            ea_results = export_results(cat_pol_predictions, true_labels, feature_set, WITHIN_SUBJECT_RESULTS, cat_name,
                                        subs_ea, models_ea)
            # changes the names to the subject's name without file extension
            ea_results.columns = [col.split("_")[0] for col in ea_results.columns]
            EA_WITHIN_RESULTS.append(ea_results)

    subs_results_df = DataFrame.from_dict(subjects_results, orient='index').transpose()
    subs_results_df.index = [[feature_set] * len(subs_results_df.index), subs_results_df.index]
    FULL_WITHIN_RESULTS.append(subs_results_df)

    within_learning_plots(models_results, subjects_results, feature_set, paradigm, "Accuracy Score",
                          averaged_chance, chance_levels)
    # remove the video's suffix name and stay with the subject name
    subs_ea = {key.split("_")[0]: subs_ea[key] for key in subs_ea}
    # models_ea already contains averaged results so replace this with subs_ea with switched keys
    models_ea = {model: {sub: subs_ea[sub][model] for sub in subs_ea.keys()} for model in models_results.keys()}
    within_learning_plots(models_ea, subs_ea, feature_set, paradigm, "EA Score")


def within_learning_plots(models_results, subs_results, feature_set, paradigm, y_label, averaged_chance=None,
                          chance_levels=None):
    averaged_models_results = {key: np.nanmean(list(models_results[key].values())) for key in models_results.keys()}
    std_models_results = {key: np.nanstd(list(models_results[key].values())) for key in models_results.keys()}
    best_models = plot_models_result(averaged_models_results, feature_set, paradigm, averaged_chance,
                                     y_label, error=std_models_results)

    # plots the models' results for each subject (not averaged), only for the best models
    # changes the key name in respect to the change in the best models' names
    plot_levels_bars({key: models_results[key.replace("\n", "_")] for key in best_models}, feature_set,
                     paradigm, averaged_chance, y_label=y_label, x_label="Model")

    # # plots for all subjects the max/average score
    # averaged_subjects_results = {key: np.max(list(subjects_results[key].values())) for key in subjects_results.keys()}
    # std_subjects_results = {key: np.std(list(subjects_results[key].values())) for key in subjects_results.keys()}
    # plot_model_performances(averaged_subjects_results, "Within subject", feature_set)  # , error=std_subjects_results)

    # gets best models for each subject
    subs_best_scores = get_subs_best_scores(subs_results)
    plot_levels_bars(subs_best_scores, feature_set, paradigm, chance_levels, y_label=y_label)

    # saves for each subject its best results
    # if "EA" in y_label:
    #     save_subjects_results(subs_results, feature_set, ALL_SUBJECTS_WITHIN_EA)
    # else:
    #     save_subjects_results(subs_results, feature_set, ALL_SUBJECTS_WITHIN_RESULTS)


def save_subjects_results(subs_results, feature_set, all_results):
    subs_best = {key: max(subs_results[key].values()) for key in subs_results}  # todo: max or mean?
    subs_df = DataFrame(subs_best, index=[feature_set])
    all_results.append(subs_df)


def plot_levels_bars(results, feature_set, paradigm, chance, y_label, x_label="Target"):
    """
    Plot multi levels bar graph.
    :param results: results dictionary where the keys are the groups (models) and the values are dictionaries where
    the keys are the sub-group (subjects) and the values are the bars (the subjects their model's scores)
    :param feature_set: a tuple with the features set name in the first index
    :param paradigm: The current trial's paradigm (string)
    :param chance: chance level for the model
    :param y_label: the y label (the values)
    :param x_label: the x label (the keys of the dictionary)
    :return:
    """
    if not results:
        return

    print(results)
    results_df = DataFrame.from_dict(results, orient='index')
    dfs = [(results_df, "all subjects' scores")]
    if len(results) > 10:
        mean_results = list({target: np.mean(list(results[target].values())) for target in results}.items())
        mean_results.sort(key=lambda item: item[1], reverse=True)
        best_targets = mean_results[:10]
        best_keys = [item[0] for item in best_targets]
        dfs.append((results_df.loc[best_keys], "top 10 subjects' scores"))

    for df, name in dfs:
        title = y_label + " - " + paradigm + " - %s - %s" % (name, feature_set)
        ax = df.plot(kind='bar', rot=0, title=title, legend=False)

        set_plt_params(ax, title, x_label, y_label, legend=False, chance=chance, width=WIDE_WIN)

        if DEBUG:
            plt.show()
        else:
            fig = ax.figure
            fig.savefig(path.join(OUTPUT_PATH, FIGS_DIR, title + '_%s.png' % x_label))

            # saves figure legend as a different file
            plt.legend(fontsize=TICKS_SIZE)
            fig_legend = pylab.figure(figsize=(4, 8))
            pylab.figlegend(*ax.get_legend_handles_labels(), loc='center')
            fig_legend.savefig(path.join(OUTPUT_PATH, FIGS_DIR, title + '_%s_legend.png' % x_label))
            ax.get_legend().remove()

            plt.close()  # close the figure


def get_double_dict_values(results, inner_dict_keys, outer_dict_keys):
    """
    Convert the results from dict of {model: subject: [values]} to [[subject 1 scores],[subject 2 scores] ...]
    """
    # other: list(zip([model.values() for model in results.values()]))

    # inner_dict_keys = results[list(results.keys())[0]].keys()
    # outer_dict_keys = results.keys()

    # new_results = dict()
    # for inner_key in inner_dict_keys:
    #     curr_values = list()
    #     for outer_key in outer_dict_keys:
    #         curr_values.append(results[outer_key][inner_key])

    return [[results[outer_key][inner_key] for outer_key in outer_dict_keys] for inner_key in inner_dict_keys]


def balance_set(features, labels):
    print("before bal", Counter(labels))
    labels_dict = Counter(labels)
    # max_amount = int(np.median(list(labels_dict.values())))
    max_amount = min(labels_dict.most_common(5), key=lambda x: x[1])[1]  # takes the fifth most common value
    all_indices = []
    for label in labels_dict.keys():
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)
        all_indices.extend(label_indices[:max_amount])
    print("after bal leave %f from the data" % (len(all_indices) / len(labels)), Counter(labels[all_indices]))
    return features[all_indices], labels[all_indices]


def main(func=lambda x: x, mode=""):
    for feature_set in FEATURES_FILES:
        print("CURR feature_set:", feature_set)
        full_features_amnt = feature_set[1] - IGNORED_COLS_NUM - 2
        n_features_list = [full_features_amnt]
        if full_features_amnt < 100:
            n_features_list += list(range(30, full_features_amnt, 15))
        else:
            n_features_list += [50]
        n_features_list += list(range(100, min(750, full_features_amnt), 150)) + \
            list(range(1000, min(6000, full_features_amnt), 1500))
        for n_features in n_features_list:
            print("CURR n_features:", n_features)
            videos_features, videos_labels = get_videos_features_labels(feature_set, n_features, transform=func)

            models = []
            # models += [(RandomForestClassifier(n_estimators=70, random_state=1), "RF_70_estimators")]  # Random Forest
            models += [(RandomForestClassifier(n_estimators=i, max_depth=j),
                       RF_MODEL_NAME + "_%d_trees_%d_depth" % (i, j))
                       for i in range(50, 250, 50) for j in range(6, 16, 3)]  # Random Forest
            if mode not in ["binary", "change"]:
                if feature_set == FEATURES_FILES[0]:
                    models += [(make_pipeline(PolynomialFeatures(i), linear_model.Ridge()),
                               POL_MODEL_NAME + "_%d_degree" % i) for i in range(1, 4)]
                else:
                    models += [(make_pipeline(PolynomialFeatures(i), linear_model.Ridge()),
                                POL_MODEL_NAME + "_%d_degree" % i) for i in range(1, 2)]

            models += [(svm.LinearSVC(max_iter=2000), SVM_MODEL_NAME)]  # SVM
            perform_general_learning(videos_features, videos_labels, models, "%s_%d" % (feature_set[0][:-4],
                                                                                        n_features), mode)
            perform_within_sub_learning(videos_features, videos_labels, models, "%s_%d" % (feature_set[0][:-4],
                                                                                           n_features), mode)

    # saving all the data frames
    if mode != "":
        mode += "_"
    within_test_size_output = concat(WITHIN_TEST_SIZE, sort=False).transpose()
    within_test_size_output.to_csv(path.join(OUTPUT_PATH, "%swithin_subject_test_sizes.csv" % mode))
    within_test_vids_output = concat(WITHIN_TEST_VIDS, sort=False)
    within_test_vids_output.to_csv(path.join(OUTPUT_PATH, "%swithin_subject_test_videos.csv" % mode))
    for df, name in [(DataFrame(GENERAL_CHANCE), "%sgeneral_learning_chance_level.csv" % mode),
                     (DataFrame(WITHIN_CHANCE), "%swithin_subject_chance_level.csv" % mode),
                     (within_test_size_output, "%swithin_subject_test_sizes.csv" % mode)
                     ]:
        df.to_csv(path.join(OUTPUT_PATH, name))
    for data, paradigm in [(EA_GENERAL_RESULTS, "%sgeneral_learning_ea" % mode),
                           (EA_WITHIN_RESULTS, "%swithin_subject_ea" % mode),
                           (FULL_GENERAL_RESULTS, "%sgeneral_learning_full" % mode),
                           (FULL_WITHIN_RESULTS, "%swithin_subject_full" % mode)
                           ]:
        with ExcelWriter(path.join(OUTPUT_PATH, paradigm + "_results.xlsx")) as writer:
            save_results(writer, data, "results")
    with ExcelWriter(path.join(OUTPUT_PATH, '%sall_models_results.xlsx' % mode)) as writer:
        general_results = save_all_models(writer, ALL_MODELS_GENERAL_RESULTS, "general")
        within_results = save_all_models(writer, ALL_MODELS_WITHIN_RESULTS, "within")
    with ExcelWriter(path.join(OUTPUT_PATH, '%sall_models_ea.xlsx' % mode)) as writer:
        general_ea = save_all_models(writer, ALL_MODELS_GENERAL_EA, "general")
        within_ea = save_all_models(writer, ALL_MODELS_WITHIN_EA, "within")
    with ExcelWriter(path.join(OUTPUT_PATH, '%sall_models_train_results.xlsx' % mode)) as writer:
        general_train = save_all_models(writer, ALL_MODELS_GENERAL_TRAIN, "general")

    # plots train-test trade-off
    plot_train_test(general_results, general_train, mode)

    # calculate the accuracy and EA correlation
    accuracy_ea_correlation(mode, general_results, general_ea, within_results, within_ea)

    general_test = concat(ALL_MODELS_GENERAL_TEST, sort=False).transpose()
    general_test.to_csv(path.join(OUTPUT_PATH, "%sall_models_test_results.csv" % mode))
    general_test = general_test.values.flatten()
    general_ea_test = concat(ALL_MODELS_GENERAL_EA_TEST, sort=False).transpose().values.flatten()
    best_acc = np.nanargmax(general_results.values)
    best_ea = np.nanargmax(general_ea.values)
    best_acc_model_name = general_results.index[best_acc // len(general_results.columns)]
    best_ea_model_name = general_ea.index[best_ea // len(general_ea.columns)]
    df = DataFrame({"best_acc_model (%s)" % best_acc_model_name:
                        {"accuracy": general_test[best_acc], "ea": general_ea_test[best_acc]},
                    "best_ea_model (%s)" % best_ea_model_name:
                        {"accuracy": general_test[best_ea], "ea": general_ea_test[best_ea]}})
    df.to_csv(path.join(OUTPUT_PATH, "%stest_results.csv" % mode))
    print("--------------TEST-------------")
    print(df)
    print("--------------TEST-------------")


def accuracy_ea_correlation(mode, general_results, general_ea, within_results, within_ea):
    all_corr = dict()
    print("--general correlations--")
    all_corr["general"] = df_correlation(general_results, general_ea)
    print("--within correlations--")
    all_corr["within"] = df_correlation(within_results, within_ea)
    DataFrame(all_corr).to_csv(path.join(OUTPUT_PATH, '%saccuracy_ea_correlation.csv' % mode))


def df_correlation(df1: DataFrame, df2: DataFrame):
    all_corr = dict()
    df1 = df1.transpose().sort_index()
    df2 = df2.transpose().sort_index()
    for column in df1.columns:
        # pearson correlation between the columns (model) where the values are not nan
        corr = pearsonr(df1[column].dropna(), df2[column].dropna())[0]
        all_corr[column] = corr
        print(column, corr)
    # overall correlation
    df1_values = df1.values.flatten().astype(np.float)
    df2_values = df2.values.flatten().astype(np.float)
    all_corr["overall"] = pearsonr(df1_values[~np.isnan(df1_values)], df2_values[~np.isnan(df2_values)])[0]
    return all_corr


def plot_train_test(general_results, general_train, mode):
    for feature_set in FEATURES_FILES:
        name = feature_set[0][:-4]
        title = "%strain_test_results_%s" % (mode, name)
        test = prepare_train_test_df(general_results, name, "test")
        train = prepare_train_test_df(general_train, name, "train")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # setting the amount of colors in the plot to be the amount of models
        # cm = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # cm = list(mcolors.XKCD_COLORS.values())
        # ax.set_color_cycle(cm[::len(cm) // (test.shape[1] - 1)])
        ax.set_color_cycle(COLORS[:test.shape[1]])

        test.plot(ax=ax)
        train.plot(ax=ax, style="--")
        set_plt_params(ax, title.replace("_", " "), "Number of features", "Accuracy score", width=WIDE_WIN)

        if DEBUG:
            plt.show()
        else:
            # saves figure legend as a different file
            fig_legend = pylab.figure(figsize=(4, 10))
            pylab.figlegend(*ax.get_legend_handles_labels(), loc='center')
            fig_legend.savefig(path.join(OUTPUT_PATH, FIGS_DIR, title + '_legend.png'))
            ax.get_legend().remove()

            fig.savefig(path.join(OUTPUT_PATH, FIGS_DIR, title + '.png'))
            plt.close()  # close the figure


def prepare_train_test_df(df: DataFrame, features_name: str, suffix=""):
    columns = [column for column in df.columns if features_name in column]
    df = df[columns]
    df.columns = [int(column.split("_")[-1]) for column in columns]
    df.index = [index + " " + suffix for index in df.index]
    return df.transpose().sort_index()


def save_all_models(writer, dfs, name):
    output = concat(dfs, sort=False).transpose()
    output.to_excel(writer, sheet_name=name)
    summary = get_df_stats(output, output.index)
    summary.to_excel(writer, sheet_name=name + '_models_summary')
    features_summary = get_df_stats(output)
    features_summary.to_excel(writer, sheet_name=name + '_features_summary')
    return output


def save_results(writer, data, name):
    df = concat(data, sort=False)
    summary = get_df_stats(df)
    df = df.reset_index().rename(columns={"level_0": "feature_set", "level_1": "model"})
    df.to_excel(writer, sheet_name=name, index=False)
    summary.to_excel(writer, sheet_name=name + '_subject_summary')
    features_groups = df.groupby('feature_set')
    features_groups.mean().to_excel(writer, sheet_name=name + '_mean_features')
    features_max = features_groups.max().drop(columns='model')
    features_max.to_excel(writer, sheet_name=name + '_max_features')
    features_summary = get_df_stats(features_max.transpose())
    features_summary.to_excel(writer, sheet_name=name + '_features_summary')


def get_df_stats(df, index=None):
    if index is not None:
        axis = 1
    else:
        axis = 0
    summary = DataFrame(index=index)
    summary['mean'] = df.mean(axis)
    summary['std'] = df.std(axis)
    summary['max'] = df.max(axis)
    summary['idxmax'] = df.idxmax(axis)
    return summary


def draw_tree():
    videos_features, videos_labels = get_videos_features_labels(FEATURES_FILES[0])
    X_train, X_test, y_train, y_test, _ = split_general_train_test(videos_features, videos_labels)

    feature_names = read_csv(path.join(BASE_DIR, FEATURES_DIR, FEATURES_PREFIX + FEATURES_FILES[0][0]), delimiter=',',
                             skiprows=4, usecols=(0,), names=['features'], nrows=88)
    feature_names = list(feature_names['features'])
    feature_names = [name.split(" ")[1] for name in feature_names]

    clf_tree = RandomForestClassifier(n_estimators=15, max_depth=9)
    clf_tree.fit(X_train, y_train)
    for i, estimator in enumerate(clf_tree.estimators_):
        export_graphviz(estimator, out_file='tree%d.dot' % i,
                        rounded=True, proportion=False,
                        feature_names=np.array(feature_names),
                        precision=2, filled=True)
        filename = path.join(OUTPUT_PATH, 'trees', 'tree%d.png' % i)
        call(['dot', '-Tpng', 'tree%d.dot' % i, '-o', filename, '-Gdpi=600'])


if __name__ == '__main__':
    run_main = True
    if run_main:
        main()
        output_path = OUTPUT_PATH
        for curr_mode, transform in [
            ("intensity", intensity_labels_transform),
            ("binary", binary_labels_transform),
            ("change", diff_labels_transform),
            ("binary intensity", binary_intensity_labels_transform)
        ]:
            print("MODE:", curr_mode)
            # clears variables
            ALL_MODELS_GENERAL_TRAIN = list()
            ALL_MODELS_GENERAL_RESULTS = list()
            ALL_MODELS_WITHIN_RESULTS = list()
            ALL_MODELS_GENERAL_EA = list()
            ALL_MODELS_WITHIN_EA = list()
            EA_GENERAL_RESULTS = list()
            EA_WITHIN_RESULTS = list()
            WITHIN_TEST_SIZE = list()
            WITHIN_TEST_VIDS = list()
            FULL_GENERAL_RESULTS = list()
            FULL_WITHIN_RESULTS = list()
            GENERAL_CHANCE = dict()
            WITHIN_CHANCE = dict()
            OUTPUT_PATH = path.join(output_path, curr_mode)
            main(transform, mode=curr_mode)
    else:
        modes = []
        modes = ["", "intensity", "binary", "change"]
        for curr_mode in modes:
            results_path = path.join(BASE_DIR, 'results/results', curr_mode)
            if curr_mode != "":
                curr_mode += "_"
            accuracy_file = '%sall_models_results.xlsx' % curr_mode
            ea_file = '%sall_models_ea.xlsx' % curr_mode
            accuracy_general = read_excel(path.join(results_path, accuracy_file), sheet_name='general')
            ea_general = read_excel(path.join(results_path, ea_file), sheet_name='general')
            accuracy_within = read_excel(path.join(results_path, accuracy_file), sheet_name='within')
            ea_within = read_excel(path.join(results_path, ea_file), sheet_name='within')
            accuracy_ea_correlation(accuracy_general, ea_general, accuracy_within, ea_within, curr_mode)
    # draw_tree()
