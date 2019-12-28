import pandas as pd
import numpy as np
from os import path, listdir
from code_files.preprocessing_features import BASE_DIR, VID_NAME_COL, LABEL_COL, get_labels, average_labels, \
    intensity_labels_transform, binary_labels_transform
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

HUMANS_DIR = "Human EA/val_ratings_result"
ML_BASE_DIR = "results"
ML_FINAL_DIR = "full_predictions"
ML_RESULTS_FILE = "Human EA/ML_results.csv"
WORKER_INFO_FILE = "Human EA/worker_info.csv"
AVERAGED_RATINGS = True
SCALE_10 = True
if AVERAGED_RATINGS:
    VID_EXTENSION = ".wav"
else:
    VID_EXTENSION = ".MP4"

humans_files = listdir(path.join(BASE_DIR, HUMANS_DIR))
# # counts the number of subjects
# df = pd.DataFrame()
# for file in humans_files:
#     df = df.append(pd.read_csv(path.join(BASE_DIR, HUMANS_DIR, file), usecols=(0,)), ignore_index=True)
# print(df.nunique())
# ids = [file[:6] for file in humans_files]
# audios = [file for file in humans_files if 'Audio' in file]
# vids_df = pd.DataFrame(False, index=list(df['worker_id'].unique()), columns=humans_files)
# vids_df2 = pd.DataFrame(False, index=list(df['worker_id'].unique()), columns=np.unique(ids))
# vids_df3 = pd.DataFrame(False, index=list(df['worker_id'].unique()), columns=audios)
# vids_df4 = pd.DataFrame(index=list(df['worker_id'].unique()), columns=np.unique(ids))
# for file in humans_files:
#     subjects = pd.read_csv(path.join(BASE_DIR, HUMANS_DIR, file), usecols=(0,))['worker_id']
#     for subject in subjects:
#         vids_df[file][subject] = True
#         vids_df2[file[:6]][subject] = True
#         if file in audios:
#             vids_df3[file][subject] = True
#         vids_df4[file[:6]][subject] = file.split('_')[-1][0]
# print(vids_df)

# ml_files = listdir(path.join(BASE_DIR, ML_BASE_DIR, ML_FINAL_DIR))
ml_results = pd.read_csv(path.join(BASE_DIR, ML_RESULTS_FILE), index_col=0)
ml_results = ml_results.mean()  # getting the mean results for each condition

worker_info = pd.read_csv(path.join(BASE_DIR, WORKER_INFO_FILE), usecols=(0, 9), index_col=0)
# worker_info = worker_info.drop_duplicates('worker_id', 'last').set_index('worker_id').transpose()

vids_ratings = dict()
for file in humans_files:
    file_suffix = "_Full_AudioOnly.csv"
    # file_suffix = "_Full_VideoOnly.csv"
    # file_suffix = "_Full_Full.csv"
    if file.endswith(file_suffix):
        df = pd.read_csv(path.join(BASE_DIR, HUMANS_DIR, file), index_col=0)
        if SCALE_10:
            df /= 10
        df = df.round()
        df = df.transpose()
        vid = file[3:-len(file_suffix)] + VID_EXTENSION
        vids_ratings[vid] = df

final_results = dict()
errors = dict()
for cond, transform_func in [
    ('valence', (lambda x: x)),
    ('intensity', intensity_labels_transform),
]:
    # Calculates the average EA scores of all subjects
    if AVERAGED_RATINGS:
        if SCALE_10:
            targets = get_labels(transform_func).groupby(VID_NAME_COL)
        else:
            targets = get_labels(transform_func, label_col="Rating").groupby(VID_NAME_COL)
    else:
        targets = pd.read_excel(path.join(BASE_DIR, "ratings", "ratings_table.xlsx"), index_col=0, sheet_name="ratings")
        targets = targets.reset_index().drop("Time", axis=1)
    subjects_scores = dict()
    for vid in vids_ratings:
        df = vids_ratings[vid]
        if AVERAGED_RATINGS:
            df = average_labels(df)
        for subject in df.columns:
            if worker_info.loc[subject].values.any():
                continue
            ratings = df[subject].dropna()
            ratings = transform_func(ratings)
            if ratings.astype(np.int).std() == 0:
                continue
            if AVERAGED_RATINGS:
                target = targets.get_group(vid)[LABEL_COL]
            else:
                target = targets[vid].dropna()
            vid_length = min(len(ratings), len(target))
            if subjects_scores.get(subject) is None:
                subjects_scores[subject] = list()
            subjects_scores[subject].append(pearsonr((ratings[:vid_length]).astype(np.int), (target[:vid_length]).astype(np.int))[0])
            # corrs.append(pearsonr((ratings[:vid_length]), (target[:vid_length]))[0])
        # subjects_scores[vid] = np.nanmean(corrs)  # SHOULD BE THE OOPRSITE DIRECTION. MEAN OVER VIDEOS
    subjects_final = {subject: np.nanmean(subjects_scores[subject]) for subject in subjects_scores.keys()}

    all_human_ea_scores = list(subjects_final.values())
    final_results[cond] = {"ML": ml_results[cond], "humans": np.nanmean(all_human_ea_scores)}
    errors[cond] = np.nanstd(all_human_ea_scores) / np.sqrt(len(all_human_ea_scores))

    print(all_human_ea_scores)
    print(np.nanmean(all_human_ea_scores), errors[cond])
    # print(ml_results)

final_df = pd.DataFrame(final_results).transpose()
print(final_df)
errors_df = pd.DataFrame({'humans': errors})
final_df.plot(kind='bar', rot=0, yerr=errors_df)
plt.grid(axis='y')
plt.show()
