from os import path, listdir, rename
import pandas as pd
import numpy as np
from collections import Counter
import itertools
from code_files.preprocessing_features import BASE_DIR, VID_NAME_COL, TIME_COL, SUBJECT_COL

CURR_DIR = path.join(BASE_DIR, 'ratings')
OUTPUT_PATH = path.join(CURR_DIR, 'all_together.csv')
STABLE_REF_COL = 'Scale-10'


def combine_files():
    all_together = []

    rating_files = listdir(CURR_DIR)
    for filename in rating_files:
        if 'RatingLog' not in filename or '.csv' not in filename:
            continue
        full_path = path.join(CURR_DIR, filename)
        data = pd.read_csv(full_path, sep=",")
        data[VID_NAME_COL] = data.apply(lambda row: str(row['SubID']) + '_' + row['MovieID'].split('_')[1], axis=1)
        # if (data['Time'][1] - data['Time'][0]) < 0.4:
        data = data[data[TIME_COL] % 0.5 <= 0.06]  # leave only every 0.5 seconds ratings
        data[SUBJECT_COL] = data['SubID']
        all_together.append(data[[SUBJECT_COL, VID_NAME_COL, TIME_COL, 'Rating']])

    output = pd.concat(all_together, ignore_index=True)
    # adding rounded rates in different scales
    output['Scale-10'] = output.apply(lambda row: round(row['Rating'] / 10), axis=1)
    output['Scale-20'] = output.apply(lambda row: round(row['Rating'] / 5), axis=1)
    output.to_csv(OUTPUT_PATH, index=False, sep=',')


def ratings_table():
    ratings = pd.read_csv(OUTPUT_PATH, sep=',')
    ratings["positive"] = ratings['Rating'] > 55
    ratings["negative"] = ratings['Rating'] < 45
    vid_groups = ratings.groupby(VID_NAME_COL)
    output = pd.DataFrame({TIME_COL: np.arange(0, ratings[TIME_COL].max() + 0.5, 0.5)})
    for vid, group in vid_groups:
        output[vid] = group.reset_index()['Rating']

    videos_stats, stats = get_videos_stats(vid_groups)
    subs_stats = get_subs_stats(ratings)

    stats["targets_amount"] = len(subs_stats)
    stats["min_recordings_per_target"] = subs_stats['n_videos'].min()
    stats["max_recordings_per_target"] = subs_stats['n_videos'].max()
    stats["mean_recordings_per_target"] = subs_stats['n_videos'].mean()
    stats["mean_target_mean_length"] = subs_stats['length_mean'].mean()
    stats = pd.DataFrame.from_dict(stats, orient='index').reset_index()

    filename = path.join(CURR_DIR, "ratings_table.xlsx")
    with pd.ExcelWriter(filename) as writer:
        output.to_excel(writer, sheet_name="ratings", index=False)
        videos_stats.to_excel(writer, sheet_name="videos_stats")
        subs_stats.to_excel(writer, sheet_name="targets_stats")
        stats.to_excel(writer, sheet_name="general_stats", header=False, index=False)


def get_videos_stats(vid_groups):
    ratings_stats = compute_ratings_stats(vid_groups)

    vid_times = vid_groups['Time'].max()
    ratings_stats['length'] = vid_times

    stats = {
        "min_length": vid_times.min(),
        "max_length": vid_times.max(),
        "mean_length": vid_times.mean(),
        "median_length": vid_times.median(),
        "std_length": vid_times.std(),
        "total_recordings": len(vid_groups.groups.keys()),
        "positive_amount": is_mostly_val(vid_groups, 'positive').sum(),
        "negative_amount": is_mostly_val(vid_groups, 'negative').sum(),
        # "neutral_amount": pos_neg_ratio(vid_groups, pd.core.series.Series.__eq__).sum(),
        "mean_min_rate": ratings_stats['rate_min'].mean(),
        "mean_max_rate": ratings_stats['rate_max'].mean(),
        "mean_mean_rate": ratings_stats['rate_mean'].mean(),
        "mean_std_rate": ratings_stats['rate_std'].mean(),
        "std_mean_rate": ratings_stats['rate_mean'].std(),
    }

    return ratings_stats, stats


def get_subs_stats(ratings):
    def get_sub_groups(df):
        return df.groupby(SUBJECT_COL)

    sub_groups = get_sub_groups(ratings)
    double_grouped = ratings.groupby([SUBJECT_COL, VID_NAME_COL])
    sub_times = get_sub_groups(double_grouped['Time'].max())

    stats = compute_ratings_stats(sub_groups)

    stats["n_videos"] = sub_groups['vid_name'].nunique()
    stats["n_positive"] = get_sub_groups(is_mostly_val(double_grouped, 'positive')).sum()
    stats["n_negative"] = get_sub_groups(is_mostly_val(double_grouped, 'negative')).sum()
    # stats["n_neutral"] = get_sub_groups(is_mostly_val(double_grouped, pd.core.series.Series.__eq__)).sum()
    stats["length_min"] = sub_times.min()
    stats["length_max"] = sub_times.max()
    stats["length_mean"] = sub_times.mean()
    stats["length_median"] = sub_times.median()

    return stats


def compute_ratings_stats(groups):
    ratings = groups['Rating']
    ratings_stats = pd.DataFrame()
    ratings_stats["rate_min"] = ratings.min()
    ratings_stats["rate_max"] = ratings.max()
    ratings_stats["rate_mean"] = ratings.mean()
    ratings_stats["rate_std"] = ratings.std()
    ratings_stats["rate_median"] = ratings.median()
    return ratings_stats


def is_mostly_val(groups, val):
    return groups[val].sum() / groups[val].count() > 0.5


def find_stable():
    data = pd.read_csv(OUTPUT_PATH, sep=",")
    stable_seqs = {}
    data_iter = data.iterrows()
    row = next(data_iter)[1]
    curr_vid = row[VID_NAME_COL]
    curr_value = row[STABLE_REF_COL]
    stable_seq_len = 1
    vid_seqs = []
    for _, row in data_iter:
        if row[VID_NAME_COL] == curr_vid and curr_value == row[STABLE_REF_COL]:
            stable_seq_len += 1
        else:
            if stable_seq_len > 1:  # add only if it is longer than 1
                vid_seqs.append(stable_seq_len)
            curr_value = row[STABLE_REF_COL]
            stable_seq_len = 1
        if row[VID_NAME_COL] != curr_vid:
            stable_seqs[curr_vid] = vid_seqs
            curr_vid = row[VID_NAME_COL]
            vid_seqs = []
    # saving the last sequence
    if stable_seq_len > 1:  # add only if it is longer than 1
        vid_seqs.append(stable_seq_len)
    # saving the last video
    stable_seqs[curr_vid] = vid_seqs

    all_seqs = list(itertools.chain.from_iterable(stable_seqs.values()))
    summary = pd.DataFrame.from_dict(Counter(all_seqs), orient='index')
    summary.columns = ["amount"]

    output = pd.DataFrame.from_dict(stable_seqs, orient='index').transpose()
    stats = pd.DataFrame()
    stats["min"] = output.min()
    stats["max"] = output.max()
    stats["mean"] = output.mean()
    stats["median"] = output.median()
    with pd.ExcelWriter(path.join(CURR_DIR, 'stable_info.xlsx')) as writer:
        output.to_excel(writer, sheet_name="output", index=False)
        stats.to_excel(writer, sheet_name="stats")
        summary.to_excel(writer, sheet_name="amounts")


def fix_time(data):
    # get the time and valence vectors for this videoID
    curr_time_vec = data[TIME_COL].values
    valence_vec = data['Rating'].values

    # create a "rounded" time vector. 0, 0.5, 1.0, 1.5...
    time_rounded = [float(x) / 2 for x in range(int(2 * curr_time_vec[-1]) + 1)]

    # interpolate the valence vector
    valence_interpolated = np.round(np.interp(time_rounded, curr_time_vec, valence_vec))

    return pd.DataFrame({TIME_COL: time_rounded, 'Rating': valence_interpolated})


def fix_data_file():
    file = "ID53_original.csv"
    full_path_file = path.join(CURR_DIR, file)
    data = pd.read_csv(full_path_file, sep=',')
    # rename(path.join(CURR_DIR, file), full_path_file.replace('RatingLog', "original"))

    new_data = pd.DataFrame(columns=data.columns)
    for key, group in data.groupby('Serial'):
        group = group.reset_index()
        fixed_data = fix_time(group)  # contains only the 2 columns that were changed
        # since fixed_data is not in the same length, creating the extra columns again
        for col in data.columns:
            if col not in fixed_data.columns:
                fixed_data[col] = group[col][0]
        new_data = new_data.append(fixed_data, ignore_index=True)

    new_data = new_data[data.columns]
    new_data.to_csv( path.join(CURR_DIR, "ID53_RatingLog.csv"), index=False)


# fix_data_file()
# combine_files()
# ratings_table()
find_stable()
