"""
Dataset Construction for Eye-tracking Analysis.

This script constructs datasets from preprocessed eye-tracking data. It performs:
- Loading of post-processed data from `Processing.py`
- Data filtering and validation based on RR intervals
- Creation of local and global datasets for analysis
- Saving of intermediate and final datasets in pickle format.

Usage:
1. Ensure the post-processed data from `Processing.py` is available.
2. Configure the script paths and parameters.
3. Run the script: `python DatasetsConstruction.py`.

Input Data Requirements:
- Post-processed data from `Processing.py` (single and baselines).
- Valid RR intervals must be present in the data.

Output Datasets:
- dt_100: Dataset with at least 100 trials per condition.
- dt_100_3: Dataset with at least 3 blocks per subject.
- dt_ind: Dataset with individual trial offsets applied.
- dt_ind_15: Dataset filtered for at least 15 subjects per condition.
- dt_across: Dataset with averaged blocks.
- dt_across_15: Dataset filtered for at least 15 subjects per condition (averaged blocks).

Dependencies:
- numpy, pandas, os, pickle

Copyright (C) 2023 Giovanni Chiarion, chiarion.giovanni@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import pickle
from os.path import join, isfile

import numpy as np
import pandas as pd

# Configure pandas and plotly settings
pd.options.mode.copy_on_write = True  # Enable copy-on-write mode for pandas


# Define paths and directories
path = "/mnt/HDD2/giovanni/data/"
pwd = os.getcwd()

# Define condition labels and color mapping for visualization
original_condition = ["Baseline", "Sync", "Async", "Iso"]
labels = ["Baseline", "Synch", "Asynch", "Isoch"]
cond_map = dict(zip(original_condition, labels))

# %% Loading Data
# Load post-processed datasets if available
ppp_path = join(path, "intermediateData")
filenames = [
    f.split(".")[0] for f in os.listdir(ppp_path) if isfile(join(ppp_path, f))
]
ppp_dataset_names = [
    "dt_100",
    "dt_100_3",
    "dt_ind",
    "dt_ind_15",
    "dt_across",
    "dt_across_15",
]

if all((x in filenames) for x in ppp_dataset_names):
    # Load pre-processed datasets if they exist
    for file in ppp_dataset_names:
        globals()[file] = pd.read_pickle(join(ppp_path, file + ".pickle"))
else:
    # Load raw data and baselines if pre-processed datasets are unavailable
    pickle_file_single = path + "agg/post_processed_single.pickle"
    baseline_file_single = (
        path + "agg/baselines_cond/post_processed_single_BASELINES.pickle"
    )

    if os.path.exists(pickle_file_single) and os.path.exists(
        baseline_file_single
    ):
        print(
            f"Files {pickle_file_single} and {baseline_file_single} found. Loading..."
        )

        with open(pickle_file_single, "rb") as f:
            loaded_data = pickle.load(f)

        dataframe = loaded_data["dataframes"]["windowed"]
        dataframe["Condition"] = dataframe["Condition"].map(
            cond_map
        )  # Map condition labels
        dataframe["events_condition"] = dataframe["Condition"]
        dataframe.loc[
            dataframe["Condition"] == "Baseline", "events_condition"
        ] = "Synch"
        dataframe["events_condition"] = dataframe[
            "events_condition"
        ].cat.remove_unused_categories()

        wnd = loaded_data["dataframes"]["wnd"]
        wnd_sec = loaded_data["dataframes"]["wnd_sec"]
        num_tr = loaded_data["params"]["num_tr"]
        fs = loaded_data["params"]["fs"]

        del loaded_data  # Free up memory

        print("Loading baselines...")
        baselines = pd.read_pickle(baseline_file_single)
        baselines = baselines.query(
            "events_condition != 'Sync'"
        )  # Remove Sync condition
        baselines["Condition"] = baselines["Condition"].map(cond_map)
        baselines["events_condition"] = baselines["events_condition"].map(
            cond_map
        )

        print("Data loaded successfully.")
    else:
        print(f"ERROR! Missing {pickle_file_single} file in {path}")

# %% Local Dataset Creation
if not all([(x in locals()) for x in ppp_dataset_names]):
    # Filter data based on valid RR intervals
    min_rr = 0.5
    max_rr = 1.3
    dataframe.loc[~dataframe["RR"].between(min_rr, max_rr), "RR"] = np.nan
    baselines.loc[~baselines["RR"].between(min_rr, max_rr), "RR"] = np.nan

    def select_subj(g):
        # Remove subjects with insufficient trials per condition
        subj_trials = g.groupby(
            ["Condition", "events_condition"], observed=True
        )["trigger_counter"].nunique()
        if (subj_trials < min_trials_cond).any():
            return g.iloc[0:0]
        return g

    min_trials_cond = 100  # Minimum trials per condition and subject
    temp = pd.concat([dataframe, baselines])
    dt_100 = (
        temp.groupby(["subj_name", "event_block"], observed=True)
        .apply(select_subj)
        .reset_index(drop=True)
    )

    def copySecondRR(g):
        # Copy RR values from trigger_counter=1 to trigger_counter=0 to handle edge cases
        tc0 = g[g["trigger_counter"] == 0]
        tc1 = g[g["trigger_counter"] == 1]
        if (
            tc0.empty
            or tc1.empty
            or not tc0["RR"].isna().iloc[0]
            or tc1["RR"].isna().iloc[0]
        ):
            return g["RR"]
        result = g["RR"].copy()
        result.loc[g["trigger_counter"] == 0] = g.loc[
            g["trigger_counter"] == 1, "RR"
        ].iloc[0]
        return result

    dt_100["RR"] = dt_100.groupby(
        ["subj_name", "event_block", "Condition", "events_condition"],
        observed=True,
        group_keys=False,
    ).apply(copySecondRR)

    # Filter for merged event blocks with sufficient subjects
    min_block = 3
    dt_100_3 = dt_100.groupby(
        ["subj_name", "Condition", "events_condition"], observed=True
    ).filter(lambda x: x["event_block"].nunique() >= min_block)

    # Save datasets
    dt_100.to_pickle(os.path.join(ppp_path, "dt_100.pickle"))
    dt_100_3.to_pickle(os.path.join(ppp_path, "dt_100_3.pickle"))

# %% Global Dataset Generation
if not all([(x in locals()) for x in ppp_dataset_names]):
    # Apply offset to normalize data
    s = {}  # Track missing trials
    y = "pupil_right_norm"

    def apply_offset(g, n):
        subj = g.name[0]
        block = g.name[1]
        cond = g.name[2]
        mask = g["trigger_counter"] < n
        first_n = g[mask]
        num_tr = len(first_n["trigger_counter"].unique())
        if num_tr < 0.5 * n:
            if not s.get(cond):
                s[cond] = []
            s[cond].append([subj, block, num_tr])
            if num_tr == 0:
                if cond == "Baseline":
                    first_available = g["trigger_counter"].unique()[0]
                    mask = g["trigger_counter"] == first_available
                    first_n = g[mask]
                else:
                    return None
        off = (
            first_n.groupby("trigger_counter", observed=True)[[y, "RR"]]
            .mean()
            .mean()
        )
        g[y] = g[y] - off[y]
        g["RR"] = g["RR"] - off["RR"]
        return g

    n = 10
    dt_ind = dt_100.groupby(
        ["subj_name", "event_block", "Condition", "events_condition"],
        observed=True,
    )[
        [
            "time",
            "time_subj_trig",
            y,
            "window_sec",
            "trigger_counter",
            "r_peaks",
            "RR",
        ]
    ].apply(
        apply_offset, n
    )
    dt_ind = dt_ind.reset_index(drop=False).set_index(
        dt_ind.index.get_level_values(-1)
    )

    def discard_subj(s):
        # Identify subjects to discard due to missing trials
        to_discard = [[] for _ in range(6)]
        for cond, val in s.items():
            if cond != "Baseline":
                for el in val:
                    subj = el[0]
                    block = el[1]
                    num_tr = el[2]
                    if num_tr == 0:
                        to_discard[block].append(subj)
        return to_discard

    subj_to_discard = discard_subj(s)
    dt_ind = (
        dt_ind.groupby("event_block")
        .apply(
            lambda g: g.query(f"subj_name not in {subj_to_discard[g.name]}")
        )
        .droplevel(0)
    )
    dt_ind.to_pickle(os.path.join(ppp_path, "dt_ind.pickle"))

    # Filter for trials with at least 15 subjects
    temp = dt_ind.groupby(
        [
            "Condition",
            "trigger_counter",
            "subj_name",
            "event_block",
            "events_condition",
        ],
        observed=True,
    ).filter(lambda x: x["subj_name"].nunique() >= 15)
    dt_ind_15 = pd.merge(
        dt_ind,
        temp[
            [
                "subj_name",
                "event_block",
                "Condition",
                "trigger_counter",
                "events_condition",
            ]
        ],
        how="inner",
    )
    dt_ind_15.to_pickle(os.path.join(ppp_path, "dt_ind_15.pickle"))

    # Repeat for averaged blocks
    dt_across = (
        dt_100_3.groupby(
            ["subj_name", "event_block", "Condition", "events_condition"],
            observed=True,
        )
        .apply(apply_offset, n)
        .reset_index(drop=False)
    )
    dt_across = (
        dt_across.groupby("event_block")
        .apply(
            lambda g: g.query(f"subj_name not in {subj_to_discard[g.name]}")
        )
        .droplevel(0)
    )
    dt_across.to_pickle(os.path.join(ppp_path, "dt_across.pickle"))

    temp = dt_across.groupby(
        [
            "Condition",
            "trigger_counter",
            "subj_name",
            "event_block",
            "events_condition",
        ],
        observed=True,
    ).filter(lambda x: x["subj_name"].nunique() >= 15)
    dt_across_15 = pd.merge(
        dt_across,
        temp[["subj_name", "event_block", "Condition", "trigger_counter"]],
        how="inner",
    )
    dt_across_15.to_pickle(os.path.join(ppp_path, "dt_across_15.pickle"))
