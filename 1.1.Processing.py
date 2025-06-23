#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eye-tracking Data Preprocessing Pipeline.

This script processes eye-tracking data from Pupil Labs eye-trackers. It performs:
- Data loading and windowing around triggers
- Trial rejection based on NaN values, statistical outliers, and rolling window artifacts
- Normalization and baseline correction of pupil data
- RR interval analysis for cardiac data synchronization
- Data saving in pickle format for downstream analysis.

Usage:
1. Ensure the raw data is in a pickle file with the required fields (see below).
2. Configure the script parameters (data_path, process_type, thresholds, etc.).
3. Run the script: `python Processing.py`.

Input Data Requirements:
The raw data must be a Pandas DataFrame (saved as a pickle file) with the following fields:
- subj_name: Subject identifier (str)
- event_block: Number of the experimental block (int from 0)
- Condition: The condition name (Synch, Asynch, Isoch, or Baseline)
- trigger_counter: Number of the trial (int from 0)
- event_code: Label containing the position of sound onset (ST)
- pupil_right: Data from the right pupil
- ECG: ECG data

Dependencies:
- numpy, pandas, plotly, pickle, os, time, EyetrackerAnalysis (custom module)

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

import numpy as np
import pandas as pd
from os.path import join
from EyetrackerAnalysis import EyetrackerAnalysis
import pickle
import os
from time import perf_counter_ns

# =============================================================================
# Configuration Settings
# =============================================================================

# Plotting configuration
pd.options.mode.copy_on_write = (
    True  # Enable copy-on-write mode for safer operations
)

# Data processing parameters
data_path = "/mnt/HDD2/giovanni/data/agg/data.pickle"  # Path to raw data
process_type = "single"  # Processing mode: "single" or "multi"
compute_baselines = False  # Whether to compute baseline conditions

# Threshold parameters for trial rejection
if process_type == "single":
    path = "/mnt/HDD2/giovanni/data/agg/"
    pickle_file = path + "processed_" + process_type + ".pickle"
    num_tr = 1  # Number of triggers per trial
    thr_nan = 1  # Threshold for NaN values
    thr_std = 3  # Standard deviation threshold
    n_samp_rej = 1  # Number of samples to reject
    thr_rolling = 30  # Rolling window threshold
    rolling_tol = 3  # Rolling tolerance
    thr_across = 3  # Across-trial threshold
elif process_type == "multi":
    path = "/mnt/HDD2/giovanni/data/agg/multi/"
    pickle_file = path + "processed_" + process_type + ".pickle"
    num_tr = 3  # Number of triggers per trial
    thr_nan = 1  # Threshold for NaN values
    thr_std = 5  # Standard deviation threshold (higher for multi-trial)
    n_samp_rej = 1  # Number of samples to reject
    thr_rolling = 30  # Rolling window threshold
    rolling_tol = 5  # Rolling tolerance (higher for multi-trial)
    thr_across = 5  # Across-trial threshold (higher for multi-trial)

# =============================================================================
# Data Loading and Initial Processing
# =============================================================================


def load_or_process_data():
    """
    Load preprocessed data if available, otherwise process raw data.

    Returns:
        e: EyetrackerAnalysis object with processed data
        params: Dictionary of processing parameters
    """
    if os.path.exists(pickle_file):
        print(f"File {pickle_file} found. Loading...")

        with open(pickle_file, "rb") as f:
            loaded_data = pickle.load(f)

        # Initialize EyetrackerAnalysis object
        e = EyetrackerAnalysis(
            data_folder=path,
            data_path=data_path,
            fs=500,  # Sampling frequency (Hz)
        )

        # Load preprocessed dataframes
        e.windowed = loaded_data["dataframes"]["windowed"]
        e.data_df = loaded_data["dataframes"]["data_df"]
        e.wnd = loaded_data["dataframes"]["wnd"]
        e.wnd_sec = loaded_data["dataframes"]["wnd_sec"]

        # Extract processing parameters
        params = {
            "num_tr": loaded_data["params"]["num_tr"],
            "fs": loaded_data["params"]["fs"],
            "thr_nan": loaded_data["params"]["thr_nan"],
            "thr_std": loaded_data["params"]["thr_std"],
            "n_samp_rej": loaded_data["params"]["n_samp_rej"],
            "thr_rolling": loaded_data["params"]["thr_rolling"],
            "rolling_tol": loaded_data["params"]["rolling_tol"],
        }

        del loaded_data
        print("Data loaded successfully.")
        return e, params

    else:
        print(f"File {pickle_file} not found. Processing raw data...")

        # Initialize and process data
        e = EyetrackerAnalysis(
            data_folder=path,
            data_path=data_path,
            fs=500,
        )

        # Load and window data
        e.load_data(load_windowed=False)
        e.make_windowed(
            wnd_sec=[-100 * 1e-3, 600 * 1e-3],  # Window from -100ms to 600ms
            num_tr=num_tr,
            offset_samp_soundmock=26,  # Offset for sound mock triggers
        )

        # Trial rejection
        _, _, _, original = e.make_trial_rejection(
            thr_nan, thr_std, n_samp_rej, thr_rolling, rolling_tol
        )

        # Normalization and baseline correction
        e.make_zScore()
        e.make_baseline_correction(on="pupil_right_norm")

        # Prepare data for saving
        params = {
            "num_tr": num_tr,
            "fs": e.fs,
            "thr_nan": thr_nan,
            "thr_std": thr_std,
            "n_samp_rej": n_samp_rej,
            "thr_rolling": thr_rolling,
            "rolling_tol": rolling_tol,
        }

        data_to_save = {
            "dataframes": {
                "original": original,
                "windowed": e.windowed,
                "data_df": e.data_df,
                "wnd": e.wnd,
                "wnd_sec": e.wnd_sec,
            },
            "params": params,
        }

        # Save processed data
        with open(pickle_file, "wb") as f:
            pickle.dump(data_to_save, f)
        print(f"Data saved successfully to {pickle_file}")

        return e, params


# Load or process data
e, params = load_or_process_data()
fs = params["fs"]  # Sampling frequency

# =============================================================================
# Data Selection and Preparation
# =============================================================================

# Select relevant dataframes
selected_data = e.data_df
selected_windowed = e.windowed

# Align windowed data to first ST (sound trigger) event
first_trig_pos = (
    selected_windowed[selected_windowed.event_code == "ST"]
    .window_samp.head(1)
    .values[0]
)

# Convert sample indices to seconds relative to first trigger
selected_windowed["window_sec"] = (
    selected_windowed["window_samp"] - first_trig_pos
) / fs

# Convert categorical variables to proper category dtype
selected_data["event_block"] = selected_data["event_block"].astype("category")
selected_windowed["event_block"] = selected_windowed["event_block"].astype(
    "category"
)
selected_windowed["trigger_counter"] = selected_windowed[
    "trigger_counter"
].astype("int16")

# Remove trials with trigger_counter > 300 in Baseline
selected_windowed = selected_windowed.query("trigger_counter < 300")

# =============================================================================
# Trial Cleaning and Rejection
# =============================================================================


def clean_trials(dt, process_type, e):
    """
    Clean trials by removing duplicates and incomplete trials.

    Args:
        dt: DataFrame containing windowed data
        process_type: "single" or "multi" processing mode
        e: EyetrackerAnalysis object

    Returns:
        Cleaned DataFrame
    """
    if process_type == "single":
        # Remove duplicate indices
        dt_no_dup = dt[~dt.index.duplicated(keep="first")]

        # Calculate expected number of samples per trial
        num_samp = np.sum(np.abs(e.wnd_sec)) * e.fs + 1

        def remove_trials(g):
            """Filter function to keep only complete trials."""
            return g["pupil_right"].count() == num_samp

        # Apply filtering
        dt = dt_no_dup.groupby(
            ["subj_name", "event_block", "Condition", "trigger_counter"],
            observed=True,
        ).filter(remove_trials)

        assert dt.index.is_unique  # Verify no duplicate indices remain
        del dt_no_dup

    return dt


dt = clean_trials(selected_windowed, process_type, e)


def reject_outlier_trials(dt, thr_across, y="pupil_right_norm"):
    """
    Reject trials that are outliers across the dataset.

    Args:
        dt: DataFrame containing windowed data
        thr_across: Threshold for rejection (in standard deviations)
        y: Column name to use for rejection criteria

    Returns:
        Filtered DataFrame with outliers removed
    """

    def remove_exc_trials(g, thr, y):
        """Helper function to remove trials outside threshold bounds."""
        grp = g.groupby("trigger_counter", observed=True)[y]
        m = grp.mean().mean()
        s = grp.mean().std()
        bounds = (m - thr * s, m + thr * s)
        keep = grp.transform(lambda x: x.between(*bounds).all())
        return g.loc[keep, :]

    # Apply rejection across groups
    dt = dt.groupby(
        ["event_block", "Condition", "subj_name"],
        observed=True,
    ).apply(remove_exc_trials, thr_across, y)
    dt.index = dt.index.droplevel([0, 1, 2])

    return dt


# Apply trial rejection
dt = reject_outlier_trials(dt, thr_across)

# Calculate and report rejection statistics
num_before = (
    e.windowed.groupby(
        ["subj_name", "event_block", "Condition"], observed=True
    )["trigger_counter"]
    .nunique()
    .sum()
)

num_actual = (
    dt.groupby(["subj_name", "event_block", "Condition"], observed=True)[
        "trigger_counter"
    ]
    .nunique()
    .sum()
)

removed = num_before - num_actual
print(
    f"Removed trials: {removed} out of {num_before} "
    f"({np.round(removed*100/num_before)}% of total)"
)
print(
    f"Removed hours from original: {(len(selected_windowed) - len(dt)) / fs / 3600:.2f}"
)

# =============================================================================
# Data Normalization and Correction
# =============================================================================


def normalize_and_correct(dt, process_type):
    """
    Apply z-score normalization and baseline correction.

    Args:
        dt: DataFrame containing windowed data
        process_type: "single" or "multi" processing mode

    Returns:
        DataFrame with normalized and corrected data
    """
    # Z-score normalization
    m = dt.groupby(["subj_name", "event_block"], observed=True)[
        "pupil_right"
    ].transform("mean")
    s = dt.groupby(["subj_name", "event_block"], observed=True)[
        "pupil_right"
    ].transform("std")
    dt.loc[:, "pupil_right_norm"] = (dt["pupil_right"] - m) / s

    # Baseline correction
    def make_baseline_corr(g):
        """Calculate baseline-corrected values."""
        norm = g[on]
        med = g.query("window_sec<0")[on].median()
        return norm - med

    # Apply to pupil_right_norm
    on = "pupil_right_norm"
    abbr = "".join(word[0] for word in on.split("_")) if "_" in on else on

    baseline_corrected = dt.groupby(
        ["subj_name", "Condition", "event_block", "trigger_counter"],
        observed=True,
    )[[on, "window_sec"]].apply(make_baseline_corr)
    baseline_corrected.name = f"baseline_corrected_{abbr}"
    baseline_corrected.index = baseline_corrected.index.droplevel([0, 1, 2, 3])

    dt = dt.drop(columns=f"baseline_corrected_{abbr}", errors="ignore")
    dt = dt.join(baseline_corrected)

    return dt


dt = normalize_and_correct(dt, process_type)

# =============================================================================
# RR Interval Analysis (Cardiac Data)
# =============================================================================


def process_rr_intervals(selected_data, dt):
    """
    Process RR intervals from cardiac data and map to trials.

    Args:
        selected_data: DataFrame containing raw data with R-peaks
        dt: DataFrame containing windowed trial data

    Returns:
        DataFrame with RR intervals mapped to trials
    """
    # Calculate RR intervals from R-peaks
    selected_data["RR"] = (
        selected_data.groupby(
            ["subj_name", "event_block", "Condition"],
            observed=True,
            as_index=False,
        )
        .apply(lambda x: x.loc[x["r_peaks"]], include_groups=True)
        .reset_index(level=0, drop=True)
        .groupby(["subj_name", "event_block", "Condition"], observed=True)[
            "time"
        ]
        .diff()
    )

    # Track processing time per subject
    prev_subj = ""
    previous_time = None

    def RR_to_trial(g, rr_peaks_data):
        """
        Map RR intervals to trials based on closest R-peak to sound trigger.

        Args:
            g: Trial group DataFrame
            rr_peaks_data: DataFrame containing RR interval data

        Returns:
            Trial group with RR interval assigned
        """
        nonlocal prev_subj, previous_time
        subj, block, cond, tr = g.name

        # Print progress when starting new subject
        if subj != prev_subj:
            actual_time = perf_counter_ns()
            if previous_time:
                print(f"--- {(actual_time-previous_time)/1e9} seconds\n")
            previous_time = perf_counter_ns()
            print(f"Evaluating {subj}")
            prev_subj = subj

        # Find relevant RR intervals for this subject/block/condition
        relevant_rr = rr_peaks_data.query(
            "subj_name == @subj & event_block == @block & Condition == @cond",
            engine="python",
        )

        # Get trigger time for this trial
        trigger_time = g.loc[g["event_code"] == "ST", "time"].iloc[0]

        # Find closest R-peak to trigger
        closest_rr_index = (relevant_rr["time"] - trigger_time).abs().idxmin()
        closest_rr_time = relevant_rr.at[closest_rr_index, "time"]

        # Assign RR if within acceptable range, otherwise mark as NaN
        if (
            np.abs(closest_rr_time - trigger_time) > 0.8
        ):  # 800ms maximum distance
            print(
                f"No close rpeak found for trial {tr} in block {block} for {cond}"
            )
            g["RR"] = np.nan
        else:
            closest_rr = relevant_rr.at[closest_rr_index, "RR"]
            g["RR"] = closest_rr

        return g

    # Prepare RR peak data
    rr_peaks_data = selected_data.loc[
        selected_data["r_peaks"],
        ["time", "RR", "subj_name", "event_block", "Condition"],
    ]

    # Map RR intervals to trials
    dt = (
        dt.groupby(
            ["subj_name", "event_block", "Condition", "trigger_counter"],
            observed=True,
        ).apply(RR_to_trial, rr_peaks_data)
    ).reset_index(level=[0, 1, 2, 3], drop=True)

    return dt


dt = process_rr_intervals(selected_data, dt)

# =============================================================================
# Data Saving
# =============================================================================


def save_processed_data(dt, e, path, process_type):
    """
    Save processed data to pickle file.

    Args:
        dt: Processed DataFrame to save
        e: EyetrackerAnalysis object with window parameters
        path: Base path for saving
        process_type: "single" or "multi" processing mode
    """
    data_to_save = {
        "dataframes": {
            "windowed": dt,
            "wnd": e.wnd,
            "wnd_sec": e.wnd_sec,
        },
        "params": {
            "num_tr": num_tr,
            "fs": fs,
        },
    }

    with open(path + f"post_processed_{process_type}.pickle", "wb") as f:
        pickle.dump(data_to_save, f)
    print(
        "Data saved successfully to",
        path + f"post_processed_{process_type}.pickle",
    )


save_processed_data(dt, e, path, process_type)

# =============================================================================
# Baseline Processing (Optional)
# =============================================================================

if compute_baselines and process_type == "single":

    def clear_workspace():
        """Clear workspace variables except key configuration parameters."""
        gl = globals().copy()
        for var in gl:
            if var.startswith("_"):
                continue
            if "func" in str(globals()[var]):
                continue
            if "module" in str(globals()[var]):
                continue
            if var in [
                "process_type",
                "num_tr",
                "thr_nan",
                "thr_std",
                "n_samp_rej",
                "thr_rolling",
                "rolling_tol",
                "path",
                "testing",
            ]:
                continue
            del globals()[var]

    clear_workspace()

    # Reinitialize configuration
    pd.options.mode.copy_on_write = True

    # Initialize and load data
    e = EyetrackerAnalysis(
        data_folder=path + "agg",
        data_path=data_path,
        fs=500,
    )
    e.load_data(load_windowed=True)
    fs = e.fs
    triggers = e.triggers
    data_df = e.data_df

    # Process baseline conditions
    conds = ["Sync", "Async", "Iso"]
    path_baseline = path + "agg/baselines_cond"

    def create_baseline_conditions(data_df, conds, path_baseline):
        """
        Create baseline conditions by copying event codes from experimental conditions.

        Args:
            data_df: DataFrame containing all data
            conds: List of condition names to process
            path_baseline: Path to save baseline files
        """
        baselines = {}
        cond_filenames = [x + "_Baseline" for x in conds]
        filenames = [
            f.split(".")[0]
            for f in os.listdir(path_baseline)
            if os.path.isfile(join(path_baseline, f))
        ]

        if not all(x in filenames for x in cond_filenames):
            for cond in conds:
                print(f"Processing {cond}")

                # Copy Baseline data
                baselines[cond] = data_df.loc[
                    data_df["Condition"] == "Baseline"
                ]
                baselines[cond].loc[:, "event_code"] = -1  # Reset event codes

                # Copy event codes from experimental condition to baseline
                def mimicSTST(g, cond):
                    """Copy event codes from condition to baseline."""
                    subj, eb = g.name
                    b_index = (
                        baselines[cond]
                        .loc[
                            (baselines[cond]["event_block"] == eb)
                            & (baselines[cond]["subj_name"] == subj)
                        ]
                        .index
                    )
                    min_len = min(len(g), len(b_index))
                    baselines[cond].loc[b_index[:min_len], "event_code"] = (
                        g.values[:min_len]
                    )

                data_df.loc[data_df["Condition"] == cond].groupby(
                    ["subj_name", "event_block"]
                )["event_code"].apply(mimicSTST, cond)

            # Save baseline conditions
            for cond in baselines:
                baselines[cond].to_pickle(
                    join(path_baseline, f"{cond}_Baseline.pickle")
                )

    create_baseline_conditions(data_df, conds, path_baseline)

    # Process each baseline condition
    filenames = [
        f.split(".")[0]
        for f in os.listdir(path_baseline)
        if os.path.isfile(join(path_baseline, f))
    ]

    for cond in conds:
        if f"windowed_{cond}" not in filenames:
            print(f"Processing {cond} baseline...")

            # Load baseline data
            temp = pd.read_pickle(
                join(path_baseline, f"{cond}_Baseline.pickle")
            )
            e.data_df = data_df
            e.data_df.loc[e.data_df["Condition"] == "Baseline"] = temp

            # Configure and window data
            e.windowed_name = f"windowed_{cond}"
            e.data_folder = path + "baselines_cond/"
            e.make_windowed(
                wnd_sec=[-100 * 1e-3, 600 * 1e-3],
                num_tr=num_tr,
                offset_samp_soundmock=26,
                mock_sound=False,  # Important to avoid creating ST in R peaks
                save=False,
            )

            # Trial rejection and normalization
            _, _, _, original = e.make_trial_rejection(
                thr_nan, thr_std, n_samp_rej, thr_rolling, rolling_tol
            )
            e.make_zScore()
            e.make_baseline_correction(on="pupil_right_norm")

            # Prepare and save data
            data_to_save = {
                "dataframes": {
                    "original": original,
                    "windowed": e.windowed,
                    "data_df": e.data_df,
                    "wnd": e.wnd,
                    "wnd_sec": e.wnd_sec,
                },
                "params": {
                    "num_tr": num_tr,
                    "fs": fs,
                    "thr_nan": thr_nan,
                    "thr_std": thr_std,
                    "n_samp_rej": n_samp_rej,
                    "thr_rolling": thr_rolling,
                    "rolling_tol": rolling_tol,
                },
            }

            with open(
                join(path_baseline, f"windowed_{cond}.pickle"), "wb"
            ) as f:
                pickle.dump(data_to_save, f)
            print(
                f"Data saved to {join(path_baseline, 'windowed_{cond}.pickle')}"
            )

    # Post-process each baseline condition
    process_type = "single"
    for cond in conds:
        if os.path.exists(
            path_baseline + f"/post_processed_{process_type}_{cond}.pickle"
        ):
            continue

        if os.path.exists(path_baseline + f"/windowed_{cond}.pickle"):
            print(f"Loading {cond} windowed data...")
            with open(path_baseline + f"/windowed_{cond}.pickle", "rb") as f:
                loaded_data = pickle.load(f)

        # Prepare data for analysis
        selected_data = loaded_data["dataframes"]["data_df"]
        selected_windowed = loaded_data["dataframes"]["windowed"]
        wnd = loaded_data["dataframes"]["wnd"]
        wnd_sec = loaded_data["dataframes"]["wnd_sec"]
        fs = loaded_data["params"]["fs"]

        # Align to first ST trigger
        first_trig_pos = (
            selected_windowed[selected_windowed.event_code == "ST"]
            .window_samp.head(1)
            .values[0]
        )
        selected_windowed["window_sec"] = (
            selected_windowed["window_samp"] - first_trig_pos
        ) / fs

        # Convert data types
        selected_data["event_block"] = selected_data["event_block"].astype(
            "category"
        )
        selected_windowed["event_block"] = selected_windowed[
            "event_block"
        ].astype("category")
        selected_windowed["trigger_counter"] = selected_windowed[
            "trigger_counter"
        ].astype("int16")
        selected_windowed = selected_windowed.query("trigger_counter <=300")

        # Clean trials
        dt = clean_trials(selected_windowed, process_type, e)

        # Reject outlier trials
        dt = reject_outlier_trials(dt, thr_across)

        # Calculate rejection statistics
        num_before = (
            loaded_data["dataframes"]["windowed"]
            .groupby(["subj_name", "event_block", "Condition"], observed=True)[
                "trigger_counter"
            ]
            .nunique()
            .sum()
        )
        num_actual = (
            dt.groupby(
                ["subj_name", "event_block", "Condition"], observed=True
            )["trigger_counter"]
            .nunique()
            .sum()
        )
        removed = num_before - num_actual
        print(
            f"Removed {removed} trials ({np.round(removed*100/num_before)}%)"
        )

        # Normalization and correction
        dt = normalize_and_correct(dt, process_type)

        # RR interval processing
        dt = process_rr_intervals(selected_data, dt)

        # Save post-processed data
        data_to_save = {
            "dataframes": {
                "windowed": dt,
                "wnd": wnd,
                "wnd_sec": wnd_sec,
            },
            "params": {
                "num_tr": num_tr,
                "fs": fs,
            },
        }

        with open(
            path_baseline + f"/post_processed_{process_type}_{cond}.pickle",
            "wb",
        ) as f:
            pickle.dump(data_to_save, f)
        print(
            f"Saved to {path_baseline}/post_processed_{process_type}_{cond}.pickle"
        )

    # Aggregate all baseline conditions
    baselines = []
    for cond in ["Sync", "Async", "Iso"]:
        print(f"Loading {cond} baseline...")
        temp = pd.read_pickle(
            path_baseline + f"/post_processed_single_{cond}.pickle"
        )["dataframes"]["windowed"]
        temp = temp.loc[temp["Condition"] == "Baseline"]
        temp["events_condition"] = cond
        baselines.append(temp)

    print("Combining all baselines...")
    baselines = pd.concat(baselines)
    baselines.to_pickle(
        path_baseline + "/post_processed_single_BASELINES.pickle"
    )
