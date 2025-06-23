"""
EyeTracker Analysis Module for Pupillometry Data Processing.

This module provides a class for analyzing eye-tracking data with ECG synchronization.
It includes methods for processing, windowing, and cleaning data, as well as tools for
R-peak detection, trial rejection, normalization, and statistical analysis.

Key Features:
- Data windowing around trigger events
- Automatic R-peak detection from ECG signals
- Trial rejection based on multiple criteria
- Baseline correction and normalization
- Support for monocular and binocular data

Usage Example:
```python
# Initialize analysis object
e = EyetrackerAnalysis(data_folder="/path/to/data", data_path="/path/to/data.pickle", fs=500)

# Load raw data
e.load_data(load_windowed=False)

# Create windowed data
e.make_windowed()

# Perform trial rejection
e.make_trial_rejection(thr_nan=1, thr_std=3, n_samp_rej=1, thr_rolling=30, rolling_tol=3)

# Apply z-score normalization
e.make_zScore()

# Apply baseline correction
e.make_baseline_correction(on="pupil_right_norm")
Dependencies:


numpy, pandas, wfdb, os, pickle

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
along with this program.  If not, see https://www.gnu.org/licenses/.

"""

import numpy as np
import pandas as pd
from os.path import join
from wfdb.processing import xqrs_detect, correct_peaks


class EyetrackerAnalysis:
    """
    A class for analyzing eyetracker data with ECG synchronization.

    This class provides methods for loading, preprocessing, windowing, and analyzing
    pupillometry data from eyetracker experiments. It includes functionality for
    R-peak detection, trial rejection, baseline correction, and normalization.

    Key Features:
    - Data windowing around trigger events
    - Automatic R-peak detection from ECG signals
    - Trial rejection based on multiple criteria
    - Baseline correction and normalization
    - Support for monocular data handling

    Attributes:
        data_folder (str): Path to the data folder
        data_path (str): Path to the pickle file containing raw data
        fs (int): Sampling frequency in Hz (default: 500)
        windowed_name (str): Name for windowed data files
        triggers (list): List of trigger codes to analyze
        data_df (pd.DataFrame): Raw data DataFrame
        windowed (pd.DataFrame): Windowed data DataFrame
        wnd (list): Window boundaries in samples [start, end]
        wnd_sec (list): Window boundaries in seconds [start, end]
        wnd_total_len (int): Total window length in samples
    """

    def __init__(
        self, data_folder, data_path=None, fs=500, windowed_name="windowed"
    ):
        """
        Initialize the EyetrackerAnalysis object.

        Args:
            data_folder (str): Path to the folder containing data files
            data_path (str, optional): Full path to the pickle file with raw data
            fs (int, optional): Sampling frequency in Hz. Defaults to 500.
            windowed_name (str, optional): Name prefix for windowed data files.
                                         Defaults to "windowed".

        Note:
            Data must be loaded using load_data() method before any analysis.
        """
        self.data_folder = data_folder
        self.data_path = data_path
        self.fs = fs
        self.windowed_name = windowed_name

    def load_data(self, triggers=["ST"], load_windowed=True):
        """
        Load raw data and optionally windowed data from pickle files.

        This method loads the main dataset and optionally pre-computed windowed data.
        If windowed data is not loaded, make_windowed() must be called subsequently.

        Args:
            triggers (list, optional): List of trigger codes to analyze.
                                     Defaults to ["ST"].
            load_windowed (bool, optional): Whether to load existing windowed data.
                                           Defaults to True.

        Raises:
            FileNotFoundError: If pickle files are not found in specified paths

        Note:
            The method prints summary information about loaded data and windowing parameters.
        """
        self.triggers = triggers

        # Load main dataset from pickle file
        self.data_df = pd.read_pickle(self.data_path)

        if load_windowed:
            # Load pre-computed windowed data
            self.windowed = pd.read_pickle(
                join(self.data_folder, "windowed.pickle")
            )

            # Calculate and store fundamental windowing parameters
            self._evaluate_wnd()

            print(
                f"""Loaded Windowed file around {self.triggers} with 
                {self.wnd} window samples, {self.wnd_sec} seconds
                ({self.wnd_total_len} in total)!"""
            )
        else:
            print(
                "Since load_windowed parameter is False, windowed must be created with make_windowed()"
            )

        print(f"Data:\n{self.data_df.head()}\n\n")

    def _evaluate_wnd(self):
        """
        Calculate window parameters from existing windowed data or user specifications.

        This private method determines window boundaries either from existing windowed
        data or converts user-specified time windows to sample indices based on
        the sampling frequency.

        The method sets:
        - self.wnd: Window boundaries in samples [before_trigger, after_trigger]
        - self.wnd_sec: Window boundaries in seconds [before_trigger, after_trigger]
        - self.wnd_total_len: Total window length in samples
        """
        if "wnd_sec" not in self.__dict__:
            # Extract window parameters from existing windowed data
            # Get the first trigger occurrence for reference
            first_trig = self.windowed.query(
                f"trigger_counter==0 & event_code=='{self.triggers[0]}'"
            ).iloc[0]

            # Extract window start position (negative value indicating pre-trigger samples)
            before = first_trig.window_samp

            # Get subject, condition, and block identifiers for consistent querying
            s = first_trig["subj_name"]
            c = first_trig["Condition"]
            b = first_trig["event_block"]

            # Find the end of the window for this trial
            end_first = self.windowed.query(
                f"trigger_counter==0 & subj_name==@s & event_block==@b & Condition==@c"
            )
            after = end_first["window_samp"].iloc[-1]

            # Set window boundaries in samples
            self.wnd = [-before, after]
            # Convert to seconds
            self.wnd_sec = [el / self.fs for el in self.wnd]
        else:
            # Convert user-specified time windows to sample indices
            self.wnd = [
                int(np.round(self.wnd_sec[0] * self.fs)),
                int(np.round(self.wnd_sec[1] * self.fs)),
            ]

        # Calculate total window length
        self.wnd_total_len = self.wnd[1] - self.wnd[0]

    def make_windowed(
        self,
        wnd_sec=[-100 * 1e-3, 600 * 1e-3],
        num_tr=1,
        offset_samp_soundmock=26,
        mock_sound=True,
        save=True,
    ):
        """
        Create windowed data around trigger events from continuous recordings.

        This method segments continuous data into trials based on trigger events,
        creating fixed-duration windows around each trigger. It handles monocular
        data, performs consistency checks, and optionally creates mock sound triggers
        for baseline conditions.

        Args:
            wnd_sec (list, optional): Window boundaries in seconds [start, end].
                                    Defaults to [-0.1, 0.6] (100ms pre, 600ms post).
            num_tr (int, optional): Number of consecutive triggers to treat as one trial.
                                   Defaults to 1.
            offset_samp_soundmock (int, optional): Sample offset for mock sound triggers
                                                 in baseline. Defaults to 26.
            mock_sound (bool, optional): Whether to create mock sound triggers for
                                       baseline condition. Defaults to True.
            save (bool, optional): Whether to save windowed data to pickle file.
                                 Defaults to True.

        Returns:
            tuple: (windowed_dataframe, window_samples, total_window_length, window_seconds)

        Note:
            The method skips trials where experimental conditions are not consistent
            across the entire window duration.
        """
        # Add mock sound triggers to baseline condition if requested
        if mock_sound:
            self._mock_sound_baseline(offset_samp=offset_samp_soundmock)

        # Store window parameters
        self.wnd_sec = wnd_sec
        wnd = [
            int(np.round(wnd_sec[0] * self.fs)),
            int(np.round(wnd_sec[1] * self.fs)),
        ]
        self.wnd = wnd

        # Create working copy of data
        data_df_reset = self.data_df.copy()

        # Handle monocular data by standardizing to pupil_right
        check, left_subjs = self.check_monocular(data_df_reset)
        if check:
            print(
                f"Treating monocular data as all derived from pupil_right; "
                f"copying pupil_left data in pupil_right for subjects {left_subjs}"
            )

            # Copy left pupil data to right pupil column for left-eye subjects
            mask = data_df_reset["subj_name"].isin(left_subjs)
            data_df_reset.loc[mask, "pupil_right"] = data_df_reset.loc[
                mask, "pupil_left"
            ]

            print("Dropping pupil_left column from windowed dataset")
            data_df_reset.drop(columns="pupil_left", inplace=True)

        # Find trigger indices for windowing
        # Using [pandas.pydata.org](https://pandas.pydata.org/docs/dev/user_guide/indexing.html) query method for efficient filtering
        indices = data_df_reset.query(f"event_code=={self.triggers}").index

        # Verify data integrity
        assert data_df_reset.index.is_unique
        assert data_df_reset.index.is_monotonic_increasing

        # Update main data reference
        self.data_df = data_df_reset

        # Initialize containers for windowed data
        dfs = []
        skipped_triggers = (
            []
        )  # Track skipped triggers due to inconsistent labels
        cnt = 0
        br = -1  # Debugging variable for breaking loop if needed

        # Process each trigger with specified step size
        for i, first_trig_index in self._enumerate2(indices, 0, num_tr):
            if br != -1:  # Debug break condition
                break

            cnt += 1
            # Progress indicator for large datasets
            if cnt % 10000 == 0:
                print(f"Evaluating trigger {cnt}/{int(len(indices)/num_tr)}")

            # Calculate window boundaries around trigger
            start_index = max(0, first_trig_index + wnd[0])

            try:
                # Find the last trigger in the trial group
                second_index = indices[i + num_tr - 1]
            except IndexError:
                # Skip if not enough triggers remaining
                continue

            end_index = min(len(data_df_reset), second_index + wnd[1])

            # Extract windowed data
            selected = data_df_reset.loc[start_index:end_index].copy()
            wnd_total_len = len(selected)
            self.wnd_total_len = wnd_total_len

            # Quality control: Check consistency of experimental conditions
            # All samples in the window must have the same subject, condition, and block
            condition_columns = ["subj_name", "Condition", "event_block"]
            if not (
                selected[condition_columns]
                .eq(selected[condition_columns].iloc[0])
                .all()
                .all()
            ):
                # Skip trials with inconsistent conditions
                skipped_triggers.append(first_trig_index)
                continue

            try:
                # Add relative time index within window
                selected.loc[:, "window_samp"] = range(0, wnd_total_len)
            except ValueError:
                # Handle edge case where window extends beyond data
                actual_length = len(selected)
                if actual_length < wnd_total_len:
                    selected.loc[:, "window_samp"] = range(0, actual_length)
                    # Mark trial as invalid by setting pupil data to NaN
                    selected.loc[:, "pupil_right"] = np.nan

            # Store processed window
            dfs.append(selected)

        print("\nConcatenating Dataframes ...")
        windowed = pd.concat(dfs)

        # Optimize data types for memory efficiency
        windowed["window_samp"] = windowed["window_samp"].astype("int16")

        # Create trial counter within each experimental condition
        windowed["trigger_counter"] = windowed.groupby(
            ["subj_name", "event_block", "Condition", "window_samp"],
            observed=True,
        ).cumcount()

        # Calculate relative time from trial start
        windowed["time_subj_trig"] = windowed["time"] - windowed.groupby(
            ["subj_name", "event_block", "Condition"]
        )["time"].transform("first")

        self.windowed = windowed

        # Save processed data if requested
        if save:
            windowed.to_pickle(
                join(self.data_folder, f"{self.windowed_name}.pickle")
            )
            print(f"{self.windowed_name}.pickle saved in {self.data_folder}")

        return self.windowed, self.wnd, wnd_total_len, self.wnd_sec

    def _mock_sound_baseline(self, offset_samp=26, save=False):
        """
        Generate mock sound triggers for baseline condition based on R-peaks.

        This private method creates artificial sound triggers in the baseline condition
        by detecting R-peaks in ECG data and placing triggers at a specified offset
        after each R-peak. This allows baseline data to be processed with the same
        windowing approach as experimental conditions.

        Args:
            offset_samp (int, optional): Number of samples after R-peak to place
                                       mock trigger. Defaults to 26 (52ms at 500Hz).
            save (bool, optional): Whether to save updated data to pickle file.
                                 Defaults to False.

        Note:
            This method modifies the event_code column for baseline condition data.
        """
        # Check if R-peaks have been detected previously
        if "r_peaks" in self.data_df.columns:
            print("Detected r_peak data")
        else:
            print("Not detected r_peak data, evaluating r peaks...")
            self._detect_rpeaks()

        # Create mock sound triggers offset from R-peaks in baseline condition
        mocked_snd_trigger = (
            self.data_df.query("Condition=='Baseline'")["r_peaks"]
            .map(
                {False: -1, True: "ST"}
            )  # Convert boolean R-peaks to trigger codes
            .shift(
                offset_samp, fill_value=-1
            )  # Shift timing to mimic sound delay
        )

        print(
            f"Pushing fake ST as Baseline event_code with {offset_samp} samples of offset"
        )

        # Update event_code for baseline condition
        baseline_mask = self.data_df.query("Condition=='Baseline'").index
        self.data_df.loc[baseline_mask, "event_code"] = mocked_snd_trigger

        if save:
            self.data_df.to_pickle(join(self.data_folder, "data.pickle"))
            print(f"data.pickle saved in {self.data_folder}")

    def _detect_rpeaks(self):
        """
        Detect R-peaks in ECG signals using WFDB algorithms.

        This private method processes ECG data to identify cardiac R-peaks using
        the XQRS detection algorithm followed by peak correction. R-peaks are
        detected separately for each subject, condition, and experimental block
        to ensure optimal detection parameters.

        The detected R-peaks are stored as boolean values in the 'r_peaks' column,
        where True indicates the presence of an R-peak at that sample.

        Note:
            This method modifies the data_df by adding an 'r_peaks' column and
            automatically saves the updated data to pickle file.
        """
        print("-- Detecting R Peaks --")

        # Initialize R-peaks column
        self.data_df.loc[:, "r_peaks"] = False
        current_subject = ""

        # Process each experimental segment separately
        for name, grp in self.data_df.groupby(
            ["subj_name", "Condition", "event_block"], observed=True
        ):
            # Progress indicator for subject processing
            if current_subject != grp["subj_name"].values[0]:
                print(f"Evaluating subject {grp['subj_name'].values[0]}")
                current_subject = grp["subj_name"].values[0]

            # Detect R-peaks using XQRS algorithm
            r_peaks = xqrs_detect(grp["ECG"].values, self.fs, verbose=False)

            # Correct and filter R-peak detections
            # Parameters: signal, peaks, search_radius=300samples, correct_width=50samples
            r_peaks = correct_peaks(grp["ECG"].values, r_peaks, 300, 50)
            r_peaks = r_peaks[r_peaks > 0]  # Remove invalid peak indices

            # Convert peak indices to boolean mask for the group
            self.data_df.loc[grp.index, "r_peaks"] = np.isin(
                np.arange(len(grp)), r_peaks
            )

        # Save updated dataframe with R-peak information
        self.data_df.to_pickle(join(self.data_folder, "data.pickle"))
        print(f"data.pickle saved in {self.data_folder}")

    def offset_estimation(self):
        """
        Estimate timing offset between triggers and R-peaks in synchronization trials.

        This method analyzes the 'Sync' condition to determine the temporal relationship
        between experimental triggers and physiological R-peaks. It calculates
        descriptive statistics (mean, median, SEM) for the offset distance.

        This information is crucial for understanding the temporal precision of
        the experimental setup and can guide the selection of appropriate offset
        values for mock sound generation.

        Prints:
            Statistical summary of R-peak distances from trigger onset
        """
        # Group synchronization trials by experimental parameters
        syncgroup = self.windowed.query("Condition=='Sync'").groupby(
            ["subj_name", "event_block", "trigger_counter"], observed=True
        )

        # Find absolute position of R-peaks and trial starts within each group
        abs_position_r_peak = syncgroup["r_peaks"].idxmax()
        abs_position_initial_grp = syncgroup["r_peaks"].idxmin()

        # Calculate distance from trial start to R-peak
        avg_r_peak_distance = abs_position_r_peak - abs_position_initial_grp

        # Generate descriptive statistics
        stats = avg_r_peak_distance.agg(["mean", "median", "sem"])
        print(stats)

    def check_monocular(self, data):
        """
        Determine if the dataset contains monocular recordings.

        This method analyzes pupil data availability for left and right eyes
        across all subjects to determine if the dataset is monocular (each subject
        recorded from only one eye) or binocular.

        Args:
            data (pd.DataFrame): DataFrame containing pupil_left and pupil_right columns

        Returns:
            tuple: (is_monocular, left_eye_subjects)
                - is_monocular (bool): True if dataset is fully monocular
                - left_eye_subjects (list or None): List of subjects recorded from left eye

        Note:
            Monocular datasets require data standardization where all pupil data
            is represented in the pupil_right column for consistent processing.
        """
        # Check data availability for each eye by subject
        right = data.groupby("subj_name", observed=True)["pupil_right"].any()

        try:
            left = data.groupby("subj_name", observed=True)["pupil_left"].any()
        except KeyError:
            print("No left pupil subject found")
            return False, None

        # Check if dataset is perfectly monocular (mutually exclusive eye recordings)
        if right.eq(~left).all():
            print("The dataset is monocular for all the subjects!")
            left_subjs = list(left[left].index.values)
            return True, left_subjs
        else:
            # Mixed monocular/binocular dataset
            print("The dataset is NOT monocular for all the subjects!")
            return False, None

    def make_trial_rejection(
        self, thr_nan=1, thr_std=3, n_samp_rej=1, thr_rolling=30, rolling_tol=3
    ):
        """
        Perform comprehensive trial rejection based on multiple quality criteria.

        This method implements a three-phase trial rejection pipeline:
        1. NaN threshold: Reject trials with excessive missing data
        2. Statistical outliers: Reject trials with extreme values (mean ± std)
        3. Rolling window outliers: Reject trials with localized artifacts

        Args:
            thr_nan (int, optional): Maximum allowed NaN samples per trial. Defaults to 1.
            thr_std (float, optional): Standard deviation multiplier for outlier detection.
                                     Defaults to 3.
            n_samp_rej (int, optional): Maximum allowed outlier samples per trial (phase 2).
                                      Defaults to 1.
            thr_rolling (int, optional): Rolling window size in samples. Defaults to 30.
            rolling_tol (int, optional): Maximum allowed outlier samples per trial (phase 3).
                                       Defaults to 3.

        Returns:
            tuple: (final_data, phase2_data, phase1_data, original_data)
                All DataFrames with progressively applied rejection criteria

        Note:
            The method prints detailed statistics about rejection rates at each phase
            and updates self.windowed with the final cleaned dataset.
        """
        # Store original data for comparison
        original = self.windowed.copy()

        # PHASE 1: NaN-based rejection
        # Remove trials exceeding NaN threshold using efficient groupby filtering
        windowed_after1 = self.windowed.groupby(
            ["Condition", "event_block", "subj_name", "trigger_counter"],
            observed=True,
            as_index=False,
        ).filter(lambda x: x["pupil_right"].isna().sum() <= thr_nan)

        # Calculate rejection statistics for Phase 1
        remaining_after1 = (
            windowed_after1.groupby(
                ["Condition", "event_block", "subj_name"], observed=True
            )["trigger_counter"]
            .unique()
            .apply(len)
        ).sum()

        initial_trials_num = (
            self.windowed.groupby(
                ["Condition", "event_block", "subj_name"], observed=True
            )["trigger_counter"]
            .unique()
            .apply(len)
        ).sum()

        deleted_after1 = initial_trials_num - remaining_after1

        print(
            f"""{deleted_after1} trials 
        ({np.round(deleted_after1*100/initial_trials_num,1)}%)
        have been rejected due to more than {thr_nan} Nan samples inside"""
        )

        # PHASE 2: Statistical outlier rejection
        # Calculate trial-wise mean and standard deviation
        m = windowed_after1.groupby(
            ["Condition", "event_block", "subj_name", "trigger_counter"],
            observed=True,
        )["pupil_right"].transform("mean")

        s = windowed_after1.groupby(
            ["Condition", "event_block", "subj_name", "trigger_counter"],
            observed=True,
        )["pupil_right"].transform("std")

        # Identify outlier samples beyond mean ± threshold*std
        above = windowed_after1["pupil_right"] > (m + thr_std * s)
        below = windowed_after1["pupil_right"] < (m - thr_std * s)
        idx_outside = above | below
        windowed_after1["idx"] = idx_outside

        # Filter trials based on outlier sample count
        windowed_after2 = windowed_after1.groupby(
            ["Condition", "event_block", "subj_name", "trigger_counter"],
            observed=True,
            as_index=False,
        ).filter(lambda x: x["idx"].sum() <= n_samp_rej)

        # Calculate rejection statistics for Phase 2
        remaining_after2 = (
            windowed_after2.groupby(
                ["Condition", "event_block", "subj_name"], observed=True
            )["trigger_counter"]
            .unique()
            .apply(len)
        ).sum()

        deleted_total = initial_trials_num - remaining_after2
        deleted_by2 = deleted_total - deleted_after1

        print(
            f"""{deleted_by2} trials 
        ({np.round(deleted_by2*100/initial_trials_num,1)}%)
        have been rejected due to more than {n_samp_rej} samples above or below
        mean + {thr_std}*std threshold in its trial"""
        )

        # PHASE 3: Rolling window outlier rejection
        # Calculate rolling statistics to detect localized artifacts
        windowed_after2["rolling_mean"] = windowed_after2.groupby(
            ["Condition", "event_block", "subj_name", "trigger_counter"],
            observed=True,
        )["pupil_right"].transform(lambda x: x.rolling(thr_rolling).mean())

        windowed_after2["rolling_std"] = windowed_after2.groupby(
            ["Condition", "event_block", "subj_name", "trigger_counter"],
            observed=True,
        )["pupil_right"].transform(lambda x: x.rolling(thr_rolling).std())

        # Identify outliers using rolling window statistics
        windowed_after2["outlier"] = (
            windowed_after2["pupil_right"]
            > windowed_after2["rolling_mean"]
            + thr_std * windowed_after2["rolling_std"]
        ) | (
            windowed_after2["pupil_right"]
            < windowed_after2["rolling_mean"]
            - thr_std * windowed_after2["rolling_std"]
        )

        # Count outliers per trial
        windowed_after2["num_outliers"] = windowed_after2.groupby(
            ["Condition", "event_block", "subj_name", "trigger_counter"],
            observed=True,
        )["outlier"].transform("sum")

        # Filter trials based on rolling window outlier count
        windowed_after3 = windowed_after2[
            windowed_after2["num_outliers"] < rolling_tol
        ]

        # Calculate rejection statistics for Phase 3
        remaining_after3 = (
            windowed_after3.groupby(
                ["Condition", "event_block", "subj_name"], observed=True
            )["trigger_counter"]
            .unique()
            .apply(len)
        ).sum()

        deleted_after3 = initial_trials_num - remaining_after3
        deleted_by3 = remaining_after2 - remaining_after3

        print(
            f"""{deleted_by3} trials 
        ({np.round(deleted_by3*100/initial_trials_num,1)}%)
        have been rejected with a {thr_rolling} samples rolling window due to more than 
        {rolling_tol} samples above or below mean + {thr_std}*std threshold in its trial"""
        )

        # Print comprehensive summary
        print(
            f""" A total of {deleted_after3} trials 
        ({np.round(deleted_after3*100/initial_trials_num,1)}%)
        have been rejected in the trial rejection phase out of 
        {initial_trials_num} initial trials, remaining with 
        {initial_trials_num-deleted_after3} trials"""
        )

        print("windowed has been updated rejecting noisy trials.")

        # Update main windowed dataset and clean up processing columns
        self.windowed = windowed_after3
        print("Purging useless columns used for processing")
        self.windowed = self.windowed.drop(
            columns=[
                "idx",
                "rolling_mean",
                "rolling_std",
                "outlier",
                "num_outliers",
            ],
        )

        return windowed_after3, windowed_after2, windowed_after1, original

    def make_zScore(self):
        """
        Apply z-score normalization to pupil data within experimental blocks.

        This method standardizes pupil diameter measurements by converting them
        to z-scores (standard deviations from the mean) within each subject's
        experimental block. This normalization accounts for individual differences
        in baseline pupil size and ensures comparable measurements across subjects.

        The z-score is calculated as: (value - block_mean) / block_std

        Returns:
            pd.DataFrame: Updated windowed DataFrame with normalized data

        Note:
            Creates a new column 'pupil_right_norm' while preserving original data.
            The normalization is performed separately for each subject and block.
        """
        # Calculate block-wise mean for each subject
        m = self.windowed.groupby(["subj_name", "event_block"], observed=True)[
            "pupil_right"
        ].transform(lambda x: x.mean())

        # Calculate block-wise standard deviation for each subject
        s = self.windowed.groupby(["subj_name", "event_block"], observed=True)[
            "pupil_right"
        ].transform(lambda x: x.std())

        # Apply z-score transformation
        self.windowed.loc[:, "pupil_right_norm"] = (
            self.windowed["pupil_right"] - m
        ) / s

        print("windowed has been updated performing zScore.")
        return self.windowed

    def make_deAverage(self):
        """
        Apply de-averaging to remove condition-specific baseline shifts.

        This method removes the mean response for each condition, subject, and block,
        effectively centering the data around zero. This is useful for isolating
        relative changes from condition-specific baseline differences.

        The de-averaged value is calculated as: value - condition_mean

        Returns:
            pd.DataFrame: Updated windowed DataFrame with de-averaged data

        Note:
            Creates a new column 'deaveraged' while preserving original data.
            The de-averaging is performed separately for each combination of
            subject, condition, and experimental block.
        """
        # Calculate condition-wise mean for each subject and block
        m = self.windowed.groupby(
            ["subj_name", "Condition", "event_block"], observed=True
        )["pupil_right"].transform(lambda x: x.mean())

        # Apply de-averaging transformation
        self.windowed.loc[:, "deaveraged"] = self.windowed["pupil_right"] - m

        print("windowed has been updated performing a de-averaging.")
        return self.windowed

    def make_baseline_correction(self, on="pupil_right_norm"):
        """
        Apply baseline correction using pre-trigger median values.

        This method corrects for trial-to-trial baseline variations by subtracting
        the median value from the pre-trigger period (baseline) from each sample
        in the trial. This ensures that all trials start from a common baseline.

        Args:
            on (str, optional): Column name to apply baseline correction to.
                              Defaults to "pupil_right_norm".

        Returns:
            pd.DataFrame: Updated windowed DataFrame with baseline-corrected data

        Note:
            Creates a new column with the pattern 'baseline_corrected_{abbreviation}'.
            The baseline period is defined as samples before the trigger event.
            Uses median for robust baseline estimation against outliers.
        """
        # Generate abbreviated column name for output
        abbr = (
            on
            if len(on.split("_")) == 1
            else "".join(word[0] for word in on.split("_"))
        )

        # Find trigger onset sample within trial window
        # Use first available trial as reference for trigger timing
        trigger_moment = self.windowed.query(
            f"subj_name=='{self.windowed.iloc[0].subj_name}' &  "
            f"Condition=='{self.windowed.iloc[0].Condition}' & "
            f"trigger_counter=={self.windowed.iloc[0].trigger_counter} & "
            f"event_block=={self.windowed.iloc[0].event_block} & "
            f"event_code=='{self.triggers[0]}'"
        )["window_samp"].array[0]

        # Calculate median baseline for each trial (pre-trigger period)
        medians = self.windowed.groupby(
            ["Condition", "event_block", "subj_name", "trigger_counter"],
            observed=True,
        )[on].transform(lambda x: x.head(trigger_moment).median())

        # Apply baseline correction
        self.windowed.loc[:, f"baseline_corrected_{abbr}"] = (
            self.windowed[on] - medians
        )

        print(
            f"windowed has been updated performing Baseline correction on {on}."
        )
        return self.windowed

    def get_conditions_names(self, df=None):
        """
        Extract unique condition names from the dataset.

        Args:
            df (pd.DataFrame, optional): DataFrame to extract conditions from.
                                       Defaults to self.windowed.

        Returns:
            list: Sorted list of unique condition names

        Note:
            This method assumes the DataFrame has a MultiIndex with conditions
            at level 1. For regular DataFrames, use df['Condition'].unique().
        """
        if df is None:
            df = self.windowed
        return df.index.get_level_values(1).unique().to_list()

    def get_subjects_names(self, df=None):
        """
        Extract unique subject identifiers from the dataset.

        Args:
            df (pd.DataFrame, optional): DataFrame to extract subjects from.
                                       Defaults to self.windowed.

        Returns:
            list: Sorted list of unique subject identifiers as integers

        Note:
            This method assumes the DataFrame has a MultiIndex with subjects
            at level 0 and converts string IDs to integers for proper sorting.
        """
        if df is None:
            df = self.windowed
        return sorted([int(el) for el in df.index.levels[0].to_list()])

    def remove(self, dictionary):
        """
        Filter data based on exclusion criteria.

        This method provides a flexible way to remove data based on multiple
        criteria specified in a dictionary format. It filters both the main
        dataset and windowed data simultaneously.

        Args:
            dictionary (dict): Dictionary where keys are column names and values
                             are lists of values to exclude from the analysis.

        Returns:
            tuple: (filtered_data_df, filtered_windowed_df)
                Both DataFrames with specified values excluded

        Example:
            # Remove specific subjects and conditions
            filtered = remove({
                'subj_name': ['s1', 's3'],
                'Condition': ['Baseline']
            })
        """
        # Build query string for filtering
        conditions = []
        for field, val in dictionary.items():
            condition = f"{field} not in {val}"
            conditions.append(condition)

        query_string = " & ".join(conditions)

        # Apply filtering to both datasets
        filtered_windowed = self.windowed.query(query_string)
        filtered_data = self.data_df.query(query_string)

        return filtered_data, filtered_windowed

    def _enumerate2(self, xs, start=0, step=1):
        """
        Custom enumeration function with configurable step size.

        This private method provides enumeration functionality similar to
        Python's built-in enumerate(), but allows for custom step sizes.
        This is useful for processing every nth trigger or skipping trials.

        Args:
            xs (sequence): Sequence to enumerate
            start (int, optional): Starting index. Defaults to 0.
            step (int, optional): Step size between iterations. Defaults to 1.

        Yields:
            tuple: (index, value) pairs following the specified step pattern

        Example:
            # Process every 2nd trigger starting from index 0
            for i, trigger_idx in _enumerate2(trigger_indices, 0, 2):
                # Process trigger_idx...
        """
        for i in range(start, len(xs), step):
            yield (i, xs[i])


if __name__ == "__main__":
    """
    Example usage demonstrating basic workflow for eyetracker analysis.

    This example shows how to:
    1. Initialize the analysis object with data paths
    2. Load raw data without pre-computed windowed data
    3. Create windowed data around trigger events
    4. Calculate windowing parameters

    Modify the data_folder and data_path to match your specific setup.
    """
    # Initialize analysis object with data paths and sampling frequency
    e = EyetrackerAnalysis(
        data_folder="/mnt/HDD2/giovanni/data/agg",
        data_path="/mnt/HDD2/giovanni/data/agg/data.pickle",
        fs=500,  # 500 Hz sampling frequency
    )

    # Load raw data without pre-computed windowed data
    e.load_data(load_windowed=False)

    # Create windowed data around trigger events
    e.make_windowed()

    # Calculate and display windowing parameters
    e._evaluate_wnd()
