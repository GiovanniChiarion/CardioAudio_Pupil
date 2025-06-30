"""
Eye-tracking Data Processing and Analysis.

This script provides a class for processing and analyzing eye-tracking data from Pupil Labs eye-trackers. It includes methods for modifying ASC files, processing raw data, handling ECG synchronization, and concatenating datasets.

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

import glob
import re
import numpy as np
import pandas as pd
import mne
import mat73
import os

# To read the date written in the .asc file, otherwise throws an error
import locale

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


class MyEyetracker:

    def __init__(self, path, subj_name="all"):
        self.general_dirpath = path
        self.subj_name = subj_name  # string of subj_folder or list of strings or "all", if "all" overwrited to a list of strings by self._set_fullpath()
        # -- PARAMETERS --

        self.interpolate_blinks = True
        self.blink_max_duration_sec = 0.3

        # ----------------

        self.fullpath_raw = self._set_fullpath()
        self.path = [
            os.path.join(self.general_dirpath, f"{subj_name}")
            for subj_name in self.subj_name
        ]

        # --------------------------------------------------------------
        # -- Comment the following lines to perform concat operation --
        # self.fullpath_new_asc = self._modify_asc()
        # --------------------------------------------------------------

    def _modify_asc(self, write_file=True):
        """
        Modifies the contents of an ASC file by replacing specific codes with their corresponding values.

        Args:
            write_file (bool, optional): Whether to write the modified contents to a new file. Defaults to True.

        Returns:
            None

        Examples:
            >>> _modify_asc(write_file=True)
            Conversion done!
            'Modified file contents'
        """
        codes = {
            16: "ST",  # Sound Trigger 16
            32: "ST",  # Sound Trigger 32
            96: "SY1",  # Synchronized Phase Start
            112: "SY2",  # Synchronized Phase Stop
            160: "A1",  # Asynchronized Phase Start
            176: "A2",  # Asynchronized Phase Stop
            128: "I1",  # Isochronous Phase Start
            144: "I2",  # Isochronous Phase Stop
            192: "B1",  # Baseline Phase Start
            208: "B2",  # Baseline Phase Stop
        }
        self.codes_numstr = codes
        self.codes_strnum = {
            "ST": 16,  # Sound Trigger 16 or 32
            "SY1": 96,  # Synchronized Phase Start
            "SY2": 112,  # Synchronized Phase Stop
            "A1": 160,  # Asynchronized Phase Start
            "A2": 176,  # Asynchronized Phase Stop
            "I1": 128,  # Isochronous Phase Start
            "I2": 144,  # Isochronous Phase Stop
            "B1": 192,  # Baseline Phase Start
            "B2": 208,  # Baseline Phase Stop
        }

        pattern = r"(MSG.*?){}(\n)"
        fullpath_new_asc = []

        not_modified_paths = self._check_mod_asc()

        for i, path in enumerate(self.fullpath_raw):
            if path in not_modified_paths:
                name = glob.glob(f"{path}/*[!new].asc")
                name = name[0]
                fullpath_new_asc.append(f"{name[:-4]}_new.asc")
                with open(name, "r") as f:
                    file = f.read()
                for code in codes:
                    pt = pattern.format(code)
                    file = re.sub(pt, f"\g<1>{codes[code]}\g<2>", file)
                if write_file:
                    with open(fullpath_new_asc[i], "w") as f:
                        f.write(file)
                print(f"Conversion of subject {self.subj_name[i]} done!")
            else:
                name = glob.glob(f"{path}/*new.asc")
                fullpath_new_asc.append(
                    name[0]
                )  # Added 0 because it was a list 24/04/2024
        return fullpath_new_asc

    def data_process_withECG(self, remove=None, write_file=False):
        """
        Processes the data from an ASC file by performing various operations such as removing channels, renaming annotations,
        removing specific annotations, checking calibration, interpolating blinks, generating events, and saving dataframes.

        Args:
            remove (list, optional): List of annotations to remove. Defaults to None.
            write_file (bool, optional): Whether to write the processed data to files. Defaults to False.

        Returns:
            pd.DataFrame: The processed data.

        Examples:
            >>> data_process(remove=["NA"], write_file=True)
            'Processed data dataframe'
        """
        if remove is None:
            remove = ["NA", "errors"]

        for cnt, path_name_new_asc in enumerate(self.fullpath_new_asc):
            path = self.path[cnt]
            raw_data = mne.io.read_raw_eyelink(path_name_new_asc, preload=True)
            self.fs = raw_data.info["sfreq"]

            # Removing DIN channel
            try:
                raw_data = raw_data.drop_channels("DIN")
            except:
                print("Channel(s) DIN not found, nothing dropped.")

            # CHANGE ANNOTATION NAMES
            annotations = raw_data.annotations
            if "saccade" in list(annotations.count()):
                annotations.rename({"fixation": "F", "saccade": "S"})
            if "" in list(annotations.count()):
                annotations.rename({"": "NA"})

            # REMOVE SOME ANNOTATIONS
            if "errors" in remove:
                remove_ids = [
                    i
                    for i, a in enumerate(annotations)
                    if a["description"]
                    in [
                        "ERROR MESSAGES LOST 65234",
                        "ERROR MESSAGES LOST 65535",
                    ]
                ]
                annotations.delete(remove_ids)
            if "NA" in remove:
                remove_ids = [
                    i
                    for i, a in enumerate(annotations)
                    if a["description"] == "NA"
                ]
                annotations.delete(remove_ids)

            # Finding BAD BLINK annotations with duration above blink_max_duration not interpolatable
            missing_indices = [
                i
                for i, el in enumerate(annotations)
                # Check duration
                if (
                    el["description"] == "BAD_blink"
                    and el["duration"] > self.blink_max_duration_sec
                )
            ]

            # Deleting annotations
            annotations.delete(missing_indices)
            # Setting new annotations
            raw_data.set_annotations(annotations)

            if self.interpolate_blinks:
                # Interpolating BAD BLINKS converting annotations in "blink"
                raw_data = mne.preprocessing.eyetracking.interpolate_blinks(
                    raw_data, buffer=(0.05, 0.2)
                )
            else:
                # Changing real blinks BAD_blink annotations in "blink"
                annotations.rename({"BAD_blink": "blink"})
                # Setting new annotations
                raw_data.set_annotations(annotations)
            raw_eye = raw_data

            # Loading ECG dataset
            raw_ecg = self.load_ecg(path, fs=1024)

            # Cropping datasets into Conditions
            crop_eye, ann_crop_eye, crop_ecg, ann_crop_ecg = (
                self.cropping_datasets(raw_eye, raw_ecg)
            )

            # Aligning datasets
            aligned = self.align_eye_ecg(
                crop_eye, ann_crop_eye, crop_ecg, ann_crop_ecg
            )

            data_df = self.concatenate_conditions(aligned, ann_crop_eye)

            # Works for only monocular acquisitions!
            pupil_col_name = [
                el for el in data_df.columns if el.startswith("pupil")
            ][0]
            
            # Replace 0 values with NaN
            data_df[pupil_col_name] = data_df[pupil_col_name].mask(
                data_df[pupil_col_name] == 0
            )

            # CONVERTING TO BETTER DTYPES TO INCREASE PERFORMANCES
            data_df.event_code = data_df.event_code.astype("category")
            data_df.event_block = data_df.event_block.astype("category")
            data_df.Condition = data_df.Condition.astype("category")

            data_df.to_pickle(f"{path}/data.pickle")

            #  -- Testing that the dataframe have the same content. -- 
            # import plotly.express as px
            # # Load your DataFrame
            # a = pd.read_pickle(f"{path}/data.pickle")
            # # a=data_df
            # # Filter your DataFrame
            # b = a.query("Condition=='Sync' & event_block==0")

            # # Create the initial line plot for 'pupil_right'
            # fig = px.line(b, x='rel_time', y='pupil_right', labels={'y': 'pupil_right'}, title="Pupil and ECG over Time")

            # # Add the second line plot for 'ECG'
            # fig.add_scatter(x=b['rel_time'], y=b['ECG'], mode='lines', name='ECG')

            # # Add multiple vertical lines
            # for x_value in b.query("event_code=='ST'")["rel_time"].values:
            #     fig.add_vline(x=x_value, line_width=1, line_dash="solid", line_color="red")

            # # Show the plot
            # fig.show(renderer='firefox')

            # data_df.to_hdf(f"{path}/data.hdf",format="table",key="data",complevel=9)

    def load_ecg(self, path, fs):
        selection_mask_string = list(set(self.codes_numstr.values()))
        selection_mask_codes = [
            self.codes_strnum[el]
            for el in self.codes_strnum
            if el in selection_mask_string
        ]
        mat = mat73.loadmat(os.path.join(path, "raw_data.mat"))

        # Extract rows 65 e 69
        ecg_signal = mat["y"][
            64, :
        ]  # Remembering that the index is 64 pfor the row 65 (matlab)
        triggers_ecg = mat["y"][
            68, :
        ]  # Remembering that the index is 68 for the row 69

        # All 32 triggers must be converted in 16 because of dictionary inversion
        triggers_ecg = [16 if el == 32 else el for el in triggers_ecg]

        # Select only triggers in selection_mask
        triggers_ecg = [
            el if el in selection_mask_codes else 0 for el in triggers_ecg
        ]

        info = mne.create_info(["ECG", "STIM"], fs, ["ecg", "misc"])
        raw_ecg = mne.io.RawArray(np.stack([ecg_signal, triggers_ecg]), info)
        return raw_ecg

    def cropping_datasets(self, raw_eye, raw_ecg):
        ann_crop_eye = raw_eye.annotations.copy()
        events_ecg = mne.find_events(raw_ecg, "STIM")
        ann_crop_ecg = mne.annotations_from_events(
            events_ecg, raw_ecg.info["sfreq"], self.codes_numstr
        )

        annotations_crop = {"eye": ann_crop_eye, "ecg": ann_crop_ecg}

        conditions = [el for el in self.codes_numstr.values() if el != "ST"]

        for i, ann_key in enumerate(annotations_crop):
            ann = annotations_crop[ann_key]
            print(f"Cropping dataset {ann_key}")
            idx = [
                i
                for i, el in enumerate(ann)
                if el["description"] not in conditions
            ]
            ann.delete(idx)

            # Set proper duration in annotations
            for j in range(0, len(ann), 2):
                d1 = ann.description[j]
                d2 = ann.description[j + 1]
                # Checking that the two desctipion names coincides
                assert (
                    d1[:-1] == d2[:-1]
                ), f"The first conditions seems not to be Baseline on {ann_key}: first is {d1}, second is {d2}"
                # Checking that the first is a starting condition and the second an end
                assert (
                    d1[-1] == "1"
                ), f"The first conditions seems not to be Baseline on {ann_key}: first is {d1}, second is {d2}"
                assert (
                    d2[-1] == "2"
                ), f"The first conditions seems not to be Baseline on {ann_key}: first is {d1}, second is {d2}"
                # Setting proper duration to the first annotation
                ann.duration[j] = ann.onset[j + 1] - ann.onset[j]
            # Deleting ending condition annotations
            ann.delete(range(1, len(ann), 2))

        # Cropping by annotations duration
        crop_eye = raw_eye.copy().crop_by_annotations(annotations_crop["eye"])
        crop_ecg = raw_ecg.copy().crop_by_annotations(annotations_crop["ecg"])
        ann_crop_eye = annotations_crop["eye"]
        ann_crop_ecg = annotations_crop["ecg"]

        # check if first crop is Baseline
        assert ann_crop_eye.description[0] == "B1"
        assert ann_crop_ecg.description[0] == "B1"

        # Check if the annotations both coincide
        assert all(ann_crop_eye.description == ann_crop_ecg.description)

        return crop_eye, ann_crop_eye, crop_ecg, ann_crop_ecg

    def align_eye_ecg(self, crop_eye, ann_crop_eye, crop_ecg, ann_crop_ecg):
        blocks = list(enumerate(ann_crop_eye.description))

        for i in range(len(crop_eye)):
            print(
                f"Evaluating condition number {i+1}/{len(crop_eye)}\n\n : {blocks[i][1]}"
            )
            a = crop_eye[i]  # .copy()
            b = crop_ecg[i]  # .copy()

            start_time = max(a.times[0], b.times[0])
            end_time = min(a.times[-1], b.times[-1])

            a.crop(tmin=start_time, tmax=end_time)
            b.crop(tmin=start_time, tmax=end_time)

            b.resample(sfreq=a.info["sfreq"], stim_picks="STIM")

            # If there is still a mismatch, you can crop the resampled data to match the number of samples
            if a.n_times != b.n_times:
                min_samples = min(a.n_times, b.n_times)
                a = a.crop(tmax=(min_samples - 1) / a.info["sfreq"])
                b = b.crop(tmax=(min_samples - 1) / b.info["sfreq"])

            if blocks[i][1] != "B1":
                events1, events1_dict = mne.events_from_annotations(
                    a, event_id=self.codes_strnum
                )

            a.add_channels([b], force_update_info=True)

        return crop_eye

    def concatenate_conditions(self, aligned, ann_crop_eye):
        # At the moment annotations are not taken in consideration.
        # in the future, they need to be used for analysis of
        # fixations, saccades and blinks
        aligned_df = [el.to_data_frame() for el in aligned]
        # Updating original time based on annotations
        assert len(aligned_df) == len(ann_crop_eye)
        for i in range(len(aligned_df)):
            start_time = ann_crop_eye[i]["onset"]
            aligned_df[i]["time"] = np.linspace(
                start_time,
                (len(aligned_df[i]) - 1) / self.fs + start_time,
                len(aligned_df[i]),
            )

        blocks = list(ann_crop_eye.description)
        conditions = list(set(blocks))
        concatenated = []
        mapping = {"A1": "Async", "B1": "Baseline", "SY1": "Sync", "I1": "Iso"}
        for cond in conditions:
            cond_dfs = [
                aligned_df[i] for i, el in enumerate(blocks) if el == cond
            ]
            for i, c in enumerate(cond_dfs):
                c["event_block"] = i
                c["rel_time"] = np.arange(len(c)) / self.fs
                c["STIM"] = c["STIM"].replace([0, 16], [-1, "ST"])
                c["STIM"] = c["STIM"].apply(
                    lambda x: -1 if isinstance(x, (int, float)) else x
                )
            agg = pd.concat(cond_dfs, ignore_index=True)
            agg["Condition"] = mapping[cond]
            concatenated.append(agg)
        full_dataframe = pd.concat(concatenated, ignore_index=True)
        full_dataframe.rename(columns={"STIM": "event_code"}, inplace=True)
        return full_dataframe

    def _check_mod_asc(self):
        not_modified_paths = []
        for path in self.fullpath_raw:
            name = glob.glob(f"{path}/*_new.asc")
            if len(name) != 1:
                not_modified_paths.append(path)
        return not_modified_paths

    def _set_fullpath(self):
        if isinstance(self.subj_name, int):
            raise ValueError("Integer number as subj_name are not allowed")
        elif isinstance(self.subj_name, list):
            for el in self.subj_name:
                if isinstance(el, int):
                    raise ValueError(
                        "Integer number as subj_name are not allowed"
                    )
            return [f"{self.general_dirpath}/{el}" for el in self.subj_name]

        elif self.subj_name == "all":
            self.subj_name = [
                d
                for d in os.listdir(self.general_dirpath)
                if os.path.isdir(os.path.join(self.general_dirpath, d))
                and d != "aggregated"
            ]
            return [f"{self.general_dirpath}/{el}" for el in self.subj_name]

        elif isinstance(self.subj_name, str):
            self.subj_name = [self.subj_name]
            return [f"{self.general_dirpath}/{self.subj_name[0]}"]
        else:
            raise (
                TypeError,
                "subj_name should be the subject folder name or a list or 'all'!",
            )

    def _divide_conditions(
        self, raw_data, cond_extremes, events, events_dict, FSBlink, ST
    ):
        # Selecting Baseline Phases
        ev = mne.pick_events(
            events,
            include=[
                events_dict[cond_extremes[0]],
                events_dict[cond_extremes[1]],
            ],
        )

        # Reshaping
        ev = [el[0] for el in ev]
        ev = np.array(ev).reshape((-1, 2))
        ev_df = pd.DataFrame(ev, columns=(("Start"), ("Stop")))

        # Dividing in epochs
        dfs = []
        for i in range(len(ev_df)):
            dfs.append(
                raw_data.iloc[ev_df.iloc[i]["Start"] : ev_df.iloc[i]["Stop"]]
            )
            dfs[i] = dfs[i].copy()
            dfs[i].loc[:, "event_block"] = i
            dfs[i].loc[:, "rel_time"] = np.arange(len(dfs[i])) / self.fs

        df_merged = pd.concat(dfs)

        # Inserting Instantaneous Events
        cnt = 0
        df_merged["event_code"] = -1
        df_merged["event_code"] = df_merged["event_code"].astype("object")
        for i, ev in FSBlink.iterrows():
            if (s := ev["Sample"]) in df_merged.index:
                cnt += 1
                df_merged.loc[s, "event_code"] = ev["event_code"]
        print(f"Added {cnt} instantaneous events!")

        # Inserting Sound Trigger Events
        cnt = 0
        for i, ev in ST.iterrows():
            if (s := ev["Sample"]) in df_merged.index:
                cnt += 1
                df_merged.loc[s, "event_code"] = ev["event_code"]
        print(f"Added {cnt} trigger events!")

        df_merged["event_code"] = df_merged["event_code"].astype("category")
        return df_merged

    def concat_subject_dfs(self, out_folder_path):
        dir_paths, dir_names = zip(
            *[
                [el.path, el.name]
                for el in os.scandir(self.general_dirpath)
                if el.name in self.subj_name
            ]
        )

        print(f"Starting loading of {len(dir_paths)} subject dataframes\n")
        d = []
        for i, p in enumerate(dir_paths):
            print(f"Loading subject: {dir_names[i]}")
            name = glob.glob(p + "/data.pickle")[0]
            temp = pd.read_pickle(name)
            temp["subj_name"] = dir_names[i]
            d.append(temp)
        print("Concatenating all the subjects ...\n")
        df = pd.concat(
            d, ignore_index=True
        )  # ignore_index let us not having multiple index overlapping
        if not os.path.exists(out_folder_path):
            print(f"{out_folder_path} not found, creating it ...\n")
            os.mkdir(out_folder_path)
        df.reset_index()
        df.to_pickle(os.path.join(out_folder_path, "data.pickle"))
        print(
            f"Concatenated dataframe saved in {os.path.join(out_folder_path,'data.pickle')}"
        )