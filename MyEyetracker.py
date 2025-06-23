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
import locale

# Set locale to handle date formats in ASC files
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


class MyEyetracker:
    """
    A class for processing and analyzing eye-tracking data.

    This class provides methods for modifying ASC files, processing raw data,
    handling ECG synchronization, and concatenating datasets.

    Attributes:
        general_dirpath (str): Path to the directory containing subject data.
        subj_name (str or list): Subject name(s) to process. Can be "all" to process all subjects.
        interpolate_blinks (bool): Whether to interpolate blinks. Defaults to True.
        blink_max_duration_sec (float): Maximum duration of blinks to interpolate. Defaults to 0.3.
        fullpath_raw (list): List of paths to raw data files.
        path (list): List of paths to subject directories.
    """

    def __init__(self, path, subj_name="all"):
        """
        Initialize the MyEyetracker object.

        Args:
            path (str): Path to the directory containing subject data.
            subj_name (str or list): Subject name(s) to process. Can be "all" to process all subjects.
        """
        self.general_dirpath = path
        self.subj_name = subj_name  # Can be a string, list of strings, or "all"
        self.interpolate_blinks = True
        self.blink_max_duration_sec = 0.3
        self.fullpath_raw = self._set_fullpath()
        self.path = [
            os.path.join(self.general_dirpath, f"{subj_name}")
            for subj_name in self.subj_name
        ]

    def _modify_asc(self, write_file=True):
        """
        Modify ASC files by replacing specific codes with their corresponding values.

        Args:
            write_file (bool): Whether to write the modified contents to a new file. Defaults to True.

        Returns:
            list: List of paths to the modified ASC files.
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
        self.codes_strnum = {v: k for k, v in codes.items()}

        pattern = r"(MSG.*?){}(\n)"
        fullpath_new_asc = []

        not_modified_paths = self._check_mod_asc()

        for i, path in enumerate(self.fullpath_raw):
            if path in not_modified_paths:
                name = glob.glob(f"{path}/*[!new].asc")[0]
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
                name = glob.glob(f"{path}/*new.asc")[0]
                fullpath_new_asc.append(name)
        return fullpath_new_asc

    def data_process(self, remove=None, write_file=False):
        """
        Process eye-tracking data from ASC files.

        Args:
            remove (list): List of annotations to remove. Defaults to ["NA", "errors"].
            write_file (bool): Whether to write the processed data to files. Defaults to False.

        Returns:
            pd.DataFrame: Processed data.
        """
        if remove is None:
            remove = ["NA", "errors"]

        for cnt, path_name_new_asc in enumerate(self.fullpath_new_asc):
            path = self.path[cnt]
            raw_data = mne.io.read_raw_eyelink(path_name_new_asc, preload=True)
            self.fs = raw_data.info["sfreq"]

            # Remove DIN channel if present
            try:
                raw_data = raw_data.drop_channels("DIN")
            except:
                print("Channel(s) DIN not found, nothing dropped.")

            # Rename annotations
            annotations = raw_data.annotations
            if "saccade" in list(annotations.count()):
                annotations.rename({"fixation": "F", "saccade": "S"})
            if "" in list(annotations.count()):
                annotations.rename({"": "NA"})

            # Remove specified annotations
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

            # Interpolate blinks if enabled
            if self.interpolate_blinks:
                raw_data = mne.preprocessing.eyetracking.interpolate_blinks(
                    raw_data, buffer=(0.05, 0.2)
                )
            else:
                annotations.rename({"BAD_blink": "blink"})
                raw_data.set_annotations(annotations)

            # Generate events and save dataframes
            events, events_dict = mne.events_from_annotations(
                raw_data, regexp="^(?![Bb][Aa][Dd]|[Ee][Dd][Gg][Ee]).*$"
            )
            raw_data = raw_data.to_data_frame()
            pupil_col_name = [
                el for el in raw_data.columns if el.startswith("pupil")
            ][0]
            raw_data[pupil_col_name] = raw_data[pupil_col_name].mask(
                raw_data[pupil_col_name] == 0
            )

            if write_file:
                pd.DataFrame(events).to_csv(f"{path}/events.csv", header=False)
                pd.DataFrame(events_dict, index=[0]).to_csv(
                    f"{path}/events_dict.csv", index=False
                )
                raw_data.to_csv(f"{path}/raw.csv")

            # Process ST events
            ST = mne.pick_events(events, include=[events_dict["ST"]])
            ST = [[el[0], "ST"] for el in ST]
            ST = np.array(ST).reshape((-1, 2))
            ST = pd.DataFrame(ST, columns=("Sample", "event_code"))
            offset = 4.88 * 1e-3  # seconds
            ST["Sample"] += int(np.round(offset * self.fs))

            # Process fixation, saccade, and blink events
            FSBlink = mne.pick_events(
                events,
                include=[
                    events_dict["F"],
                    events_dict["S"],
                    events_dict["blink"],
                ],
            )
            reversed_dict = {v: k for k, v in events_dict.items()}
            FSBlink = [[el[0], reversed_dict[el[2]]] for el in FSBlink]
            FSBlink = np.array(FSBlink).reshape((-1, 2))
            FSBlink = pd.DataFrame(FSBlink, columns=("Sample", "event_code"))

            # Divide data into conditions
            cond_couples = {
                "SY": ("SY1", "SY2"),
                "A": ("A1", "A2"),
                "B": ("B1", "B2"),
                "I": ("I1", "I2"),
            }
            B1B2_merged = self._divide_conditions(
                raw_data, cond_couples["B"], events, events_dict, FSBlink, ST
            )
            SY1SY2_merged = self._divide_conditions(
                raw_data, cond_couples["SY"], events, events_dict, FSBlink, ST
            )
            A1A2_merged = self._divide_conditions(
                raw_data, cond_couples["A"], events, events_dict, FSBlink, ST
            )
            I1I2_merged = self._divide_conditions(
                raw_data, cond_couples["I"], events, events_dict, FSBlink, ST
            )

            # Concatenate all conditions
            data_df = pd.concat(
                [B1B2_merged, SY1SY2_merged, A1A2_merged, I1I2_merged],
                keys=("Baseline", "Sync", "Async", "Iso"),
                names=("Condition", "Sample"),
            )
            data_df = data_df.reset_index()

            # Convert to efficient data types
            data_df.event_code = data_df.event_code.astype("category")
            data_df.event_block = data_df.event_block.astype("category")
            data_df.Condition = data_df.Condition.astype("category")
            data_df.Sample = data_df.Sample.astype("int32")

            # Save processed data
            data_df.to_pickle(f"{path}/data.pickle")
            pd.DataFrame(events_dict, index=[0]).to_csv(
                f"{path}/events_dict.csv", index=False
            )

    def _check_mod_asc(self):
        """
        Check which ASC files have not been modified.

        Returns:
            list: List of paths to unmodified ASC files.
        """
        not_modified_paths = []
        for path in self.fullpath_raw:
            name = glob.glob(f"{path}/*_new.asc")
            if len(name) != 1:
                not_modified_paths.append(path)
        return not_modified_paths

    def _set_fullpath(self):
        """
        Set the full paths to the raw data files based on subject names.

        Returns:
            list: List of paths to raw data files.
        """
        if isinstance(self.subj_name, int):
            raise ValueError("Integer number as subj_name is not allowed")
        elif isinstance(self.subj_name, list):
            for el in self.subj_name:
                if isinstance(el, int):
                    raise ValueError(
                        "Integer number as subj_name is not allowed"
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
            raise TypeError(
                "subj_name should be the subject folder name, a list, or 'all'!"
            )

    def _divide_conditions(
        self, raw_data, cond_extremes, events, events_dict, FSBlink, ST
    ):
        """
        Divide data into conditions based on event markers.

        Args:
            raw_data (pd.DataFrame): Raw data to divide.
            cond_extremes (tuple): Tuple of start and end markers for the condition.
            events (np.array): Events array.
            events_dict (dict): Dictionary mapping event names to IDs.
            FSBlink (pd.DataFrame): DataFrame of fixation, saccade, and blink events.
            ST (pd.DataFrame): DataFrame of sound trigger events.

        Returns:
            pd.DataFrame: Data divided into the specified condition.
        """
        ev = mne.pick_events(
            events,
            include=[
                events_dict[cond_extremes[0]],
                events_dict[cond_extremes[1]],
            ],
        )
        ev = [el[0] for el in ev]
        ev = np.array(ev).reshape((-1, 2))
        ev_df = pd.DataFrame(ev, columns=("Start", "Stop"))

        dfs = []
        for i in range(len(ev_df)):
            dfs.append(
                raw_data.iloc[ev_df.iloc[i]["Start"] : ev_df.iloc[i]["Stop"]]
            )
            dfs[i] = dfs[i].copy()
            dfs[i].loc[:, "event_block"] = i
            dfs[i].loc[:, "rel_time"] = np.arange(len(dfs[i])) / self.fs

        df_merged = pd.concat(dfs)

        # Insert instantaneous events
        cnt = 0
        df_merged["event_code"] = -1
        df_merged["event_code"] = df_merged["event_code"].astype("object")
        for i, ev in FSBlink.iterrows():
            if (s := ev["Sample"]) in df_merged.index:
                cnt += 1
                df_merged.loc[s, "event_code"] = ev["event_code"]
        print(f"Added {cnt} instantaneous events!")

        # Insert sound trigger events
        cnt = 0
        for i, ev in ST.iterrows():
            if (s := ev["Sample"]) in df_merged.index:
                cnt += 1
                df_merged.loc[s, "event_code"] = ev["event_code"]
        print(f"Added {cnt} trigger events!")

        df_merged["event_code"] = df_merged["event_code"].astype("category")
        return df_merged

    def concat_subject_dfs(self, out_folder_path):
        """
        Concatenate dataframes from multiple subjects.

        Args:
            out_folder_path (str): Path to save the concatenated dataframe.
        """
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
        df = pd.concat(d, ignore_index=True)
        if not os.path.exists(out_folder_path):
            print(f"{out_folder_path} not found, creating it ...\n")
            os.mkdir(out_folder_path)
        df.reset_index()
        df.to_pickle(os.path.join(out_folder_path, "data.pickle"))
        print(
            f"Concatenated dataframe saved in {os.path.join(out_folder_path,'data.pickle')}"
        )
