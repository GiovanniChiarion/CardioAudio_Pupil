# Eye-tracking Data Analysis Pipeline  
**Version: 1.0**  
**Author: Giovanni Chiarion**  
**Contact:** chiarion.giovanni@gmail.com  

This project provides a robust pipeline for preprocessing and analyzing eye-tracking data from Pupil Labs eye-trackers. It includes tools for data loading, windowing, trial rejection, normalization, RR interval analysis, and dataset construction. The pipeline is designed to handle both single-trial and multi-trial processing modes, making it suitable for a wide range of experimental designs.

---

## Table of Contents  
1. [Project Overview](#project-overview)  
2. [Script Workflow](#script-workflow)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Input Data Requirements](#input-data-requirements)  
6. [Output Formats](#output-formats)  
7. [Dependencies](#dependencies)  
8. [License](#license)  
9. [Contact](#contact)  

---

## Project Overview  
This project consists of four main scripts:  
1. **Processing.py**: Preprocesses raw eye-tracking data, performs windowing, trial rejection, normalization, and RR interval analysis.  
2. **DatasetsConstruction.py**: Constructs datasets from preprocessed data for downstream analysis.  
3. **EyetrackerAnalysis.py**: Provides a class with methods for eye-tracking data analysis, including R-peak detection, baseline correction, and trial rejection.  
4. **MyEyetracker.py**: Processes raw ASC files, handles ECG synchronization, and concatenates datasets from multiple subjects.  

---

## Script Workflow  
1. **Processing.py**  
   - Loads raw eye-tracking data.  
   - Segments data into trials around trigger events.  
   - Rejects noisy trials based on NaN values, statistical outliers, and rolling window artifacts.  
   - Normalizes and baseline-corrects pupil data.  
   - Analyzes RR intervals for cardiac data synchronization.  
   - Saves preprocessed data in pickle format.  

2. **DatasetsConstruction.py**  
   - Loads preprocessed data from `Processing.py`.  
   - Filters and validates data based on RR intervals.  
   - Creates local and global datasets for analysis.  
   - Saves datasets in pickle format.  

3. **EyetrackerAnalysis.py**  
   - Provides a class for eye-tracking data analysis.  
   - Includes methods for data windowing, R-peak detection, normalization, and more.  

4. **MyEyetracker.py**  
   - Modifies ASC files to standardize event codes.  
   - Processes raw eye-tracking data, including blink interpolation and event generation.  
   - Handles ECG synchronization for combined eye-tracking and cardiac data.  
   - Concatenates datasets from multiple subjects for aggregated analysis.  

---

## Installation  
### Prerequisites  
- Python 3.7 or higher  
- Required Python libraries: `numpy`, `pandas`, `plotly`, `wfdb`, `mne`, `pickle`  

### Setup  
1. Clone the repository:  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt  
   ```  
3. Ensure your raw data is in the correct format (see [Input Data Requirements](#input-data-requirements)).  

---

## Usage  
### 1. Preprocessing Data  
Run `Processing.py` to preprocess the raw data:  
```bash
python Processing.py  
```  
Configuration parameters (data paths, thresholds, etc.) are defined in the script.  

### 2. Constructing Datasets  
After preprocessing, run `DatasetsConstruction.py` to build datasets:  
```bash
python DatasetsConstruction.py  
```  

### 3. Using the Analysis Class  
Import and use the `EyetrackerAnalysis` class in your scripts:  
```python
from EyetrackerAnalysis import EyetrackerAnalysis  

# Initialize analysis object
e = EyetrackerAnalysis(data_folder="/path/to/data", data_path="/path/to/data.pickle", fs=500)

# Load raw data
e.load_data(load_windowed=False)

# Create windowed data
e.make_windowed()

# Perform trial rejection
e.make_trial_rejection(thr_nan=1, thr_std=3, n_samp_rej=1, thr_rolling=30, rolling_tol=3)
```

### Processing Raw ASC Files  
`MyEyetracker.py` contains the code to process raw ASC files and handle ECG synchronization:  

---

## Input Data Requirements  
The raw data must be a Pandas DataFrame (saved as a pickle file) with the following fields:  
- `subj_name`: Subject identifier (str)  
- `event_block`: Number of the experimental block (int from 0)  
- `Condition`: The condition name (Synch, Asynch, Isoch, or Baseline)  
- `trigger_counter`: Number of the trial (int from 0)  
- `event_code`: Label containing the position of sound onset (ST)  
- `pupil_right`: Data from the right pupil  
- `ECG`: ECG data  

---

## Output Formats  
1. **Preprocessed Data (`Processing.py`)**  
   - Saved in pickle format with the following structure:  
     ```python
     {
         "dataframes": {
             "original": DataFrame,  # Original data
             "windowed": DataFrame,  # Windowed data
             "data_df": DataFrame,   # Processed data
             "wnd": list,            # Window boundaries in samples
             "wnd_sec": list         # Window boundaries in seconds
         },
         "params": dict              # Processing parameters
     }
     ```

2. **Constructed Datasets (`DatasetsConstruction.py`)**  
   - Saved in pickle format as separate files:  
     - `dt_100.pickle`: Dataset with at least 100 trials per condition.  
     - `dt_100_3.pickle`: Dataset with at least 3 blocks per subject.  
     - `dt_ind.pickle`: Dataset with individual trial offsets applied.  
     - `dt_ind_15.pickle`: Dataset filtered for at least 15 subjects per condition.  
     - `dt_across.pickle`: Dataset with averaged blocks.  
     - `dt_across_15.pickle`: Dataset filtered for at least 15 subjects per condition (averaged blocks).  

3. **Processed ASC Data (`MyEyetracker.py`)**  
   - Saved in pickle format with the following structure:  
     ```python
     {
         "data": DataFrame,  # Processed eye-tracking data
         "events": DataFrame,  # Event markers
         "events_dict": dict  # Dictionary mapping event names to IDs
     }
     ```

---

## Dependencies  
- **NumPy**: For numerical computations.  
- **Pandas**: For data manipulation and analysis.  
- **WFDB**: For R-peak detection in ECG data.  
- **MNE**: For eye-tracking data processing.  
- **Pickle**: For saving and loading data.  

---

## License  
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.  

---

## Contact  
For questions or support, please contact:  
**Giovanni Chiarion**  
Email: chiarion.giovanni@gmail.com  
