# Analysis of the influence of gradual changes in matrix sentence similarity on neural envelope tracking

This repository contains the code and analysis pipelines accompanying the article: **"Analysis of the influence of gradual changes in matrix sentence similarity on neural envelope tracking"**. 
* **Article:** 
* **Code archived with Zenodo:** 
* **Dataset:** 

## Software used during Analysis
Software in this block is required to reproduce the analysis results.
* [MATLAB](https://de.mathworks.com/products/matlab.html) (R2024b)
* [FieldTrip toolbox](https://www.fieldtriptoolbox.org/) (version: 20250523) 
* [mTRF-Toolbox](https://github.com/mickcrosse/mTRF-Toolbox) (version: 2.7)
* [THE AUDITORY MODELING TOOLBOX](https://amtoolbox.org/) (version: 1.6.0)
* [MNE-Python](https://mne.tools/stable/index.html) (version: 1.10.1)

## Running the Analysis

**Preprocessing**:
The provided MEG data adheres to the Brain Imaging Data Structure (BIDS) standard, which ensures a clear structure and naming scheme.
Additionally, all MEG recordings have undergone an initial preprocessing stage using MaxFilter. This initial step includes spatiotemporal signal space separation (tSSS) for noise reduction and correction for head movements. To facilitate analysis across multiple runs, all of the subject's recordings have also been transformed to a common head position, using the recording in brackets (‘ses-01-task-olsa_run-02‘) as the reference.
Maxfiltered files are shared withtin the `derivatives` folder.

**Getting Started**:
To set up the project and run the analysis pipeline, please follow these steps:
1.  **Download Code:** Download the code from this repository and save it to a designated folder, for example, `analysis`.
2.  **Download Data:** Download data from OpenNeuro and save it into a separate folder, such as `bidsdata`.
3.  **Install MATLAB Toolboxes:** Install necessary toolboxes and place into a folder, for example `toolboxes`.
    * FieldTrip 
    * mTRF-Toolbox
    * AMT (Decoding)
4.  **Organize Project Structure:** Place all folders (`analysis`, `bidsdata`, `toolboxes`) within a main project directory, for instance, `project_folder`. Your project structure should resemble:
    ```
    project_folder/
    ├── analysis/
    ├── bidsdata/
    └── toolboxes/
    ```
5.  **Configure Settings:** Set up the required settings files for the analysis pipeline.
    The analysis pipelines are controlled by the main settings file (`settings_decoding.m`). Before running any analysis, you must edit this file to define the paths to the project folder and required toolboxes.
    The following key paths need to be set:
    * `settings.rootpath`: This is the base path that helps switch between different computing environments (e.g., a local machine vs. a remote server). All other paths are typically constructed relative to this.
      * Example: `settings.rootpath = '/mnt/localSSDPOOL/projects'`
    * `settings.path2project`: The full path to this main project folder.
      * Example: `settings.path2project = fullfile(settings.rootpath, 'project_folder')`
    * `settings.path2toolboxes`: The full path to the directory where all required toolboxes (like FieldTrip, mTRF-Toolbox, etc.) are stored.
      * Example: `settings.path2toolboxes = fullfile(settings.path2project, 'toolboxes')`

6. **Run the main analysis pipeline:** Adjust the settings inside `analysis_pipeline_decoding.m` and execute the script. This performs all preprocessing across your selected subjects, conditions, and sensors (in parallel, if    enabled). Depending on your configuration, this process may take a while (see the script for details). Ideally, you should run this on a dedicated server.

7. **Plot the results:** Execute `prediction_concatenated_sentences_olsa_similarity_decoding.m` to perform the final computations and reproduce the figures from the paper. Be sure to configure the script settings and plotting options to fit your needs while running.

**Note:** If you plan to use maxfiltered or ICA-preprocessed files that are not in the BIDS derivatives directory, you must run the `maxfilter` or `ICA` scripts first.
   
## Repository Structure

```text
analysis
├── 📂 analysis_speech   # Main analysis scripts and pipelines
├── 📂 helper_functions  # Additional utility functions for the main analysis
├── 📂 maxfilter         # Scripts used for Maxwell filtering of the M/EEG data
└── 📂 ICA               # Scripts used for ICA of the M/EEG data
```

## Analysis Pipelines (`/analysis_speech`)

This folder contains the core scripts used for data preprocessing, training decoders, evaluating results, and plotting.

### Main Decoding Pipeline
These scripts orchestrate the primary workflow, from data preparation to training the decoder models.

| Script Name | Description |
| :--- | :--- |
| `analysis_pipeline_decoding.m` | **Master script** to execute the analysis pipeline. Requires `settings_decoding.m`. Orchestrates a multi-modal neural decoding pipeline consisting of audio and M/EEG preprocessing, linear decoder training, and performance testing across varying intelligibility conditions. |
| `settings_decoding.m` | Contains global analysis settings and parameters for the pipeline. |
| `preprocessing_audio_olsa.m` | Transforms raw OLSA stimuli into bandpass-filtered and resampled auditory envelopes aligned with neurophysiological data parameters. |
| `preprocessing_audio_audiobooks.m` | Processes audiobook stimuli by computing auditory envelopes and segmenting them into fixed-duration epochs. |
| `preprocessing_meeg_audiobooks_decoding.m` | Performs subject-level preprocessing and synchronization of multi-modal neural recordings (MEG, EEG, and ear-EEG) with corresponding audiobook speech envelopes. |
| `preprocessing_meeg_olsa_decoding.m` | Prepares OLSA task neural recordings by synchronizing multi-modal neural recordings (MEG, EEG, and ear-EEG) with OLSA envelopes and trial-specific metadata (SNR and intelligibility levels). |
| `training_decoding.m` | Trains and evaluates a subject-specific decoder across sensor modalities, optimizing the regularization parameter via cross-validation and benchmarking against a shuffled null distribution. |

### Predictions
Scripts dedicated to applying the trained decoders to OLSA sentence-in-noise data to compute SNR dependent neural envelope tracking. 

| Script Name | Description |
| :--- | :--- |
| `prediction_concatenated_sentences_olsa_similarity_decoding.m` | Evaluates a pre-trained neural decoding model on OLSA sentence-in-noise M/EEG data. It computes prediction accuracies and generates multiple baseline metrics (shuffled, time-shifted, and mean sentence) across different signal-to-noise ratios (SNRs). |

### Visualization & Utilities
Scripts for psychometric fitting, visualizing results, generating layouts, and computing demographic statistics.

| Script Name | Description |
| :--- | :--- |
| `compute_and_plot_results_olsa_similarity_tracking.m` | Generates the main publication figures by performing comprehensive statistical analyses on pre-computed M/EEG decoding data and acoustic similarity metrics. It applies Linear Mixed-Effects models to evaluate neural tracking across varying signal-to-noise ratios and correlates neural tracking slopes with acoustic stimulus properties. |
| `plot_training_decoding.m` | Provides group-level analysis and visualization of model training results, including performance metrics, cross-validation curves, and null distributions. |
| `visualize_predictions.m` | Visualizes the performance of a pre-trained neural decoder on OLSA M/EEG data by comparing reconstructed audio envelopes to original stimuli. Generates plots for prediction accuracy across varying signal-to-noise ratios, detailed time-series comparisons, and sentence similarity rankings. |
| `explore_sentences.m` | Analyzes speech stimuli by computing amplitude envelope similarities across time-shifts and extracting temporal modulation spectra. |
| `analyze_olsa_speech_rates.m` | Calculates and visualizes sentence, word, and syllable rates from Oldenburg Sentence Test (OLSA) audio and text files. |

## Maxwell Filtering (`/maxfilter`)
This directory contains Python scripts used for the initial processing of raw MEG data, specifically applying Maxwell filtering for noise reduction and head movement compensation. 
To execute the scripts, the project-specific paths must be defined in `apply_maxfilter_parallel.py` and `check_maxfilter_stats.py`.

| Script Name | Description |
| :--- | :--- |
| `apply_maxfilter_parallel.py` | Executes parallelized Maxwell filtering (SSS/tSSS) on BIDS-formatted MEG data. It automatically detects bad channels, computes continuous head positions (cHPI), and applies movement compensation to suppress external noise. |
| `check_maxfilter_stats.py` | Analyzes and visualizes head movement statistics (pitch, yaw, roll, and translation) extracted from the maxfiltered recordings, generating group-level distributions and subject-level scatter plots. |
| `plot_trafo.py` | A visualization utility module that generates 3D plots of head position transformations (rotation and translation) between different coordinate spaces. |

## Independent Component Analysis (`/ICA`)
This directory contains Python scripts for performing Independent Component Analysis (ICA) on MEG/EEG data. The pipeline includes automated bad channel detection for EEG data using PyPREP, which can also be utilized in FieldTrip for bad channel interpolation. Additionally, the ICA can be computed individually for each recording or pooled across multiple files.

| Script Name | Description |
| :--- | :--- |
| `settings_preprocessing.py` | This configuration file stores all dynamic file paths and settings for the pipeline. |
| `run_ica_pipeline.py` | This script performs ICA-based artifact rejection on MEG/EEG data. |

## Helper Functions (`/helper_functions`)
This directory contains utility functions that support the main analysis pipelines.

### Data Management & Trial Definition

| Script Name | Description |
| :--- | :--- |
| `get_tasks_to_process.m` | Filters task lists to skip completed jobs by checking for existing output files. |
| `read_events_modified.m` | Extracts and filters trigger events from Neuromag FIF stimulus channels. |
| `my_trialfun_audiobook.m` | Custom trial definition function for audiobook data. |
| `my_trialfun_olsa.m` | Custom trial definition function for OLSA sentence trials based on specific event triggers. |

### Signal Processing & Normalization

| Script Name | Description |
| :--- | :--- |
| `cal_envelope.m` | Calculates the auditory envelope of a speech signal. |
| `apply_scaling_epochs.m` | Applies global max-absolute scaling to M/EEG or audio epochs. |
| `apply_zscore_epochs.m` | Applies global z-scoring to M/EEG or audio epochs. |
| `apply_fisher_z_transform.m` | Stabilizes the variance of correlation coefficients using the Fisher Z-transformation (atanh). |
| `apply_inverse_fisher_z_transform.m` | Reverts Fisher Z-transformed scores back into the original correlation coefficient space (r) using the hyperbolic tangent function (tanh). |
| `plot_modspecgram_TH.m` | An adapted AMToolbox function that computes and plots the temporal modulation spectrogram of an audio signal. It has been modified to output the raw spectrogram matrix. |
| `compute_spectrum.m` | Calculates the single-sided magnitude spectrum of a 1D signal, featuring optional detrending and windowing prior to the FFT. |
| `compute_olsa_modulation_spectra.m` | Computes modulation spectra from randomly concatenated OLSA audio sentences using their Hilbert envelopes, with optional filtering, downsampling, detrending, and windowing. |

### Neural Tracking & Statistics

| Script Name | Description |
| :--- | :--- |
| `compute_statistic_concatenated_trials.m` | Generates a bootstrap distribution by resampling matched trials to assess correlation stability. |
| `compute_statistic_single_trials.m` | Generates a null distribution by correlating all mismatched pairs of true and predicted trials, featuring optional circular time-shifting. |
| `compute_statistic_concatenated_trials_deranged.m` | Generates a null distribution by correlating completely mismatched (deranged) trials. |
| `compute_pairwise_permutation_tests.m` | Systematically executes pairwise permutation tests across all condition levels in a results structure. |
| `apply_multiple_comparison_correction.m` | Adjusts p-values for multiple comparisons using Bonferroni, Bonferroni-Holm, or False Discovery Rate (FDR) methods. |
| `get_significance_stars.m` | Converts numeric p-values into standard strings of asterisks (e.g., ***, , *) to visually denote statistical significance thresholds. |

### External Dependencies

| Script Name | Description |
| :--- | :--- |
| [`bluewhitered.m`](https://de.mathworks.com/matlabcentral/fileexchange/4058-bluewhitered) | Generates a divergent blue-white-red colormap. |
| [`distinguishable_colors.m`](https://github.com/cortex-lab/MATLAB-tools/blob/master/distinguishable_colors.m) | Creates maximally distinct colors. |
| [`hatchfill2.m`](https://de.mathworks.com/matlabcentral/fileexchange/53593-hatchfill2) | Adds customizable patterns (hatching/speckling) to filled shapes. |
| [`bonf_holm.m`](https://de.mathworks.com/matlabcentral/fileexchange/28303-bonferroni-holm-correction-for-multiple-comparisons) | Performs Bonferroni-Holm correction for multiple comparisons. |
| [`swtest.m`](https://de.mathworks.com/matlabcentral/fileexchange/13964-shapiro-wilk-and-shapiro-francia-normality-tests) | Shapiro-Wilk parametric hypothesis test of composite normality. |


