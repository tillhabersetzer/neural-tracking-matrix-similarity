# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:27:30 2025
Revision: Tue May 27 17:11:11 2025
- Application of maxfilter based on BIDS-formatted data

@author: Till Habersetzer
         Carl von Ossietzky University Oldenburg
         till.habersetzer@uol.de
         
Project: Maxwell Filtering and Head Position Correction
--------------------------------------------------------------
This script applies Maxwell filtering (SSS/tSSS) to the BIDS-formatted MEG
data to suppress external noise and correct for subject movement. It processes
data for all subjects and sessions in parallel for efficiency.

The main workflow includes several configurable steps:

1.  **Head Position Calculation:** For each recording, it computes the continuous
    head position (cHPI) from the raw data and saves it to a `.pos` file. This
    is a prerequisite for movement compensation.

2.  **Bad Channel Detection:** It automatically identifies noisy and flat channels
    using `mne.preprocessing.find_bad_channels_maxwell` to prevent artifacts from
    being spread across sensors during processing.

3.  **Maxwell Filtering (`mne.preprocessing.maxwell_filter`):**
    -   Applies temporal Signal-Space Separation (tSSS) to separate brain signals from
        external noise.
    -   If enabled (**MC**), it performs temporal tSSS with movement
        compensation using the pre-calculated head position data.
    -   If enabled (**HPT**), it transforms the data for all runs to a common
        head position, ensuring all recordings are in the same coordinate frame.

4.  **Output and Reporting:**
    -   Saves the cleaned data to a `derivatives` folder with a `_proc-tsss-mc` suffix.
    -   Generates a detailed HTML report for each subject and session, visualizing:
        -   Head movement traces over time.
        -   Power Spectral Density (PSD) plots before and after filtering.
        -   The magnitude of the head position transformations.
    -   Logs all operations and results to a text file for debugging.
    -   Updates several summary `.csv` tables to track the status of the
        preprocessing across the entire dataset.
    
Check out for more details:
---------------------------
Extracting and visualizing subject head movement
https://mne.tools/dev/auto_tutorials/preprocessing/59_head_positions.html
- Transformation to common head positions between runs. That means same
  head-dev-trafo for all runs

Signal-space separation (SSS) and Maxwell filtering
https://mne.tools/stable/auto_tutorials/preprocessing/60_maxwell_filtering_sss.html

Maxwell filter data with movement compensation
https://mne.tools/stable/auto_examples/preprocessing/movement_compensation.html#ex-movement-comp

Others:
-------
https://mne.tools/stable/auto_examples/preprocessing/otp.html
- Application of oversampled temporal projection (otp) (takes too long...)
  Denoising algorithm 
  - OTP algorithm on data with with sensor artifacts (flux jumps) and random noise                                                       
"""

#%% Settings
#------------------------------------------------------------------------------
from pathlib import Path
import mne
from mne.preprocessing import find_bad_channels_maxwell
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from plot_trafo import visualize_transformation
import logging
import shutil
import faulthandler
faulthandler.enable()

from joblib import Parallel, delayed
import multiprocessing

subjects  = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# subjects  = [19]
subjects  = [f"sub-{sub_idx:02d}" for sub_idx in subjects]

sessions = [1,2] # 01 or 02
# sessions = [2] # 01 or 02

fnames = dict()
fnames['ses-01'] = ["olsa_run-01",
                    "olsa_run-02",
                    "olsa_run-03",
                    "audiobook1_run-01",
                    "audiobook1_run-02",
                    "transient",
                    "noise",
                    ]
fnames['ses-02'] = ["olsa_run-01",
                    "olsa_run-02",
                    "olsa_run-03",
                    "audiobook2_run-01",
                    "audiobook2_run-02",
                    "transient",
                    "noise",
                    ]

# path to project (needs to be adjusted)
project_dir = Path(r"M:\project") 
bids_dir = Path(project_dir,"bidsdata")

# Mark bad channels to prevent noise spreading
bad_channels = ['MEG1813','MEG2622','MEG2121']

# Apply Oversampled Temporal Projection to reduce sensor noise before MaxFilter
OTP = False
         
# Apply Headposition Transformation 
HPT = True

fname_ref_head = ["ses-01","olsa_run-02"]

# Apply and compute movement correction
MC = True

# Tables to track head position computation and maxwell filtering
#------------------------------------------------------------------------------
subjects_all = [f"sub-{sub_idx:02d}" for sub_idx in list(range(0, 20 + 1))] # for tables
fnames_all = [f"{session}_{fname}" for session in fnames for fname in fnames[session]]

# Check if continuous head position is available
fname_table_headposition_tracking = "headposition_tracking_check.csv"
if Path(fname_table_headposition_tracking).is_file():
    df_headposition_tracking = pd.read_csv(fname_table_headposition_tracking, index_col="subject",sep=";")
    print("Loaded existing CSV file.")
else:    
    data = {col: [False] * len(subjects_all) for col in fnames_all}  # Initialize all as False
    df_headposition_tracking = pd.DataFrame(data, index=subjects_all)
    df_headposition_tracking.index.name = "subject"
    
# Check if maxfilter has been applied
fname_table_maxfilter = "maxfilter_check.csv"
if Path(fname_table_maxfilter).is_file():
    df_maxfilter = pd.read_csv(fname_table_maxfilter, index_col="subject",sep=";")
    print("Loaded existing CSV file.")
else:    
    data = {col: [False] * len(subjects_all) for col in fnames_all}  # Initialize all as False
    df_maxfilter = pd.DataFrame(data, index=subjects_all)
    df_maxfilter.index.name = "subject"
    
# Check translation and rotation values
fname_table_transformation = "transformation_check.csv"
if Path(fname_table_transformation).is_file():
    df_transformation = pd.read_csv(fname_table_transformation, index_col="subject",sep=";")
    print("Loaded existing CSV file.")
else:  
    data = {col: [(np.nan, np.nan, np.nan, np.nan) for _ in subjects_all] for col in fnames_all}
    df_transformation = pd.DataFrame(data, index=subjects_all)
    df_transformation.index.name = "subject"
    
# For Logging
#------------------------------------------------------------------------------
def setup_logging(log_file_path, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file_path, mode='w') # mode='w' to overwrite
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Redirect MNE log output to the same log file
    mne.set_log_level('info')  # Could also be 'debug', 'warning', etc.
    mne.set_log_file(log_file_path, overwrite=True)  # This makes MNE log to the file directly

    return logger
    
#%% Computations
#------------------------------------------------------------------------------

# Wrap all per-subject logic into a function
def process_subject(subject):
    
    # Clean start - create directory 
    dir2save = Path(bids_dir,"derivatives",subject,"maxfilter")
    if dir2save.is_dir():
        shutil.rmtree(dir2save)
    
    dir2save.mkdir(parents=True, exist_ok=True)
        
    results = {
        'df_headposition_tracking': [],
        'df_maxfilter': [],
        'df_transformation': [],
    }
    
    # Load crosstalk compensation and fine calibration files from BIDS-formatted data
    crosstalk_file = Path(bids_dir,subject,'meg', f"{subject}_acq-crosstalk_meg.fif")
    fine_cal_file = Path(bids_dir,subject,'meg', f"{subject}_acq-calibration_meg.dat")

    for ses_idx in sessions:
        session = f"ses-{ses_idx:02d}"

        figs_list_headposition = []
        figs_list_maxfilter_before = []
        figs_list_maxfilter_after = []
        figs_list_trafo = []
        captions_list_headposition = []
        captions_list_maxfilter = []
        captions_list_trafo = []

        rawdata_dir = Path(bids_dir,subject,session,"meg")
        # Create a log file for each session and participant
        log_file_path = Path(dir2save,subject + '_' + session + '_maxfilter_log.txt')
        logger_name = subject + '_' + session
        logger = setup_logging(log_file_path, logger_name)
        logger.info(f"Started processing subject: {subject} / {session}")

        for fname in fnames[session]:
            
            fname_prefix = subject + '_' + session + "_task-" + fname        
            raw_fname = Path(rawdata_dir, fname_prefix + "_meg.fif")
            
            #%% Compute Head Movements
            #------------------------------------------------------------------
            if MC and "noise" not in fname: # not for emptyrooms
            
                if HPT:
                    # destination = (0, 0, 0.04)
                    destination = Path(bids_dir,subject,fname_ref_head[0],"meg",subject + '_' + fname_ref_head[0] + '_task-' + fname_ref_head[1] + '_meg.fif')
                    
                headpos_fname = Path(dir2save, fname_prefix + "_raw.pos")
                
                # Head positions are computed if raw meg file exists and positions
                # haven't been computed yet (takes a while...)
                if raw_fname.is_file():
                    
                    if not headpos_fname.is_file():
                
                        raw = mne.io.read_raw_fif(raw_fname, allow_maxshield=False, verbose=True)
                    
                        # Compute head position
                        #----------------------
                        chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)  # HPI coil amplitudes over time
                        chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes) # time-varying HPI coil locations
                        head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=True) # head positions
                    
                        # Check if continuous head position is available and store
                        if head_pos.shape[0]>0: # cHPI active
                
                            mne.chpi.write_head_pos(headpos_fname, head_pos)
                        
                            captions_list_headposition.append(fname)
                            figs_list_headposition.append(mne.viz.plot_head_positions(head_pos, mode="traces",destination=destination,show=False))
                
                            # Mark as computed in table
                            results['df_headposition_tracking'].append((subject, f"{session}_{fname}", True))                      
                        
                    # Save figure even though it has already been computed
                    else:
                        head_pos = mne.chpi.read_head_pos(headpos_fname)
                        captions_list_headposition.append(fname)
                        figs_list_headposition.append(mne.viz.plot_head_positions(head_pos, mode="traces",destination=destination,show=False))
            
                        # Mark as computed in table
                        results['df_headposition_tracking'].append((subject, f"{session}_{fname}", True))                 
                       
            #%% Maxwell filtering
            #------------------------------------------------------------------
            if raw_fname.is_file():
                
                raw = mne.io.read_raw_fif(raw_fname, allow_maxshield=False, verbose=True)
                raw.info["bads"] = bad_channels
                
                # Oversampled temporal projection
                #--------------------------------
                if OTP:
                    # Denoise MEG channels using leave-one-out temporal projection
                    # This algorithm is computationally expensive
                    raw = mne.preprocessing.oversampled_temporal_projection(raw)
                
                # emptyroom 
                #------------
                if "noise" in fname:      
                    destination = None
                    head_pos = None
                    st_duration = None
                    coord_frame = "meg"
                    proc = "_proc-sss"
                        
                # recordings with subjects inside meg
                #------------------------------------
                else:  
                    st_duration = 10
                    coord_frame = "head"
                    
                    # Head Position Transformation
                    #-----------------------------
                    if HPT: 
                        # destination = (0, 0, 0.04)
                        destination = Path(bids_dir,subject,fname_ref_head[0],"meg",subject + '_' + fname_ref_head[0] + '_task-' + fname_ref_head[1] + '_meg.fif')
                    else:
                        destination = None
                        
                    # Movement Correction  
                    #--------------------
                    headpos_fname = Path(dir2save,fname_prefix + "_raw.pos")
                    if MC and headpos_fname.is_file(): 
                        head_pos = mne.chpi.read_head_pos(headpos_fname)
                        proc = "_proc-tsss-mc"
                    else:
                        head_pos = None       
                        proc = "_proc-tsss"         
                        
                # Detect bad channels
                #--------------------
                raw_check = raw.copy()
                auto_noisy_chs, auto_flat_chs = find_bad_channels_maxwell(
                    raw_check, 
                    cross_talk=crosstalk_file, 
                    calibration=fine_cal_file,
                    coord_frame=coord_frame, 
                    head_pos=head_pos, 
                    return_scores=False, 
                    verbose=True)
                logger.info(auto_noisy_chs)  
                logger.info(auto_flat_chs)  
                
                # Update list of bad channels
                bads = raw.info["bads"] + auto_noisy_chs + auto_flat_chs
                raw.info["bads"] = bads
                       
                # Apply MaxFilter
                #----------------
                raw_tsss = mne.preprocessing.maxwell_filter(
                    raw, 
                    cross_talk=crosstalk_file,
                    calibration=fine_cal_file, 
                    st_duration=st_duration, 
                    head_pos=head_pos, 
                    destination=destination,
                    coord_frame=coord_frame, 
                    verbose=True)
                    
                # Save data
                #----------
                raw_tsss.save(Path(dir2save,fname_prefix + proc + "_meg.fif"),overwrite=True)
                    
                figs_list_maxfilter_before.append(raw.compute_psd().plot(show=False, xscale="log"))
                figs_list_maxfilter_after.append(raw_tsss.compute_psd().plot(show=False, xscale="log"))
                captions_list_maxfilter.append(fname)
                
                # Mark as computed in table
                results['df_maxfilter'].append((subject, f"{session}_{fname}", True))                 
                
                # Estimate difference between trafos (rotation, translation)
                #-----------------------------------------------------------
                if not "noise" in fname:  
                    
                    trafo_1 = raw.info['dev_head_t']['trans']
                    trafo_2 = raw_tsss.info['dev_head_t']['trans']
                    trafo_head1_to_head2 = trafo_2 @ np.linalg.inv(trafo_1)  
                    
                    # Distance between two recordings
                    distance = np.linalg.norm(trafo_head1_to_head2[:3, 3])
                    
                    # Convert the rotation matrix to Euler angles (in radians)
                    # Specify the order of rotations: "xyz" for rotations around x (pitch), y (yaw), z (roll)
                    rotation = R.from_matrix(trafo_head1_to_head2[:3, :3])
                    euler_angles = rotation.as_euler('xyz', degrees=True)
                    pitch, yaw, roll = euler_angles
                    
                    logger.info(f"Translation distance: {distance}\n")
                    logger.info(f"Pitch (x-axis rotation): {pitch:.2f} degrees\n")
                    logger.info(f"Yaw (y-axis rotation): {yaw:.2f} degrees\n")
                    logger.info(f"Roll (z-axis rotation): {roll:.2f} degrees\n")
                    
                    # save only up to 4 digits
                    traf_data = (distance,pitch,yaw,roll)
                    results['df_transformation'].append((subject, f"{session}_{fname}", tuple(round(float(i), 4) for i in traf_data)))   
                    
                    # Add plot of transformation to report
                    figs_list_trafo.append(visualize_transformation(trafo_head1_to_head2,trafo_1[:3, 3],trafo_2[:3, 3]))
                    captions_list_trafo.append(fname)
                    
        # Append plots to report
        #-----------------------
        report_fname = Path(dir2save,subject + '_' + session + "_maxfilter-report.hdf5")
        report_html_fname = Path(dir2save,subject + '_' + session + "_maxfilter-report.html")

        if figs_list_headposition:
            
            with mne.open_report(report_fname) as report:
                report.add_figure(
                figs_list_headposition,
                title="Estimated head positions",
                caption=captions_list_headposition,
                replace=True
                )

        if figs_list_maxfilter_after:
            
            with mne.open_report(report_fname) as report:
                report.add_figure(
                figs_list_maxfilter_before,
                title="PSD before maxwell filtering",
                caption=captions_list_maxfilter,
                replace=True
                )
                report.add_figure(
                figs_list_maxfilter_after,
                title="PSD after maxwell filtering",
                caption=captions_list_maxfilter,
                replace=True
                )
                    
        if figs_list_trafo:
            
            with mne.open_report(report_fname) as report:
                report.add_figure(
                figs_list_trafo,
                title="Head position transformation",
                caption=captions_list_trafo,
                replace=True
                )
        report.save(report_html_fname, overwrite=True, open_browser=False)
                    
    return results
                    
#%% Run in parallel
#------------------------------------------------------------------------------

n_jobs = min(len(subjects), multiprocessing.cpu_count() - 1)
#n_jobs = 1
results = Parallel(n_jobs=n_jobs)(delayed(process_subject)(subj) for subj in subjects)

# Merge results into master DataFrames
#-------------------------------------
# for res in results:
for subj_results in results:
    for key in ['df_headposition_tracking', 'df_maxfilter', 'df_transformation']:
        res_df = subj_results[key]
        
        # Iterate through each tuple in the list (res_df)
        for subject, session_filename, value in res_df:  # Now unpacking (subject, session_filename, value)
            
            if key == 'df_headposition_tracking':
                    df_headposition_tracking.loc[subject, session_filename] = value
            elif key == 'df_maxfilter':
                df_maxfilter.loc[subject, session_filename] = value
            elif key == 'df_transformation':
                df_transformation.loc[subject, session_filename] = value
    
# Save to CSV
#------------
df_headposition_tracking.to_csv(fname_table_headposition_tracking, sep=";")
df_maxfilter.to_csv(fname_table_maxfilter, sep=";")
df_transformation.to_csv(fname_table_transformation, sep=";")