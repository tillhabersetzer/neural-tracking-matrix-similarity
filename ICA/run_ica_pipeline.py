# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:58:01 2026

@author: Till Habersetzer
         Carl von Ossietzky University Oldenburg
         till.habersetzer@uol.de
         
OKSIMA ICA preprocessing pipeline
---------------------------------------------------------
This script performs a comprehensive preprocessing using an Independent Component
Analysis (ICA) pipeline on BIDS-formatted MEG/EEG data. 

Key Features & Modifications:
- Toggle `combine_recordings_for_ica`: Choose between computing a single, 
  highly-stable ICA across all concatenated runs per session, OR computing 
  separate ICAs per individual recording.
- Resource Efficient: Training data is safely downsampled to 250 Hz to 
  drastically reduce RAM usage and speed up the Picard fitting algorithm.
- Artifact Rejection: Automatically identifies EOG/ECG artifacts and applies 
  the spatial filter solution back to the original full-resolution data.
- Flexible BIDS Naming: Seamlessly handles files both with and without `run` 
  identifiers (e.g., "olsa_run-01" vs "transient") without breaking naming schemes.
- Advanced Reporting: Generates detailed HTML reports injecting custom 
  Gradiometer and EEG topographies/property-plots for all excluded components.
- Metadata Tracking: Accurately computes and logs empirical data ranks for both 
  MEG and EEG, accounting for MaxFilter info and PyPREP channel interpolations.

"""

from pathlib import Path
import os
import shutil
import json
import mne
import logging
from joblib import Parallel, delayed
import faulthandler
faulthandler.enable()
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from pyprep.find_noisy_channels import NoisyChannels

import settings_preprocessing # import settings

#%% Settings
#------------------------------------------------------------------------------

# ICA Pipeline Behavior
#----------------------
# True: Concatenates all runs in a session before fitting 1 shared ICA model.
# False: Fits a separate ICA model for every single recording.
combine_recordings_for_ica = True

# ICA Downsampling setting (Hz) to speed up fitting and save RAM
ica_resample_sfreq = 250.0 
# -----------------------------

# Select subjects
# subjects = [3, 4, 5, 6]
subjects = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# subjects = [15,16,17,18,19,20]
subjects = [f"sub-{sub_idx:02d}" for sub_idx in subjects]

# Select sessions
# sessions = [1]
sessions = [1,2]
sessions = [f"ses-{ses_idx:02d}" for ses_idx in sessions]

fnames = dict()
fnames['ses-01'] = ["olsa_run-01",
                    "olsa_run-02",
                    "olsa_run-03",
                    "audiobook1_run-01",
                    "audiobook1_run-02",
                    "transient"]
fnames['ses-02'] = ["olsa_run-01",
                    "olsa_run-02",
                    "olsa_run-03",
                    "audiobook2_run-01",
                    "audiobook2_run-02",
                    "transient"]

# fnames = dict()
# fnames['ses-01'] = ["olsa_run-01",
#                     "olsa_run-02"]
# fnames['ses-02'] = ["olsa_run-01",
#                     "olsa_run-02"]

settings = settings_preprocessing.settings
use_maxfilter = settings.get('use_maxfilter', False)

# Import paths
path2bids = settings_preprocessing.get_path('path2bids')
path2derivatives = settings_preprocessing.get_path('path2derivatives')

#%% For Logging
#------------------------------------------------------------------------------
def setup_logging(log_file_path, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    mne.set_log_level('info')
    mne.set_log_file(log_file_path, overwrite=True)

    return logger

#%% Preprocess and concatenate data for ICA
#------------------------------------------------------------------------------

# Debugging
#----------
# subject = subjects[0]
# session = sessions[0]
# fname = 'olsa_run-01'
# task, run = fname.split('_')

# Wrap all per-subject logic into a function
def process_subject(subject):
    
    # Clean start
    #--------------------------------------------------------------------------
    dir2save = Path(settings_preprocessing.get_path('path2ica', subject=subject))
    if dir2save.is_dir():
        shutil.rmtree(dir2save)
        
    dir2save.mkdir(parents=True, exist_ok=True)

    # Initialize dictionary
    #--------------------------------------------------------------------------
    preprocessing_dict = {}
    
    for session in sessions:
        
        # Create a log file for each session and participant
        fname_logger = settings['get_filename'](settings,
                                                'fname_logger',
                                                subject=subject, 
                                                session=session)    
        log_file_path = Path(dir2save, fname_logger)
        logger_name = subject + '_' + session
        logger = setup_logging(log_file_path, logger_name)
        logger.info(f"Started ICA processing subject: {subject} / {session}")
        
        # Create a report for each session 
        report_title = "ICA Combined Across Recordings" if combine_recordings_for_ica else "ICA Per Single Recording"
        report = mne.Report(title=f"{subject}/{session}: {report_title}")
        
        preprocessing_dict[session] = dict()
        preprocessing_dict[session]['bad_channels'] = dict()
        preprocessing_dict[session]['ranks'] = dict()
        preprocessing_dict[session]['eog_events'] = dict()
        preprocessing_dict[session]['epochs_dropped'] = dict()
    
        #----------------------------------------------------------------------
        # Path A: Combined Runs (Fit one ICA for the whole session)
        #----------------------------------------------------------------------
        if combine_recordings_for_ica:
            raw_filt_list = []
            bad_channels_dict = {} 
            
            # Phase 1: Load, filter and aggregate
            #------------------------------------
            for fname in fnames[session]:
                # Safely split the fname into task and run
                parts = fname.split('_')
                task = parts[0]
                run = parts[1] if len(parts) > 1 else None
                
                # Create run_str dynamically to avoid double underscores when run is None
                run_str = f"_{run}" if run is not None else ""
                
                if use_maxfilter:
                    rawdata_path = os.path.join(path2derivatives, subject, 'maxfilter')
                    proc_str = 'proc-tsss-mc' if subject != 'sub-00' else 'proc-tsss'
                    filename = f'{subject}_{session}_task-{task}{run_str}_{proc_str}_meg.fif'
                else:
                    rawdata_path = os.path.join(path2bids, subject, session, 'meg')
                    filename = f'{subject}_{session}_task-{task}{run_str}_meg.fif'
                    
                data_path = os.path.join(rawdata_path, filename)
                raw = mne.io.read_raw_fif(data_path, preload=True, verbose=False)
                
                raw_filt = raw.copy().filter(l_freq=1.0, h_freq=200)
            
                # Bad channel detection
                nd = NoisyChannels(raw_filt.copy().pick_types(meg=False, eeg=True), random_state=None)
                nd.find_all_bads(ransac=True, channel_wise=True)
                
                bad_channels = [str(ch) for ch in nd.get_bads()]
                bad_channels_dict[fname] = bad_channels 
                preprocessing_dict[session]['bad_channels'][fname] = bad_channels
                logger.info(f"{fname} bad channels: {bad_channels}")  
            
                html_str = f"<p>Bad channels: {', '.join(bad_channels)}</p>" if bad_channels else "<p>No bad channels.</p>"
                report.add_html(html=html_str, title="Bad channels", section=f"{fname}: Preprocessing", replace=True)
                
                raw_filt.info["bads"] = []
                raw_filt.info["bads"] = bad_channels
                raw_filt.interpolate_bads(reset_bads=True)
                
                logger.info(f"{fname}: Resampling data to {ica_resample_sfreq} Hz for ICA training.")
                raw_filt.resample(sfreq=ica_resample_sfreq)
                
                raw_filt_list.append(raw_filt)
                del raw, nd 
                
            # Phase 2: Concatenate and Fit ICA
            #---------------------------------
            logger.info(f"Concatenating {len(raw_filt_list)} downsampled runs for session {session}...")
            raw_filt_concat = mne.concatenate_raws(raw_filt_list)
            
            # Compute rank for MEG from info (accounts for MaxFilter)
            rank_meg = mne.compute_rank(raw_filt_concat.copy().pick('meg'), rank='info') 
            
            # Compute rank for EEG empirically from the data (accounts for PyPREP interpolations)
            rank_eeg = mne.compute_rank(raw_filt_concat.copy().pick('eeg'))
            
            # Combine into a single dictionary
            ranks = {**rank_meg, **rank_eeg}
            
            preprocessing_dict[session]['ranks']['combined'] = ranks
            
            report.add_html(html=f"<p>Estimated combined data ranks (MEG & EEG): {ranks}</p>", 
                            title="Data Rank", 
                            section="Session Combined ICA", 
                            replace=True)
            
            epochs = mne.make_fixed_length_epochs(raw_filt_concat, duration=2.0, overlap=0, preload=True)
            n_epochs = len(epochs)
            
            epochs.drop_bad(reject = dict(grad=5000e-13, mag=5e-12, eeg=300e-6))
            preprocessing_dict[session]['epochs_dropped']['combined'] = {'all': n_epochs, 'dropped': sum(1 for log in epochs.drop_log if log)}
            
            report.add_epochs(epochs=epochs, title="Input epochs for ICA", psd=True, replace=True)
            
            ica = ICA(n_components=50, max_iter="auto", method='picard')
            logger.info(f"Computing ICA for session {session}...")
            ica.fit(epochs) 
            logger.info(f"ICA computation finished for session {session}.")
                
            ica.exclude = []
            eog_indices, eog_scores = ica.find_bads_eog(raw_filt_concat, ch_name=['EOG001','EOG002'])
            ica.exclude += eog_indices[:2]
            eog_evoked = create_eog_epochs(raw_filt_concat, ch_name=['EOG001','EOG002']).average()
            eog_evoked.apply_baseline((None, None))
            eog_events = mne.preprocessing.find_eog_events(raw_filt_concat, ch_name=['EOG001','EOG002'])
            preprocessing_dict[session]['eog_events']['combined'] = eog_events.shape[0]
            
            ecg_indices, ecg_scores = ica.find_bads_ecg(raw_filt_concat, ch_name='ECG003') 
            ica.exclude += ecg_indices[:2]
            ecg_evoked = create_ecg_epochs(raw_filt_concat, ch_name='ECG003').average()
            ecg_evoked.apply_baseline((None, None))
            
            report.add_ica(
                ica=ica, 
                title="Combined ICA cleaning", 
                picks=ica.exclude, 
                inst=epochs, 
                eog_evoked=eog_evoked, 
                eog_scores=eog_scores, 
                ecg_evoked=ecg_evoked, 
                ecg_scores=ecg_scores, 
                replace=True)
            
            # Add topographies and properties for gradiometers and EEG
            #------------------------------------------------------------------
            if ica.exclude: 
                # Standalone topographies
                figs_topos_grad = ica.plot_components(picks=ica.exclude, ch_type='grad', show=False)
                figs_topos_eeg  = ica.plot_components(picks=ica.exclude, ch_type='eeg', show=False)
                
                if not isinstance(figs_topos_grad, list): figs_topos_grad = [figs_topos_grad]
                if not isinstance(figs_topos_eeg, list):  figs_topos_eeg = [figs_topos_eeg]
                
                report.add_figure(
                    fig=figs_topos_grad, 
                    title="Excluded Components Topographies (Gradiometers)", 
                    section="Combined ICA cleaning", 
                    replace=True)
                report.add_figure(
                    fig=figs_topos_eeg, 
                    title="Excluded Components Topographies (EEG)", 
                    section="Combined ICA cleaning", replace=True)

                # Detailed property plots
                figs_props_grad = ica.plot_properties(epochs, picks=ica.exclude, topomap_args={'ch_type': 'grad'}, show=False)
                figs_props_eeg = ica.plot_properties(epochs, picks=ica.exclude, topomap_args={'ch_type': 'eeg'}, show=False)
                
                if not isinstance(figs_props_grad, list): figs_props_grad = [figs_props_grad]
                if not isinstance(figs_props_eeg, list):  figs_props_eeg = [figs_props_eeg]
                
                report.add_figure(
                    fig=figs_props_grad, 
                    title="Excluded Components Properties (Gradiometers)", 
                    section="Combined ICA cleaning", 
                    replace=True)
                report.add_figure(
                    fig=figs_props_eeg, 
                    title="Excluded Components Properties (EEG)", 
                    section="Combined ICA cleaning", 
                    replace=True)
            #------------------------------------------------------------------
            
            del raw_filt_concat, epochs, raw_filt_list
            
            # Phase 3: Apply to individual runs
            #----------------------------------
            for fname in fnames[session]:
                parts = fname.split('_')
                task = parts[0]
                run = parts[1] if len(parts) > 1 else None
                
                # Create run_str dynamically to avoid double underscores when run is None
                run_str = f"_{run}" if run is not None else ""
                
                if use_maxfilter:
                    rawdata_path = os.path.join(path2derivatives, subject, 'maxfilter')
                    proc_str = 'proc-tsss-mc' if subject != 'sub-00' else 'proc-tsss'
                    filename = f'{subject}_{session}_task-{task}{run_str}_{proc_str}_meg.fif'
                else:
                    rawdata_path = os.path.join(path2bids, subject, session, 'meg')
                    filename = f'{subject}_{session}_task-{task}{run_str}_meg.fif'
                    
                data_path = os.path.join(rawdata_path, filename)
                raw = mne.io.read_raw_fif(data_path, preload=True, verbose=False)
                
                raw.info["bads"] = bad_channels_dict[fname] 
                raw = raw.interpolate_bads(reset_bads=True)
                
                raw_clean = raw.copy().filter(l_freq=0.1, h_freq=None)
                ica.apply(raw_clean)
                
                # Pass run_str to your filename generator
                fname_out = settings['get_filename'](
                    settings, 
                    'fname_rawdata_preprocessed', 
                    subject=subject, 
                    session=session, 
                    task=task, 
                    run_str=run_str
                )
                
                raw_clean.save(os.path.join(dir2save, fname_out))
                del raw, raw_clean
                logger.info(f"Finished applying ICA and saved: {fname_out}")
                
            logger.info(f"Finished Phase 3: ICA applied to all runs for session {session}.")
            logger.info(f"Finished processing all runs for session {session}. Saving report...")
        
        #----------------------------------------------------------------------
        # Path B: Per-run ICA (Fit a new ICA for every single file)
        #----------------------------------------------------------------------
        else:
            for fname in fnames[session]:
                # Safely split the fname into task and run
                parts = fname.split('_')
                task = parts[0]
                run = parts[1] if len(parts) > 1 else None
                
                # Create run_str dynamically to avoid double underscores when run is None
                run_str = f"_{run}" if run is not None else ""
                
                if use_maxfilter:
                    rawdata_path = os.path.join(path2derivatives, subject, 'maxfilter')
                    proc_str = 'proc-tsss-mc' if subject != 'sub-00' else 'proc-tsss'
                    filename = f'{subject}_{session}_task-{task}{run_str}_{proc_str}_meg.fif'
                else:
                    rawdata_path = os.path.join(path2bids, subject, session, 'meg')
                    filename = f'{subject}_{session}_task-{task}{run_str}_meg.fif'
                    
                data_path = os.path.join(rawdata_path, filename)
                raw = mne.io.read_raw_fif(data_path, preload=True, verbose=False)
                
                raw_filt = raw.copy().filter(l_freq=1.0, h_freq=200)
            
                # Bad channel detection
                nd = NoisyChannels(raw_filt.copy().pick_types(meg=False, eeg=True), random_state=None)
                nd.find_all_bads(ransac=True, channel_wise=True)
      
                bad_channels = [str(ch) for ch in nd.get_bads()]
                preprocessing_dict[session]['bad_channels'][fname] = bad_channels
                logger.info(f"{fname} bad channels: {bad_channels}")  
            
                html_str = f"<p>Bad channels: {', '.join(bad_channels)}</p>" if bad_channels else "<p>No bad channels.</p>"
                report.add_html(html=html_str, title="Bad channels", section=f"{fname}: Preprocessing", replace=True)
                
                raw_filt.info["bads"] = []
                raw_filt.info["bads"] = bad_channels
                raw_filt.interpolate_bads(reset_bads=True)
                
                logger.info(f"{fname}: Resampling data to {ica_resample_sfreq} Hz for ICA training.")
                raw_filt.resample(sfreq=ica_resample_sfreq)
                
                # Compute rank for MEG from info and EEG from data
                rank_meg = mne.compute_rank(raw_filt.copy().pick('meg'), rank='info') 
                rank_eeg = mne.compute_rank(raw_filt.copy().pick('eeg'))
                ranks = {**rank_meg, **rank_eeg}
                
                preprocessing_dict[session]['ranks'][fname] = ranks
                
                report.add_html(html=f"<p>Estimated data ranks (MEG & EEG): {ranks}</p>", title="Data Rank", section=f"{fname}: Preprocessing", replace=True)
                
                epochs = mne.make_fixed_length_epochs(raw_filt, duration=2.0, overlap=0, preload=True)
                n_epochs = len(epochs)
                
                epochs.drop_bad(reject = dict(grad=5000e-13, mag=5e-12, eeg=300e-6))
                preprocessing_dict[session]['epochs_dropped'][fname] = {'all': n_epochs, 'dropped': sum(1 for log in epochs.drop_log if log)}
                
                report.add_epochs(epochs=epochs, title=f"{fname}: Input epochs for ICA", psd=True, replace=True)
                
                ica = ICA(n_components=50, max_iter="auto", method='picard')
                logger.info(f"Computing ICA for {fname}...")
                ica.fit(epochs) 
                logger.info(f"ICA computation finished for {fname}.")
                    
                ica.exclude = []
                eog_indices, eog_scores = ica.find_bads_eog(raw_filt, ch_name=['EOG001','EOG002'])
                # ica.plot_scores(eog_scores, labels='eog', exclude=eog_indices)
                ica.exclude += eog_indices[:2]
                eog_evoked = create_eog_epochs(raw_filt, ch_name=['EOG001','EOG002']).average()
                eog_evoked.apply_baseline((None, None))
                eog_events = mne.preprocessing.find_eog_events(raw_filt, ch_name=['EOG001','EOG002'])
                preprocessing_dict[session]['eog_events'][fname] = eog_events.shape[0]
                
                
                ecg_indices, ecg_scores = ica.find_bads_ecg(raw_filt, ch_name='ECG003') 
                # # ica.plot_scores(ecg_scores, labels='ecg', exclude=ecg_indices)
                ica.exclude += ecg_indices[:2]
                ecg_evoked = create_ecg_epochs(raw_filt, ch_name='ECG003').average()
                ecg_evoked.apply_baseline((None, None))
                
                report.add_ica(
                    ica=ica, 
                    title=f"{fname}: ICA cleaning", 
                    picks=ica.exclude, 
                    inst=epochs, 
                    eog_evoked=eog_evoked, 
                    eog_scores=eog_scores, 
                    ecg_evoked=ecg_evoked, 
                    ecg_scores=ecg_scores, 
                    replace=True)
                
                # Add topographies and properties for gradiometers and EEG
                #--------------------------------------------------------------
                if ica.exclude: 
                    figs_topos_grad = ica.plot_components(picks=ica.exclude, ch_type='grad', show=False)
                    figs_topos_eeg  = ica.plot_components(picks=ica.exclude, ch_type='eeg', show=False)
                    
                    if not isinstance(figs_topos_grad, list): figs_topos_grad = [figs_topos_grad]
                    if not isinstance(figs_topos_eeg, list):  figs_topos_eeg = [figs_topos_eeg]
                    
                    report.add_figure(
                        fig=figs_topos_grad, 
                        title="Excluded Components Topographies (Gradiometers)", 
                        section=f"{fname}: ICA cleaning", 
                        replace=True)
                    report.add_figure(
                        fig=figs_topos_eeg, 
                        title="Excluded Components Topographies (EEG)", 
                        section=f"{fname}: ICA cleaning", 
                        replace=True)

                    figs_props_grad = ica.plot_properties(epochs, picks=ica.exclude, topomap_args={'ch_type': 'grad'}, show=False)
                    figs_props_eeg = ica.plot_properties(epochs, picks=ica.exclude, topomap_args={'ch_type': 'eeg'}, show=False)
                    
                    if not isinstance(figs_props_grad, list): figs_props_grad = [figs_props_grad]
                    if not isinstance(figs_props_eeg, list):  figs_props_eeg = [figs_props_eeg]
                    
                    report.add_figure(
                        fig=figs_props_grad, 
                        title="Excluded Components Properties (Gradiometers)", 
                        section=f"{fname}: ICA cleaning", 
                        replace=True)
                    report.add_figure(
                        fig=figs_props_eeg, 
                        title="Excluded Components Properties (EEG)", 
                        section=f"{fname}: ICA cleaning", 
                        replace=True)
                #--------------------------------------------------------------
                
                # Apply to FULL RESOLUTION raw data
                raw.info["bads"] = []
                raw.info["bads"] = bad_channels
                raw = raw.interpolate_bads(reset_bads=True)
                
                raw_clean = raw.copy().filter(l_freq=0.1, h_freq=None)
                ica.apply(raw_clean)
                
                # Pass run_str to your settings dictionary
                fname_out = settings['get_filename'](settings, 'fname_rawdata_preprocessed', subject=subject, session=session, task=task, run_str=run_str)
                raw_clean.save(os.path.join(dir2save, fname_out))
                
                del raw, raw_filt, epochs, raw_clean
                
                logger.info(f"Finished applying ICA and saved full resolution data: {fname_out}")
                
            logger.info(f"Finished processing all runs for session {session}. Saving report...")

        #%% Save report & metadata for each session
        #----------------------------------------------------------------------
        fname_report = settings['get_filename'](settings, 'fname_report', subject=subject, session=session)
        report.save(os.path.join(dir2save, fname_report), overwrite=True, open_browser=False)
        
    #%% Save metadata for both sessions
    #--------------------------------------------------------------------------
    fname_metadata = settings['get_filename'](settings, 'fname_metadata', subject=subject) 
    with open(os.path.join(dir2save, fname_metadata), 'w') as f:
        json.dump(preprocessing_dict, f, indent=4) 
            
#%% Run in parallel
#------------------------------------------------------------------------------

n_jobs = 6
Parallel(n_jobs=n_jobs)(delayed(process_subject)(subj) for subj in subjects)

print("\nScript finished.")
logging.shutdown()