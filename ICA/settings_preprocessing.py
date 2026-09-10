# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 13:35:34 2025

@author: Till Habersetzer
         Carl von Ossietzky University Oldenburg
         till.habersetzer@uol.de

Configuration file for the OKSIMA ICA preprocessing pipeline.

This script centralizes all essential paths, filename templates, and
preprocessing parameters for the project. It uses a dictionary with lambda
functions to dynamically generate paths, ensuring consistency and making
the pipeline easier to maintain and adapt to different environments.

"""

import os

#%% Preprocessing settings
#------------------------------------------------------------------------------

# Define a single dictionary to hold all settings
settings = {
    
# Essential Paths
#------------------------------------------------------------------------------
'rootpath': r"M:\\", # Corrected local computer path (raw string)

# Project paths are derived from the root path
'path2project': lambda: os.path.join(settings['rootpath'], 'project'),

# BIDS and derivatives paths
'path2bids': lambda: os.path.join(settings['path2project'](), 'bidsdata'),
'path2derivatives': lambda: os.path.join(settings['path2bids'](), 'derivatives'),
'path2ica': lambda subject: os.path.join(settings['path2derivatives'](), subject, 'ica'),

# Filenames (using .format() style for placeholders)
#------------------------------------------------------------------------------
'fnames': {
    'fname_report': '{subject}_{session}_ica-report.html',
    # Changed _{run} to {run_str}
    'fname_rawdata_preprocessed': '{subject}_{session}_task-{task}{run_str}_proc-ica_meg.fif',
    'fname_metadata': '{subject}_metadata_proc-ica_meg.json',
    'fname_logger': '{subject}_{session}_ica_log.txt',
},
'get_filename': lambda s, fname_key, **kwargs: s['fnames'][fname_key].format(**kwargs),

# General Preprocessing Settings
#------------------------------------------------------------------------------
'use_maxfilter': True,
}
    

#%% Helper functions to resolve lambda paths 
#------------------------------------------------------------------------------
# This makes accessing paths cleaner in your main script
def get_path(path_name, **kwargs):
    """
    Resolves a path from the settings dictionary.
    Handles both static strings and dynamic lambda functions.
    """
    path_val = settings[path_name]
    if callable(path_val):
        # If it's a lambda, call it with any provided arguments
        return path_val(**kwargs)
    return path_val

    
