# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:05:58 2025

@author: Till Habersetzer
         Carl von Ossietzky University Oldenburg
         till.habersetzer@uol.de
         
Script Description
------------------
This script reads a CSV file containing head movement data (translation and 
rotation) from maxfiltered MEG recordings for multiple subjects. 
It parses this data and generates two sets of visualizations:

1.  Histograms showing the overall distribution of pitch, yaw, roll, and 
    translation values across all subjects combined.
2.  Scatter plots that display the individual movement data for each subject, 
    allowing for inter-subject comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
import os.path as op

#%% Load the CSV file and organize data
#------------------------------------------------------------------------------
path4data = op.join(r"M:\project\analysis\maxfilter","transformation_check.csv")
df = pd.read_csv(path4data, sep=';')

# Extract subject names
subjects = df.iloc[:, 0]
data = df.iloc[:, 1:]

# Prepare containers for each of the four values
trans, rotx, roty, rotz = {}, {}, {}, {}

# Fill the containers
for i, subject in enumerate(subjects):
    t, rx, ry, rz = [], [], [], []
    for val in data.iloc[i]: # go through rows
        tup = tuple(float(x.strip()) if x.strip().lower() != 'nan' else float('nan') for x in val.strip('()').split(','))
        if pd.notna(tup[0]):  # Ignore empty/missing entries (NaN or None)
            t.append(tup[0]*1000) # m -> mm
            rx.append(tup[1])
            ry.append(tup[2])
            rz.append(tup[3])
    trans[subject] = t
    rotx[subject] = rx
    roty[subject] = ry
    rotz[subject] = rz

#%% Plot histograms - all
#------------------------------------------------------------------------------
# Flatten all values from each dictionary
all_rotx = [val for sublist in rotx.values() for val in sublist if not np.isnan(val)]
all_roty = [val for sublist in roty.values() for val in sublist if not np.isnan(val)]
all_rotz = [val for sublist in rotz.values() for val in sublist if not np.isnan(val)]
all_trans = [val for sublist in trans.values() for val in sublist if not np.isnan(val)]

# Flatten all rotation values to compute shared min/max
all_rot = all_rotx  + all_roty + all_rotz
rot_ymin, rot_ymax = min(all_rot), max(all_rot)
trans_ymin, trans_ymax = min(all_trans), max(all_trans)

# --- Plot histograms ---
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.ravel()

axs[0].hist(all_rotx, bins=20, color='skyblue')
axs[0].set_title('Pitch (x-axis rotation)')
axs[0].set_xlabel('Degrees (°)')
axs[0].set_xlim(rot_ymin, rot_ymax)

axs[1].hist(all_roty, bins=20, color='salmon')
axs[1].set_title('Yaw (y-axis rotation)')
axs[1].set_xlabel('Degrees (°)')
axs[1].set_xlim(rot_ymin, rot_ymax)

axs[2].hist(all_rotz, bins=20, color='limegreen')
axs[2].set_title('Roll (z-axis rotation)')
axs[2].set_xlabel('Degrees (°)')
axs[2].set_xlim(rot_ymin, rot_ymax)

axs[3].hist(all_trans, bins=20, color='orange')
axs[3].set_title('Translation distance')
axs[3].set_xlabel('Millimeters (mm)')
axs[3].set_xlim(trans_ymin, trans_ymax)

plt.tight_layout()
plt.show()

#%% Plot for each subject
#------------------------------------------------------------------------------
subjects = list(rotx.keys())
n_subjects = len(subjects)

cmap = colormaps['nipy_spectral'].resampled(n_subjects)
subject_colors = {subj: cmap(i) for i, subj in enumerate(subjects)}

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plotting function
def plot_subject_data(ax, data, title, ylabel, ymin, ymax):
    for subj in subjects:
        ax.scatter([subj] * len(data[subj]), data[subj],
                   color=subject_colors[subj], label=subj, alpha=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Subjects', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(ymin, ymax)

# Plot each of the four types
plot_subject_data(axs[0, 0], rotx, 'Pitch (x-axis rotation)', 'Degrees (°)', rot_ymin, rot_ymax)
plot_subject_data(axs[0, 1], roty, 'Yaw (y-axis rotation)', 'Degrees (°)', rot_ymin, rot_ymax)
plot_subject_data(axs[1, 0], rotz, 'Roll (z-axis rotation)', 'Degrees (°)', rot_ymin, rot_ymax)
plot_subject_data(axs[1, 1], trans, 'Translation distance', 'Millimeters (mm)', trans_ymin, trans_ymax)

# Combine legends from all axes
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=np.ceil(n_subjects/2), frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for legend
plt.show()