# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 11:40:04 2025

@author: Till Habersetzer
         Carl von Ossietzky University Oldenburg
         till.habersetzer@uol.de
"""

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rscipy
import numpy as np

def plot_rotation_circle(ax, idx, center, axes_H1, angle, color, radius=0.2):
    """Plots a full rotation circle around the given axis with an arrowhead at a gap."""
    idxs = [0,1,2]
    idxs.remove(idx)
    
    # axis should be orthogonal
    perp_vec_1 = axes_H1[:,idxs[0]]
    perp_vec_2 = axes_H1[:,idxs[1]]
    perp_vec_1 /= np.linalg.norm(perp_vec_1) # not necessary
    perp_vec_2 /= np.linalg.norm(perp_vec_2)
    
    # Generate points for a full circle
    radius = 0.25
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_points = np.array([center + radius * (np.cos(t) * perp_vec_1 + np.sin(t) * perp_vec_2) for t in theta])
    
    # Create a gap for the arrowhead
    gap_index = int(0.9 * len(circle_points))  # Place arrow near the end of the circle
    circle_part = circle_points[:gap_index]
    
    ax.plot(circle_part[:, 0], circle_part[:, 1], circle_part[:, 2], color=color, linewidth=2)
    
    # Arrowhead at the gap
    arrow_start = circle_points[gap_index-1]
    arrow_end = circle_points[-1]
    arrow_vector = (arrow_end - arrow_start) * 1.5
    ax.quiver(*arrow_start, *arrow_vector, color=color, arrow_length_ratio=1.0)
    
    # Label the rotation angle
    center += center
    
    ax.text(center[0], center[1], center[2], f"{angle:.1f}°", color=color, fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
     

def visualize_transformation(trafo_h1_h2, origin_H1_D, origin_H2_D):
    """Visualizes the relative transformation matrix (rotation + translation) in 3D."""
    
    # Extract rotation matrix and translation vector
    t = trafo_h1_h2[:3, 3] * 1000 # (m->mm)
    R = trafo_h1_h2[:3, :3]
    
    origin_H1_D = origin_H1_D*1000
    origin_H2_D = origin_H2_D*1000
    
    # Convert rotation matrix to Euler angles
    rotation = Rscipy.from_matrix(R)
    euler_angles = rotation.as_euler('xyz', degrees=True)
     
    # Define original coordinate system (H1) at origin (unit vectors)
    origin_H1 = np.array([0, 0, 0]).T
    axes_H1 = np.eye(3)  # Identity matrix represents unit vectors, X (red), Y (green), Z (blue)
    colors = ['r', 'g', 'b']  # X = red, Y = green, Z = blue
    
    # Apply transformation to get new coordinate system (H2)
    origin_H2 = t # Translated origin
    axes_H2 = R @ axes_H1  # Rotated axes
    
    # Plot setup
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Function to plot a coordinate frame
    def plot_frame(ax, origin, axes, label_prefix, alpha=1.0):
        for i in range(3):
            ax.quiver(*origin, *axes[:, i], color=colors[i], 
                      length=1.0, arrow_length_ratio=0.2, alpha=alpha)
            
            text_position = origin + axes[:, i] * 1.2
            ax.text(text_position[0], text_position[1], text_position[2], 
                    f"{label_prefix}{['x', 'y', 'z'][i]}", color=colors[i])
            
    # Plot original and transformed coordinate frames
    plot_frame(ax, origin_H1, axes_H1, "H1: ")
    plot_frame(ax, origin_H2, axes_H2, "H2: ", alpha=0.8)
    
    # Compute translation vector length and cap it at actual displacement
    t_length = np.linalg.norm(t)
    ax.quiver(*origin_H1, *t, color='k', 
              length=1.0, arrow_length_ratio=0.15, linestyle="dashed", label="Translation Vector")
        
    # Label the translation on the arrow itself
    mid_translation = origin_H1 + t / 2  # Midpoint of the arrow
    ax.text(mid_translation[0], mid_translation[1], mid_translation[2], 
            f"T: ({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) \nNorm: {t_length:.1f}", 
            color='k', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    
    # Labels and limits
    xlimits = [-1,t[0]+1] if t[0] > 0 else [-1+t[0],1]
    ylimits = [-1,t[1]+1] if t[1] > 0 else [-1+t[1],1]
    zlimits = [-1,t[2]+1] if t[2] > 0 else [-2+t[2],1]
    
    
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    ax.set_zlim(zlimits)
    ax.set_xlabel("Relative X / mm")
    ax.set_ylabel("Relative Y / mm")
    ax.set_zlabel("Relative Z / mm")
    plt.title("Headposition Transformation H1 -> H2 (relative to H1) \nRotation (°) & Translation (mm)",fontsize=10)
    
    # Add label for coordinate origin in device coordinates
    # Get upper right corner coordinates
    xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    pos_label = (xlim[0], ylim[0], zlim[0])
    ax.text(pos_label[0],pos_label[1],pos_label[2]+1, 
            f"(dev. origin in H1): ({origin_H1_D[0]:.1f}, {origin_H1_D[1]:.1f}, {origin_H1_D[2]:.1f})", 
            color='k', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    ax.text(pos_label[0],pos_label[1],pos_label[2], 
            f"(dev. origin in H2): ({origin_H2_D[0]:.1f}, {origin_H2_D[1]:.1f}, {origin_H2_D[2]:.1f})", 
            color='k', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot full rotation circles around each axis in the original coordinate frame 
    for i in range(0,3):
        angle = euler_angles[i]
        if abs(angle) > 1e-2:  # Skip near-zero rotations
            plot_rotation_circle(ax, i, axes_H1[:,i], axes_H1, angle, colors[i], radius=0.25)
            
    # plt.show()
    return fig
        
   