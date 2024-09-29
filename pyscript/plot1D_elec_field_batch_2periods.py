# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 23:13:24 2023

@author: Racha
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Don't display the plot
import h5py
import re
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

def extract_number(filename):
    match = re.search(r'(\d+)\.h5$', filename)  # Find the last sequence of numbers before .h5
    if match:
        return int(match.group(1))  # Convert to integer and return
    else:
        return None  # Return None if no number found

# und_numbers = 1
und_numbers = sys.argv[1]

file_numbers = 40 * int(und_numbers) + np.arange(0,81)
z2_lim_min = float(sys.argv[4])
z2_lim_max = float(sys.argv[5])
z2_lim = (z2_lim_min, z2_lim_max)
p_j_batch = []
p_j_global_max = []
m_Z_batch = []
net_sums_batch = []
net_sums_global_max = []
num_e_batch = []
num_e_global_max = []

print("Reading files...")
for num in file_numbers:
    # electrons_file = f"D://Puffin_results//gamma_100_rho0.079_helical//SSS_e_{num}.h5"
    electrons_file = sys.argv[2] + f'{num}.h5'

    with h5py.File(electrons_file, 'r') as h5e:
        Electrons = h5e['/electrons'][:]
        m_Z = Electrons[:, 2]
        m_GAMMA = Electrons[:, 5]

        rho = h5e['/runInfo'].attrs['rho']
        gamma_r = h5e['/runInfo'].attrs['gamma_r']
    # Define bin edges with fixed width
    bin_lambda = 0.05 # bin size in lambda
    d_bin = 0.05*4*np.pi*rho # in unit of z2
    MPperWave = 800
    norm_num = MPperWave*d_bin

    bin_edges = np.arange(z2_lim[0], z2_lim[1], d_bin)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate energy spread
    
    gamma_j = gamma_r * m_GAMMA
    
    
    energy_spread = (gamma_j - gamma_r) * 100 / gamma_r

    p_j = (gamma_j - gamma_r) / (rho * gamma_r)
    
    filtered_indices = np.where((m_Z >= z2_lim[0]) & (m_Z <= z2_lim[1]))
    filtered_m_Z = m_Z[filtered_indices]
    filtered_p_j = p_j[filtered_indices]
    
    num_bins = len(bin_edges) - 1
    net_sums = np.zeros(num_bins)
    num_e = np.zeros(num_bins)
    for k in range(num_bins):
        in_bin = (filtered_m_Z >= bin_edges[k]) & (filtered_m_Z < bin_edges[k+1])
        # count_true = np.sum(in_bin)
        energies_in_bin = filtered_p_j[in_bin]
        
        num_e[k] = np.sum(in_bin)/norm_num
        net_sums[k] = np.sum(energies_in_bin)/norm_num
    
    num_e_batch.append(num_e)
    num_e_global_max.append(np.max(num_e))
    
    net_sums_batch.append(net_sums)
    net_sums_global_max.append(np.max(np.abs(net_sums)))
   
    p_j_batch.append(filtered_p_j)
    p_j_global_max.append(np.max(np.abs(filtered_p_j)))
    m_Z_batch.append(filtered_m_Z)
    
print("___Read Done___")

print("Getting Max for plot...")
p_j_global_max = np.array(p_j_global_max)
p_j_ylim = 0.5 * np.ceil(np.max(p_j_global_max)/0.5)

net_sums_global_max = np.array(net_sums_global_max)
net_sums_lim =  0.5 * np.ceil(np.max(net_sums_global_max) /0.5 )

num_e_global_max = np.array(num_e_global_max)
num_e_lim =  0.5 * np.ceil(np.max(num_e_global_max)/0.5)

print(f"pj_lim = {p_j_ylim}, net_pj_lim = {net_sums_lim}, num_e_lim = {num_e_lim}")

intens_global_max = []

for i, num in enumerate(file_numbers):
    # aperp_file = f"D://Puffin_results//gamma_100_rho0.079_helical//SSS_ap_{num}.h5"
    aperp_file = sys.argv[3] + f'{num}.h5'
    # print(aperp_file)
    with h5py.File(aperp_file, 'r') as h5f:
    # Calculate z2_bar
        nz = h5f['/runInfo'].attrs['nZ2']
        meshsizeZ2 = h5f['/runInfo'].attrs['sLengthOfElmZ2']
        z2_bar = np.linspace(0, meshsizeZ2 * (nz - 1.0), nz)

        z2_lo = int(np.floor(z2_lim[0]/meshsizeZ2))
        z2_hi = int(np.floor(z2_lim[1]/meshsizeZ2))

        aperp_x = h5f['/aperp'][0, z2_lo:z2_hi]
        aperp_y = h5f['/aperp'][1, z2_lo:z2_hi]
    intensity = aperp_x**2 + aperp_y**2
    intens_global_max.append(np.max(intensity))
print("Intensity max for plot...")
intens_global_max = np.array(intens_global_max)
intens_ylim = 0.5 * np.ceil(np.max(intens_global_max)/0.5)
ap_ylim = np.sqrt(intens_ylim)

for i, num in enumerate(file_numbers):
    # aperp_file = f"D://Puffin_results//gamma_100_rho0.079_helical//SSS_ap_{num}.h5"
    aperp_file = sys.argv[3] + f'{num}.h5'
    # print(aperp_file)
    with h5py.File(aperp_file, 'r') as h5f: 
    # Calculate z2_bar
        nz = h5f['/runInfo'].attrs['nZ2']
        meshsizeZ2 = h5f['/runInfo'].attrs['sLengthOfElmZ2']
        z2_bar = np.linspace(0, meshsizeZ2 * (nz - 1.0), nz)
    
        z2_lo = int(np.floor(z2_lim[0]/meshsizeZ2))
        z2_hi = int(np.floor(z2_lim[1]/meshsizeZ2))
    
        aperp_x = h5f['/aperp'][0, z2_lo:z2_hi]
        aperp_y = h5f['/aperp'][1, z2_lo:z2_hi]
    intensity = aperp_x**2 + aperp_y**2

    z2axis = np.arange(0, len(intensity))*meshsizeZ2 # start from zero
    z2axis_old = z2_bar[z2_lo:z2_hi]
    
    plt.style.use('bmh')
    plt.rcParams['axes.grid'] = False
    # Set ticks on the top and right spines globally

    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True

    # If you also want to ensure that the bottom and left ticks are enabled (they usually are by default):
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = 'Arial'
    # Create a figure and set up a gridspec layout
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[5, 3, 4])
    
    ax0 = fig.add_subplot(gs[2])
    ax0.plot(z2axis, intensity)
    ax0.set_ylim(0, intens_ylim)
    ax0.set_xlim(min(z2axis), max(z2axis))
    ax0.set_ylabel(r'$|A|^2$')
    ax0.set_xlabel(r'$\bar{z}_2$')
    # ax0.xaxis.set_ticklabels([]) 
    # ax0.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.3)
    ax0.tick_params(direction='in')
    x_spacing_value = 1  # Adjust as needed
    ax0.xaxis.set_major_locator(MultipleLocator(x_spacing_value))
    
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(z2axis_old, aperp_x, label='$A_x$')
    ax1.plot(z2axis_old, aperp_y, label='$A_y$')
    ax1.set_ylabel(r'$A$')
    ax1.set_ylim(-ap_ylim, ap_ylim)
    ax1.set_xlim(min(z2axis_old), max(z2axis_old))
    ax1.xaxis.set_ticklabels([])  # remove x-ticks for ax0 since it shares the x-axis with ax1
    # Control the x-axis tick (and grid) spacing
    ax1.grid(True, which='both', axis='x', color='gray', linestyle='--', linewidth=0.3)
    ax1.tick_params(direction='in')
    ax1.xaxis.set_major_locator(MultipleLocator(x_spacing_value))
    ax1.legend(loc='upper right', framealpha=0, edgecolor='none')
    
    ax1_2 = ax1.twinx()
    ax1_2.bar(bin_centers, num_e_batch[i], width=(bin_edges[1] - bin_edges[0]), align='center', color='tab:green', alpha=0.4)
    ax1_2.set_ylim(0, num_e_lim)
    ax1_2.set_ylabel(r'Norm. $\bar{n}_e$', color='tab:green')
    for label in ax1_2.get_yticklabels():
        label.set_color('tab:green')
    # If you also want the tick lines to be green:
    ax1_2.tick_params(axis='y', colors='tab:green')
    
    print(f'plotting file_num = {num}')
    StepsPerPeriod = 40 # number of output file steps per period
    Nw = np.array(num)/StepsPerPeriod
    z_bar = (4*np.pi*rho)*Nw
    ax2 = fig.add_subplot(gs[0])
    ax2.text(0.02, 1.0, r'$\bar{z} =$' + f'{z_bar:.3f}', transform=ax2.transAxes,
             verticalalignment='bottom', horizontalalignment='left')
    ax2.text(0.22, 1.0, r'$N_w =$' + f'{Nw:.3f}', transform=ax2.transAxes,
                        verticalalignment='bottom', horizontalalignment='left')
    ax2.scatter(m_Z_batch[i], p_j_batch[i], s=5 , marker='o', edgecolors='black', facecolors='none', label='$p_j$')
    
    x_ticks = np.arange(z2_lim[0], z2_lim[1]+0.5, 1)
    ax2.set_xticks(x_ticks)
    
    ax2.set_ylim(-p_j_ylim, p_j_ylim)
    ax2.set_xlim(z2_lim)
    ax2.set_ylabel(r'$p_j$')
    # ax2.set_xlabel(r'$\bar{z}_2$')
    ax2.grid(True, which='both', axis='x', color='gray', linestyle='--', linewidth=0.3)
    ax2.tick_params(direction='in')
    ax2.xaxis.set_ticklabels([])  # remove x-ticks for ax0 since it shares the x-axis with ax1
    ax2.xaxis.set_major_locator(MultipleLocator(x_spacing_value))
    
    ax2_2 = ax2.twinx()
    colors = np.where(net_sums_batch[i] >= 0, 'tab:blue', 'tab:red')
    ax2_2.bar(bin_centers, net_sums_batch[i], width=(bin_edges[1] - bin_edges[0]), align='center', alpha=0.5, color=colors)
    ax2_2.set_xlim(z2_lim)
    ax2_2.set_ylim(-net_sums_lim, net_sums_lim)
    ax2_2.tick_params(direction='in')
    ax2_2.set_ylabel(r'$\Sigma p_j$')

    plt.subplots_adjust(hspace=0.075)
    fig.savefig(aperp_file[:-3] + '_elec.png', pad_inches = 0.1, dpi=220)
    # Close the input files
print(f'Undulator {und_numbers} DONE')
