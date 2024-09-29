# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 02:40:36 2023

@author: Racha
"""

import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Don't display the plot
import matplotlib.pyplot as plt

filename_e = sys.argv[1]
# filename_e = "J://Puffin_results//beam_file.h5"

hdf5_file = h5py.File(filename_e, 'r+')

# read electrons data set
e_data = hdf5_file['/electrons'][:]
m_Z = e_data[:, 2] # electron position in Z2
m_GAMMA = e_data[:, 5] # electron Gamma_j

# Calculate energy spread
gamma_r = hdf5_file['/runInfo'].attrs['gamma_r']
rho = hdf5_file['/runInfo'].attrs['rho']

meshsizeZ2 = hdf5_file['/runInfo'].attrs['sLengthOfElmZ2']
nZ2 = hdf5_file['/runInfo'].attrs['nZ2']
print("z2 mesh =", meshsizeZ2)
print("nz2 =", nZ2)

modelLengthInZ2 = nZ2*meshsizeZ2
Lc = hdf5_file['/runInfo'].attrs['Lc']
lambda_r = hdf5_file['/runInfo'].attrs['lambda_r']
modelLengthInlambda = modelLengthInZ2/(4*np.pi*rho)
print("model length in lambda =", modelLengthInlambda)

numelec = e_data.shape[0]
print("total number of electrons =", numelec)

MPsPerWave = np.int64(np.round(numelec/modelLengthInlambda))
print("MPsPerWave =", MPsPerWave)

# generate the equally space with in z2 model length
inElectronSpace = np.float64(np.linspace((nZ2-1)*meshsizeZ2/(2*numelec), (nZ2-1)*meshsizeZ2*(1-1.0/(2*numelec)), numelec))
weight = 4.0*np.pi*rho/MPsPerWave
# Generate the random normal weight array
# percent_weight = 0.1879/np.sqrt(4.0*np.pi*rho)/100
# random_weights = np.random.normal(weight, percent_weight, numelec)

numEdit_inZ2 = float(sys.argv[3]) # length of head and tail in Z2 to reset
# numEdit_inZ2 = 180
numEdit = np.ceil(MPsPerWave*float(numEdit_inZ2)/(4.0*np.pi*rho)).astype(int) # unit of len_in_z2 * (1/(4*np.pi*rho))
tailsposition = inElectronSpace[-numEdit:][0]
print(tailsposition)
headsposition = inElectronSpace[numEdit:][0]
print("number of initial electrons in the range from heads/tails =", numEdit) # initial edit length

# sort the electrons using column #2 (electron_z)
sorted_elec = e_data[e_data[:,2].argsort()]
zero_e = sorted_elec[sorted_elec[:, 6] == 0.0]
non_zero_e = sorted_elec[sorted_elec[:, 6] != 0.0]

if zero_e.shape[0] == 0:
    ("_______________________")
    print("There are no zero weights.")
    ("_______________________")
else:
    print("_______________________")
    print("There are zero weights.")
    print("Number of zero weights = ", zero_e.shape)
    print("_______________________")
    
zero_e[:, [0, 1, 2, 3, 4, 6]] = 0.0 # set everything else apart from gamma_j to zero
zero_e[:, 5] = 1.0 # set gamma_j to 1.0

# Find the index where values in the second column are greater than tailsposition
split_index = np.where(non_zero_e[:, 2] >= tailsposition)[0][0]
# Split the array at the found index
first_part, second_part = np.split(non_zero_e, [split_index])
print("Number of tails =", second_part.shape)

heads_split_index = np.where(first_part[:, 2] >= headsposition)[0][0]
first_part, main_part = np.split(first_part, [heads_split_index])
print("Number of heads =", first_part.shape)
print("Number of bodies =", main_part.shape)
print("_______________________")

if len(second_part) > numEdit:
    print("Number of tails exceed initial numbers")     
    residue_tails = second_part[numEdit:, :]
    residue_tails[:, [0, 1, 2, 3, 4, 6]] = 0.0
    residue_tails[:, 5] = 1.0 
    print("Set to zero weights =", residue_tails.shape)
        
    zero_e = np.vstack((zero_e, residue_tails))
    print("Total zero weights = ", zero_e.shape)
    second_part = second_part[:numEdit, :]
    
    second_part[: ,6] = weight
    second_part[: ,5] = 1.0
    second_part[: ,4] = 0.0
    second_part[: ,3] = 0.0
    second_part[: ,2] = inElectronSpace[-numEdit:]
    second_part[: ,1] = 0.0
    second_part[: ,0] = 0.0
    print("now tails =", second_part.shape)
    print("___________________")
    
elif len(second_part) == numEdit:
    print("Number of tails equal initial numbers")
    second_part[: ,6] = weight
    second_part[: ,5] = 1.0
    second_part[: ,4] = 0.0
    second_part[: ,3] = 0.0
    second_part[: ,2] = inElectronSpace[-numEdit:]
    second_part[: ,1] = 0.0
    second_part[: ,0] = 0.0
    print("now tails =", second_part.shape)
    print("___________________")

else:
    print("Number of tails less than initial numbers")
    fill_zero = numEdit - len(second_part)

    if zero_e.shape[0] == 0:
        print("No zero yet")
        print("Re-fill empty")
    # If zero_e is empty, ensure fill_part has the same columns as second_part but no rows
        fill_part = np.empty((0, second_part.shape[1]))
    else:
        fill_part = zero_e[-fill_zero:, :]
        print("Re-fill electrons", fill_zero)

    # Update zero_e to remove the rows now in fill_part
        zero_e = zero_e[:-fill_zero, :] if fill_zero != 0 else zero_e

    print("Total zero weights = ", zero_e.shape[0])

    second_part = np.vstack((second_part, fill_part))
    
    second_part[: ,6] = weight
    second_part[: ,5] = 1.0
    second_part[: ,4] = 0.0
    second_part[: ,3] = 0.0
    second_part[: ,2] = inElectronSpace[-len(second_part):]
    second_part[: ,1] = 0.0
    second_part[: ,0] = 0.0 
    print("now tails =", second_part.shape)
    print("___________________")


if len(first_part) > numEdit:
    print("Number of heads exceed initial numbers")
    residue_heads = first_part[numEdit:, :]
    residue_heads[:, [0, 1, 2, 3, 4, 6]] = 0.0
    residue_heads[:, 5] = 1.0 
    print("Set to zero weights = ", residue_heads.shape)
    zero_e = np.vstack((zero_e, residue_heads))
    print("Total zero weights = ", zero_e.shape)
    
    first_part = first_part[:numEdit, :]
    first_part[:numEdit,6] = weight
    first_part[:numEdit,5] = 1.0
    first_part[:numEdit,4] = 0.0
    first_part[:numEdit,3] = 0.0
    first_part[:numEdit,2] = inElectronSpace[:numEdit] # reset position # e_z
    first_part[:numEdit,1] = 0.0
    first_part[:numEdit,0] = 0.0
    print("Now heads = ", first_part.shape)
    print("___________________")
elif len(first_part) == numEdit:
    print("Number of heads equal initial numbers")
    first_part[:numEdit,6] = weight
    first_part[:numEdit,5] = 1.0
    first_part[:numEdit,4] = 0.0
    first_part[:numEdit,3] = 0.0
    first_part[:numEdit,2] = inElectronSpace[:numEdit] # reset position # e_z
    first_part[:numEdit,1] = 0.0
    first_part[:numEdit,0] = 0.0
    print("Now heads = ", first_part.shape)
    print("___________________")
else:
    print("Number of heads less than initial numbers")
    first_part[:,6] = weight
    first_part[:,5] = 1.0
    first_part[:,4] = 0.0
    first_part[:,3] = 0.0
    first_part[:,2] = inElectronSpace[:len(first_part)] # reset position # e_z
    first_part[:,1] = 0.0
    first_part[:,0] = 0.0
    print("Now heads = ", first_part.shape)
    print("___________________")
    
# first_part = np.vstack((first_part, main_part))

if zero_e.shape[0] == 0:
    rearranged_e = np.vstack((first_part, main_part, second_part))
else:
    rearranged_e = np.vstack((first_part, main_part, second_part, zero_e))

print("Total number of electron at the end = ", rearranged_e.shape)

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
# Plot p_j
ax1.plot(e_data[:, 2], (e_data[:, 5]-1)/rho, '.', markersize=0.5, color='tab:red')
ax1.plot(rearranged_e[:,2], (rearranged_e[:,5]-1)/rho, 'x', markersize=0.5, color='tab:green')

ax1.set_xlim(0)
ax1.set_xlabel(r'$z_2$')
ax1.set_ylabel('$p_j$')

# Adjust the plot layout
plt.tight_layout()
# Save the plot as a PNG file
print("Saving the plot...")
plt.plot()

output_filename = sys.argv[2]
plt.savefig(output_filename)

e_dataset = hdf5_file['/electrons']
e_dataset[...] = rearranged_e[rearranged_e[:,2].argsort()]
hdf5_file['/time'].attrs['vsStep'] = 0
hdf5_file['/time'].attrs['vsTime'] = 0.0
hdf5_file['/electrons'].attrs['iCsteps'] = 0
hdf5_file['/electrons'].attrs['iL'] = 0
hdf5_file['/electrons'].attrs['iWrite_cr'] = 0
hdf5_file['/electrons'].attrs['istep'] = 0
hdf5_file['/electrons'].attrs['time'] = 0.0
hdf5_file['/electrons'].attrs['zInter'] = 0.0
hdf5_file['/electrons'].attrs['zLocal'] = 0.0
hdf5_file['/electrons'].attrs['zTotal'] = 0.0
hdf5_file['/electrons'].attrs['zbarInter'] = 0.0
hdf5_file['/electrons'].attrs['zbarLocal'] = 0.0
hdf5_file['/electrons'].attrs['zbarTotal'] = 0.0

hdf5_file['/runInfo'].attrs['iCsteps'] = 0
hdf5_file['/runInfo'].attrs['iL'] = 0
hdf5_file['/runInfo'].attrs['iWrite_cr'] = 0
hdf5_file['/runInfo'].attrs['istep'] = 0
hdf5_file['/runInfo'].attrs['time'] = 0.0
hdf5_file['/runInfo'].attrs['zInter'] = 0.0
hdf5_file['/runInfo'].attrs['zLocal'] = 0.0
hdf5_file['/runInfo'].attrs['zTotal'] = 0.0
hdf5_file['/runInfo'].attrs['zbarInter'] = 0.0
hdf5_file['/runInfo'].attrs['zbarLocal'] = 0.0
hdf5_file['/runInfo'].attrs['zbarTotal'] = 0.0

hdf5_file.flush()
hdf5_file.close()
