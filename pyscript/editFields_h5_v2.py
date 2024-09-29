import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Don't display the plot
import matplotlib.pyplot as plt


filename_a = sys.argv[1]

hdf5_file = h5py.File(filename_a, 'r+')
# read electrons data set
a_data = hdf5_file['/aperp'][:]

def filter(data, start_index, length):
    if start_index < 0 or start_index > data.shape[1] or length < 0 or start_index + length > data.shape[1]:
        print("Invalid start index or length")
        return
    else:
        result = np.zeros(data.shape)
        for i in range(data.shape[0]):
            result[i, start_index:start_index + length] = data[i, start_index:start_index + length]
        return result
    
nz = hdf5_file['/runInfo'].attrs['nZ2']
meshsizeZ2 = hdf5_file['/runInfo'].attrs['sLengthOfElmZ2']
z2_bar = np.linspace(0, meshsizeZ2 * (nz - 1.0), nz)

samples = a_data.shape[1]
zero_arr = np.zeros(samples)

start_to_cut = float(sys.argv[3]) # unit of z2
zero_length = np.where(z2_bar >= start_to_cut)[0][0]

stop_to_cut = float(sys.argv[4])
to_cut =  np.where(z2_bar >= stop_to_cut)[0][0]
length_to_s_cut = to_cut - zero_length

s_filter = filter(a_data, zero_length, length_to_s_cut)

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
ax1.plot(z2_bar, a_data[0])
ax1.plot(z2_bar, s_filter[0])
ax1.set_xlabel(r'$z_2$')
ax1.set_ylabel('$A_x$')
# Adjust the plot layout
plt.tight_layout()
# Save the plot as a PNG file
print("Saving the field plot...")
output_filename = sys.argv[2]
plt.savefig(output_filename, dpi=300)


a_dataset = hdf5_file['/aperp']
a_dataset[...] = s_filter[:]


hdf5_file.flush()
hdf5_file.close()

