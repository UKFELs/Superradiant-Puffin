import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg') # Don't display the plot

#def fwhm(x, y):
 #   ymax = np.max(y)
  #  half_max = ymax / 2.
   # left_crossing = x[np.where(y >= half_max)[0][0]]
    #right_crossing = x[np.where(y >= half_max)[0][-1]]
    #fwhm_value = right_crossing - left_crossing
    #return fwhm_value, left_crossing, right_crossing, ymax

def fwhm(x, y):
    ymax = np.max(y)
    half_max = ymax / 2.
    
    # Find indices where y crosses half_max
    crossings = np.where(y >= half_max)[0]

    # Left crossing is the first crossing
    left_crossing = x[crossings[0]]

    # Find right crossing
    right_crossing_index = crossings[0]
    for i in range(crossings[0], len(y)):
        if y[i] < half_max:
            right_crossing_index = i
            break
    right_crossing = x[right_crossing_index]

    fwhm_value = right_crossing - left_crossing

    return fwhm_value, left_crossing, right_crossing, ymax

def cal_variance(time, pulse_intensity):
    time = np.array(time)
    pulse_intensity = np.array(pulse_intensity)
    
    mean_time = np.average(time, weights=pulse_intensity)
    variance = np.average((time - mean_time)**2, weights=pulse_intensity)
    
    return variance

FWHM = []
as_peak = []
as_en = []
file_num = []
as_en_fwhm = []
as_variance = []

file_prefix = sys.argv[1] # for example SSS_ap_ 
file_start = int(sys.argv[2])
file_stop = int(sys.argv[3])
file_step = int(sys.argv[4])
z2_start = float(sys.argv[5])
z2_stop = float(sys.argv[6])

for i in range(file_start, file_stop, file_step):
    # file_name = f"D:\Puffin_results\gamma_100_rho0.079_helical\SSS_ap_{i}.h5"
    file_name = file_prefix + f'{i}.h5'
    print(file_name) 
    
    with h5py.File(file_name, 'r') as f:
        meshsizeZ2 = f['runInfo'].attrs['sLengthOfElmZ2']
        nz = f['runInfo'].attrs['nZ2']
        rho = f['runInfo'].attrs['rho']

        z2_lim = (z2_start, z2_stop)
        z2_lo_index = int(z2_lim[0]/meshsizeZ2)
        z2_up_index = int(z2_lim[1]/meshsizeZ2)

        aperp_x = f['aperp'][0, z2_lo_index:z2_up_index]
        aperp_y = f['aperp'][1, z2_lo_index:z2_up_index]
        ap_sqr = aperp_x**2 + aperp_y**2
        ap_int = np.trapz(ap_sqr, dx=meshsizeZ2)

        z2axis = np.arange(0, len(ap_sqr))*meshsizeZ2

        mfwhm, l_crossing, r_crossing, peak = fwhm(z2axis, ap_sqr)
        index_l = int(np.ceil(l_crossing/meshsizeZ2))
        index_r = int(np.ceil(r_crossing/meshsizeZ2))
        ap_int_fwhm = np.trapz(ap_sqr[index_l:index_r], dx=meshsizeZ2)
        
        cal_var = cal_variance(z2axis, ap_sqr)
        as_variance.append(cal_var)

        FWHM.append(mfwhm)
        as_peak.append(peak)
        as_en.append(ap_int)
        as_en_fwhm.append(ap_int_fwhm)
    
    file_num.append(i)

StepsPerPeriod = 40 # number of output file steps per period
zbar = (4*np.pi*rho)*np.array(file_num)/StepsPerPeriod
FWHM = np.array(FWHM)
as_peak = np.array(as_peak)
as_en = np.array(as_en)
as_en_fwhm = np.array(as_en_fwhm)
as_variance = np.array(as_variance)

mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.size'] = 14  # adjust as needed

# Write to file
output_file = "energy_power_FWHM.dat"
with open(output_file, "w") as file:
    file.write(f"{'zbar'}\t{'energy'}\t{'peak power'}\t{'energy_fwhm'}\t{'FWHM'}\t{'Variance'}\n")
    for zb, en, pp, en_fwhm, w_FWHM, w_variance in zip(zbar, as_en, as_peak, as_en_fwhm, FWHM, as_variance):
        file.write(f"{zb}\t{en}\t{pp}\t{en_fwhm}\t{w_FWHM}\t{w_variance}\n")

fig, axs = plt.subplots(2,1, figsize=(8,12))
axs[0].plot(zbar, as_peak, label=r'$|A|_{peak}^2$')
axs[0].plot(zbar, as_en, label=r'$\Sigma |A|^2$')
axs[0].plot(zbar, as_en_fwhm, label=r'$\Sigma |A|_{FWHM}^2$')

axs[0].set_xlim(min(zbar), max(zbar))
axs[0].set_ylim(0)
axs[0].set_xlabel(r'$\bar{z}$')
axs[0].set_ylabel('Scaled Power')
axs[0].legend()

axs[1].plot(zbar, FWHM, label='FWHM')
axs[1].set_xlim(min(zbar), max(zbar))
# axs[1].set_ylim(0)
axs[1].set_xlabel(r'$\bar{z}$')
axs[1].set_ylabel('Scaled FWHM')
axs[1].legend()

fig_name = "energy_power_FWHM.png"
fig.savefig(fig_name, pad_inches = 0.1, dpi=220)
# plt.show()
