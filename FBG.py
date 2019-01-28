

'''
            Fibre Bragg Grating (FBG)               
Simulates passing a combined planet+stellar spectrum
through an FBG and measuring photometric intensity
of transmitted light given a number of CH4 gratings
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
from astropy import constants as const
import Fibre_Bragg_Module as fbg
startTime = time.time()

'''INPUT PARAMETERS FOR SIMULATION HERE'''
#***MODEL PARAMETERS***#
s_T = 5700
p_T = 1000
s_g = 4.0
p_g = 4.0

#***OBSERVATION PARAMETERS***#
contrast = -7 #Planet is 10^Contrast dimmer than Star
distance_pc = 10 #Distance to Star in Parsecs
Stellar_Radii = 1 #Solar Radii
Planet_Radii = 1 #Solar Radii
rv=150 #Radial Velocity of Planet in km/s 


#***FACILITY PARAMETERS***#
#Most facility parameters found in fbg.Upstream_Throughput.py
R_tel = 20 #Telescope Radius in Metres
corona_atten = -3 #Coronagraphic Attenuation

#***INSTRUMENT PARAMETERS***#
grating_width = 0.01
E_lim=2.5E3 # Highest excepted E_low (photon equivalent energy)
S_lim=0.2E-22 #Lowest accepted line intensity
n_grains = 108000 #Number of ms per bin
delta_w = 5 #Stretching range +-

import_models = 'n' # LINUX-SED 'D'->'E' for Raw PHOENIX GRIDS file
noise_iterative = 'y'

'''SIMULATION TAKES PLACE BELOW'''  
#***SPECTRAL PROCESSING***#
#Stellar Spectrum
if import_models=='y':
    s_data = fbg.Import_Model(s_T,s_g,'star')
    p_data = fbg.Import_Model(p_T,p_g,'planet')

else:
    pass

observation = fbg.Spectrum_Observe(p_data,s_data,R_tel,distance_pc,\
                                   contrast,rv,corona_atten)
w = observation[:,0]
sol_sys = observation[:,1]
stellar_ref = observation[:,2]


#***GRATING CONSTRUCTION***#
CH4_l = fbg.Line_List(E_lim,S_lim)

grating, l_n = fbg.Construct_Grating(w,CH4_l,grating_width)

#***GRATING STRETCH***#
#Initialise Stretching
res = w[10] - w[9]
stretch = np.arange(-delta_w/res,delta_w/res)
stretch_axis = np.linspace( -delta_w, delta_w,num= len(stretch),endpoint=True)
if len(stretch) % 2 == 0:
    pass
else:
    stretch=stretch[0:-1]
    stretch_axis=stretch_axis[0:-1]

#***** DOPPLER SHIFT *****#
velocities=((stretch_axis*const.c.value)/(np.median(w)))/1000

if noise_iterative == 'y':
    n_slices = 10
    slices=int(len(stretch)/n_slices)
    stretch_2d=np.reshape(stretch,(n_slices,slices))
    #Stretch Grating
    output_sys=[]
    output_ref=[]
    startTime14 = time.time()    
    for i in range(n_slices):
        #Stretch grating over slice for combined spectra
        output_sys_slice = fbg.Grating_Stretch(sol_sys,stretch_2d,grating,\
                                              n_grains,i)
        output_sys_slice = fbg.detector_throughput(output_sys_slice,n_grains)
        #Stretch grating over slice for stellar spectra
        output_ref_slice = fbg.Grating_Stretch(stellar_ref,stretch_2d,grating,\
                                                     n_grains,i)    
        output_ref_slice = fbg.detector_throughput(output_ref_slice,n_grains)
        #Splice grating functions
        output_sys=np.append(output_sys,output_sys_slice)
        output_ref=np.append(output_ref,output_ref_slice)
    print("Loop Time in seconds:")
    print (time.time() - startTime14)
    #***POST PROCESSING***#
    #High Pass Filter
    window = int((2/3)*(delta_w/res))
    if window % 2 == 0:
        window = window + 1
    else:
        pass

    smoothing_sys= signal.savgol_filter(output_sys,501,3)
    sys_smooth=output_sys-smoothing_sys
    smoothing_ref= signal.savgol_filter(output_ref,501,3)
    ref_smooth=output_ref-smoothing_ref
    #Subtract Reference Spectra
    output_t=sys_smooth-ref_smooth

else:
#Shift System Spectrum with Time
    ccf_sys = signal.correlate(sol_sys,grating,mode='full')
    ccf_ref = signal.correlate(stellar_ref,grating,mode='full')
    
    noise_sys=np.random.normal(n_grains,len(ccf_sys))
    noise_ref=np.random.normal(n_grains,len(ccf_ref))
            
    ccf_sys_noise = noise_sys*ccf_sys
    ccf_ref_noise = noise_ref*ccf_ref
           
    smoothing_sys= signal.savgol_filter(ccf_sys_noise,501,3)
    smoothing_ref = signal.savgol_filter(ccf_ref_noise,501,3)
        
    ccf_sys_smooth=ccf_sys_noise-smoothing_sys
    ccf_ref_smooth=ccf_ref_noise-smoothing_ref
            
    output_t = ccf_sys_smooth - ccf_ref_smooth
    
    ccf_r = np.linspace( -200, 200,num= 40001,endpoint=True) 
    output_t = output_t[(ccf_r > -delta_w) & (ccf_r < delta_w)]


#***FITTING***#
x0_init = -rv
amp_L_init = -np.max(abs(output_t[(velocities>-rv-15) & (velocities<-rv+15)]))
fwhm_L_init = 0.01/res
fwhm_G_init = 10
params=[x0_init,amp_L_init,fwhm_L_init,fwhm_G_init]
output_t, fitted_model, snr = fbg.Signal_Analysis(velocities,output_t,params)

real_time_hrs=np.round(((n_grains*len(stretch))/3.6E6),3) #in hours
print(f"Real Stretch Time in hours:{real_time_hrs}")

# plot data and fitted models
plt.plot(velocities, output_t,label="Data")
plt.plot(velocities, fitted_model(velocities), 'r--', label="Voigt Profile Fit")
plt.legend(loc=2, numpoints=1)
plt.xlabel('Velocity (km/s)')
plt.ylabel('Detected Photometric Intensity') 
plt.title(r'Planetary Light with RV=150km/s, S/N = %.1f, Observation Time=%.2f hrs'%(snr,real_time_hrs))
#plt.suptitle(r'Contrast=10$^{%.1f}$,Tel-Class=%.2f m,Corona-Attenuation = 10$^{%.3f}$' \
 #         %(contrast,R_tel*2,corona_atten), fontweight='bold')
plt.show()
print(f"{l_n} Amount of Lines in Grating")
