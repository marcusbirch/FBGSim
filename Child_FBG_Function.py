

"""
                ***FBG FUNCTION***
Primary function for doing iterative simulations
Given observed data and instrumental parameters outputs the
Signal to Noise of the resultant signal for appending to a vector
"""

from __future__ import division
import numpy as np
from scipy import signal
import Fibre_Bragg_Module as fbg
from astropy.modeling import models, fitting
from astropy import stats
from astropy import constants as const

def Child_FBG(t,CH4_l,observation,delta_w,rv):
    
    noise_iterative = 'y'
    
    w = observation[:,0]
    sol_sys = observation[:,1]
    stellar_ref = observation[:,2]

    #***GRATING CONSTRUCTION***#
    grating_width = 0.01
    grating, l_n = fbg.Construct_Grating(w,CH4_l,grating_width)
    
    res = w[10] - w[9]
    stretch = np.arange(-delta_w/res,delta_w/res)
    stretch_axis = np.linspace( -delta_w, delta_w,num= len(stretch),endpoint=True)
    #***** DOPPLER SHIFT *****#
    velocities=((stretch_axis*const.c.value)/(np.median(w)))/1000
    if noise_iterative == 'y':
    
        if len(stretch) % 2 == 0:
            pass
        else:
            stretch=stretch[0:-1]
            stretch_axis=stretch_axis[0:-1]
        n_slices = 5
        slices=int(len(stretch)/n_slices)
        stretch_2d=np.reshape(stretch,(n_slices,slices))
    
        output_sys=[]
        output_ref=[]
        for i in range(n_slices):
            #Stretch grating over slice for combined spectra
            output_sys_slice = fbg.Grating_Stretch(sol_sys,stretch_2d,grating,\
                                              t,i)
            output_sys_slice = fbg.detector_throughput(output_sys_slice,t)
            #Stretch grating over slice for stellar spectra
            output_ref_slice = fbg.Grating_Stretch(stellar_ref,stretch_2d,grating,\
                                                     t,i)
            output_ref_slice = fbg.detector_throughput(output_ref_slice,t)
            #Splice grating functions
            output_sys=np.append(output_sys,output_sys_slice)
            output_ref=np.append(output_ref,output_ref_slice)
    
        
        #***POST PROCESSING***#
        #High Pass Filter
        window = int((1/2)*(delta_w/res))
        if window % 2 == 0:
            window = window + 1
        else:
            pass
        smoothing_sys= signal.savgol_filter(output_sys,window,3)
        sys_smooth=output_sys-smoothing_sys
        
        smoothing_ref = signal.savgol_filter(output_ref,window,3)
        ref_smooth=output_ref-smoothing_ref
        
        output_t = sys_smooth - ref_smooth
    else:
        #Shift System Spectrum with Time
        ccf_sys = signal.correlate(sol_sys,grating,mode='valid')
        ccf_ref = signal.correlate(stellar_ref,grating,mode='valid')
        
        noise_sys=np.random.normal(t,len(ccf_sys))
        noise_ref=np.random.normal(t,len(ccf_ref))
        
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
    #ENTER BAYES ROUTINE HERE
    return snr