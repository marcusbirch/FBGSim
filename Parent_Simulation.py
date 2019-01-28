

"""
                    ***MAIN SCRIPT***
This is the main script used to create iterative simulations
of the FBG Instrument. Given the set of input parameters
this script iteratively calls Spectrum_Observe and Child_FBG
For different parameters to be graphed the function inputs
must be altered where the parameters are instead arrays.
Outputs square matrix of Signal to Noise given 2 parameters
Signal, the output array should be parsed to Graphing.py for
a heatmap
"""

from __future__ import division 
import numpy as np
import math as m
import time
import Fibre_Bragg_Module as fbg
from Child_FBG_Function import Child_FBG


'''SIMULATION INPUTS'''
Model_Parameter_Variance = 'n'
one_dim = 'y'
#***MODEL PARAMETERS***#
s_T = 5700
s_g = 4.0
p_T = np.arange(500,2000,100)
p_g = 4.0


#***OBSERVATION PARAMETERS***#
distance_pc = 10 #Distance to Star in Parsecs
Stellar_Radii = 1 #Solar Radii
Planet_Radii = 1 #Solar Radii
rv=60 #Radial Velocity of Planet in km/s 

#***FACILITY PARAMETERS***#
R_tel = 20     #Telescope Radius in Metres
corona_atten = -3       

#***INSTRUMENT PARAMETERS***#
contrast = -7 #Planet is 10^Contrast dimmer than Star
E_lim=2.5E3 # Highest excepted E_low (photon equivalent energy)
S_lim=0.2E-22 #Lowest accepted line intensity
n_grains = 108000 #Number of ms per bin
delta_w = 5 #Stretching range +-

#***ITERATIVE PARAMETERS***#
param_dim = 500
contrast_v = np.linspace(-0.01,-10,param_dim)
n_grains_v = (np.linspace(1,30000,param_dim,endpoint=False)).astype(int)
l_width_v = np.linspace(0.01,0.25,param_dim) 


'''SETUP SIMULATION DATA'''
startTime18 = time.time()

#Construct Line List
CH4_l = fbg.Line_List(E_lim,S_lim)

'''SIMULATION STARTS HERE'''
#Arrays for output

if (Model_Parameter_Variance == 'n') & (one_dim =='n') :
    measure_row = np.zeros(shape=len(contrast_v))
    measure = [np.zeros(shape=len(contrast_v))]
    s_data = fbg.Import_Model(s_T,s_g,'star')
    p_data = fbg.Import_Model(1000,p_g,'planet')
    for i in range(len(contrast_v)):
        for j in range(len(contrast_v)):
            observation = fbg.Spectrum_Observe(p_data,\
                                             s_data,\
                                             R_tel,\
                                             distance_pc,\
                                             contrast_v[i],rv,\
                                             corona_atten)
            snr=Child_FBG(n_grains,\
                             CH4_l,\
                             observation,\
                             delta_w,\
                             rv)
            measure_row[j] = m.log(snr)
            print(f"{j+1} out of {len(p_T)} iterations completed for {i+1} Model")
        print(f"{i+1} out of {len(p_T)} Models simulated")      
        measure = np.vstack((measure, measure_row))
    print("Simulation time in Seconds:")
    print (time.time() - startTime18)
else:
    pass
    
if (Model_Parameter_Variance == 'n') & (one_dim =='y') :
    measure = np.zeros(shape=len(contrast_v))
    s_data = fbg.Import_Model(s_T,s_g,'star')
    p_data = fbg.Import_Model(1000,p_g,'planet')
    for i in range(len(contrast_v)):
        observation = fbg.Spectrum_Observe(p_data,\
                                             s_data,\
                                             R_tel,\
                                             distance_pc,\
                                             contrast_v[i],rv,\
                                             corona_atten)
        snr=Child_FBG(n_grains,\
                             CH4_l,\
                             observation,\
                             delta_w,\
                             rv)
        measure[i] = snr
        print(f"{i+1} out of {len(contrast_v)} observations simulated")
else:
    pass
    

if (Model_Parameter_Variance == 'y') & (one_dim =='n') :
    measure_row = np.zeros(shape=len(p_T))
    measure = [np.zeros(shape=len(p_T))]
    s_data = fbg.Import_Model(s_T,s_g,'star')
    for i in range(len(p_T)):
        p_data = fbg.Import_Model(p_T[i],p_g,'planet')
        for j in range(len(p_T)):
            observation = fbg.Spectrum_Observe(p_data,\
                                             s_data,\
                                             R_tel,\
                                             distance_pc,\
                                             contrast_v[j],\
                                             rv,\
                                             corona_atten)
            snr=Child_FBG(n_grains,\
                             CH4_l,\
                             observation,\
                             delta_w,\
                             rv)
            measure_row[j] = m.log(snr)
            print(f"{j+1} out of {len(p_T)} iterations completed for {i+1} Model")
        
        print(f"{i+1} out of {len(p_T)} Models simulated")
        measure = np.vstack((measure, measure_row))
    
    print("Simulation time in Seconds:")
    print (time.time() - startTime18)
else:
    pass
    

