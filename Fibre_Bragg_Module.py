
from __future__ import division
import pandas as pd
import numpy as np
import numba as nb
import re
import shutil
import math as m
from tempfile import mkstemp
from astropy import constants as const
from astropy.io import fits
from astropy import stats
from astropy import convolution
from astropy.modeling import models, fitting

"""
        FORMAT MODEL FUNCTION
Performs the same function as a UNIX-SED
PHOENIX-GRIDS uses D for scientific notation,
e.g. 1D5 rather than 1E5. This function replaces
all D's with E's in the text file so that
it can be read by Python
"""
def Format_Model(raw_spectrum):
    def sed(pattern, replace, source, dest=None, count=0):
        fin = open(source, 'r')
        num_replaced = count
        if dest:
            fout = open(dest, 'w')
        else:
            fd, name = mkstemp()
            fout = open(name, 'w')
        for line in fin:
            out = re.sub(pattern, replace, line)
            fout.write(out)
            if out != line:
                num_replaced += 1
            if count and num_replaced > count:
                break
        try:
            fout.writelines(fin.readlines())
        except Exception as E:
            raise E
        fin.close()
        fout.close()
        try:
            shutil.move(name, source) 
        except WindowsError:
            pass
    sed('D','E',raw_spectrum)
    return raw_spectrum

"""
    IMPORT MODEL FUNCTION
This Function takes Temperature 
and Log(g) as args then imports
the corresponding PHOENIX spectra
Planet or Star has to be specified
under arguement 'body'
"""
def Import_Model(T,logg,body):
    T=T//100
    logg=np.round(logg,1)
    if body == 'planet':
        if T<10:
            model_string = \
            f".\Planet_Models\lte00{T}-{logg}-0.0a+0.0.BT-Settl.spec.7\lte00{T}-{logg}-0.0a+0.0.BT-Settl.spec.7"
        else:
            model_string = \
            f".\Planet_Models\lte0{T}-{logg}-0.0a+0.0.BT-Settl.spec.7\lte0{T}-{logg}-0.0a+0.0.BT-Settl.spec.7"
        
        #if open(model_string, 'r').read().find('D')==-1:
        #    print("Planet Already Formatted")
        #else:
        #    Format_Model(model_string)   
        data = np.genfromtxt(model_string,usecols=(0,1))
        
    else: 
        if T<10:
            model_string = \
            f".\Stellar_Models\lte00{T}-{logg}-0.0a+0.0.BT-Settl.spec.7\lte0{T}-{logg}-0.0a+0.0.BT-Settl.spec.7"
        else:
            model_string = \
            f".\Stellar_Models\lte0{T}-{logg}-0.0a+0.0.BT-Settl.spec.7\lte0{T}-{logg}-0.0a+0.0.BT-Settl.spec.7"
        
        #if open(model_string, 'r').read().find('D')==-1:
         #   print("Star Already Formatted")
        #else:
         #   Format_Model(model_string) 
        data = np.genfromtxt(model_string,usecols=(0,1))
    print('Model Imported Successfuly')
    return data

"""
   DOPPLER SHIFT FUNCTION
This Function takes a spectrum 
and shifts it given some velocity
in km/s
"""
def Doppler_Shift(w, flux, rv):
    #Radial Velocity in km/s
    rv = rv * 1000 #Convert to m/s
    doppler = np.sqrt((1+rv/const.c.value)/(1 - rv/const.c.value))
    w_shifted = w*doppler
    flux_shifted = np.interp(w_shifted, w, flux)
    return flux_shifted

"""
        SKY BACKGROUND FUNCTION
This function uses data from the Gemini Observatory
in Mauna Kea, Hawaii of Sky background emission from
OH lines, Zodiacal continuum emission(5800K black 
body scaled by the atmospheric transmission), and 
thermal emission from the atmosphere (273K black body).
The function optimistically assumes water vapour is 1mm,
and the airmass can be 1.0,1.5, or 2.0, but for larger
values, as is allowed for telluric transmission it
will auto set airmass to 2.0. This is called inside
Spectrum_Observe.
"""
def Sky_Background(w,air_mass):
    #Mauna-Kea Sky-Background Spectrum for 1mm water vapour column
    if air_mass <= 2:
        air_mass = int(air_mass*10)
        data_string = f".\Sky_Models\mk_skybg_zm_10_{air_mass}_ph.dat.txt"
    else:
        air_mass = int(2)
        data_string = f".\Sky_Models\mk_skybg_zm_10_{air_mass*10}_ph.dat.txt"
    
    data = np.genfromtxt(data_string,usecols=(0,1))
    w_sky = data[:,0] # nm
    photons_sky = data[:,1] # [ph.sec-1.arcsec-2.nm-1.m-2]
    photons_sky =photons_sky[(w_sky>3800) & (w_sky<4200)]
    w_sky = w_sky[(w_sky>3800) & (w_sky<4200)]
    photons_sky = np.interp(w,w_sky,photons_sky)
    return photons_sky

"""
     TELLURIC TRANSMISSION FUNCTION
This function imports a transmission
spectrum from Cerro Paranal at April/May
from the 'Advanced Cerro Paranal Sky Model'
published by ESO. It takes in a spectruma
and passes it through the transmission 
spectrum. Air_mass can be selected from
1.0-3.0. Water-column is auto-simulated 
prior. PWV = 0-20mm based on April/May
average precipitation.
"""
def Telluric_Transmission(w,spectrum,air_mass):
    #Telluric Spectrum
    air_mass = int(air_mass * 10)
    sky_fits = f".\Sky_Models\LBL_A{air_mass}_s3_R0300000_T.fits"
    hdul = fits.open(sky_fits)
    data = hdul[1].data
    w_sky = data['lam'] *1000
    sky = data['trans']
    sky = sky[(w_sky > 3800) & (w_sky < 4200)]
    w_sky = w_sky[(w_sky>3800) & (w_sky<4200)]
    sky = np.interp(w,w_sky,sky)
    spectrum_sky = spectrum * sky
    return spectrum_sky

"""
     UPSTREAM THROUGHPUT FUNCTION
This function uses throughput data from SCExAO by 
Jovanovic, 2015 to estimate transmission from various 
loss sources upstream of the FBG instrument. Most 
of the efficiencies are drawn from H-band data, 
as L-band is not available, and are thus only estimates
"""

def Upstream_Throughput(w,spectrum):
    #E is short for Efficiency#
    static_telescope_E = 0.94   #Loss from mirrors/misc. other pre AO loss
    ADC_E = 0.64 #Atmospheric Dispersion Compensator
    AO_E = 0.78 #AO_188 Suburu Facility AO
    instr_corona_E = 0.75 #Instrumental *NOT* Operational Coronagraphic Loss
    upstream_facility_E = AO_E*ADC_E*static_telescope_E*instr_corona_E
    
    #IR Bench SCExAO Instrument Efficiencies
    DM_E = 0.18 #Deformable Mirror
    OAP_E = 0.17 #Off Axis Parabolic Mirror
    beamsplitter_E = 0.20 #Fourth channel SCExAO Beamsplitter
    FIU_E = 0.10 #Fibre Injection Unit
    upstream_IR_bench_E = DM_E+OAP_E+beamsplitter_E+FIU_E
    
    #Total of all Efficenies
    end_to_end_throughput = upstream_facility_E * upstream_IR_bench_E
    spectrum_throughput = spectrum * end_to_end_throughput
    return spectrum_throughput
    

'''
    SPECTRUM OBSERVE FUNCTION
This Function turns formatted data from
PHOENIX GRIDS into observed data in photon
counts per ms per nm. It takes telescope 
diameter, distance to star, planetary RV, 
and planetary contrast as main arguements
'''
def Spectrum_Observe(p_data,s_data,R_tel,distance_pc,\
                     contrast,rv,corona_atten):
   
    air_mass = 1.0
    
    w = p_data[:,0] * 0.1 # [nm] from [Angstrom]
    Wmin = 3800
    Wmax = 4200
    a = np.where(w == Wmin)
    a = np.min(a)
    b = np.where(w == Wmax)
    b = np.min(b)
    w = w[a:b]
    
    flux_p = p_data[:,1]
    flux_s = s_data[:,1]
    
    flux_p = np.power((flux_p-8),10) #convert to Ergs/sec/cm2/A
    flux_s = np.power((flux_s-8),10) #convert to Ergs/sec/cm2/A
              
    E_photons = (const.h.value*const.c.value)/(w*10**(-9)) # [J]
    #total sphere divided by solid angle. where 4*pi factor cancels
    dilution_factor = (const.R_sun.value**2)/((distance_pc*const.pc.value)**2)
    
    flux_p = flux_p[a:b] * m.pow(10,-2) #[J.s-1.m-2.nm-1] conversion from ergs
    flux_s = flux_s[a:b] * m.pow(10,-2) #[J.s-1.m-2.nm-1] conversion from ergs

    flux_earth_p = flux_p *dilution_factor # [J.s-1.m-2.nm-1] (flux at earth per unit circle)
    flux_earth_s = flux_s *dilution_factor # [J.s-1.m-2.m-1] (flux at earth per unit circle)
    
    photons_earth_p = flux_earth_p / E_photons # [ph.s-1.m-2.nm-1] (photons at earth per unit circle)
    photons_earth_s = flux_earth_s / E_photons # [ph.s-1.m-2.nm-1] (photons at earth per unit circle)
    
    #Sky Background Emission from Gemini Observatory, Mauna Kea
    photons_sky = Sky_Background(w,air_mass) #Zodiacal, Thermal, Earthshine background
    
    photons_p = photons_earth_p * (m.pi * (R_tel**2)) # [ph.nm-1.s-1] (photons over Telescope)
    photons_s = photons_earth_s * (m.pi * (R_tel**2)) # [ph.nm-1.s-1] (photons over Telescope)
    photons_sky = photons_sky * (m.pi * (R_tel**2)) # [ph.nm-1.s-1] (photons over Telescope)
    
    #Bin size is 10pm
    photons_p = photons_p / len(w) # [ph.bin-1.s-1] (photons over Telescope per bin)
    photons_s = photons_s / len(w) # [ph.bin-1.s-1] (photons over Telescope per bin)
    photons_sky = photons_sky / len(w) # [ph.bin-1.s-1] (photons over Telescope per bin)

    #Define time-grain as 1ms
    photons_p = photons_p / 1000 # [ph.ms-1.10pm-1] per bin per millisecond
    photons_s = photons_s / 1000 # [ph.ms-1.10pm-1] per bin per millisecond
    photons_sky = photons_sky / 1000 # [ph.ms-1.10pm-1] per bin per millisecond
    
    #Until this point all operations are symmetric on the spectra
    #Scaling and Doppler of Planet
    photons_p = photons_p * m.pow(10,contrast) * (np.sum(photons_s)/np.sum(photons_p))
    photons_p = Doppler_Shift(w, photons_p, rv)
    
    #Coronagraphic Attenuation
    photons_s = photons_s *m.pow(10,corona_atten)
    
    #Spectrum of combined and reference spectrum
    sol_sys = photons_p + photons_s
    stellar_ref = photons_s
    
    Realistic_Loss = 'y'
    if Realistic_Loss == 'y':
        #Atmospheric Transmission from ESO SkyModel
        sol_sys = Telluric_Transmission(w,sol_sys,air_mass)
        stellar_ref = Telluric_Transmission(w,stellar_ref,air_mass)
        #Add Sky Background Emission
        sol_sys = sol_sys + photons_sky
        stellar_ref = stellar_ref + photons_sky
        #Throughput Efficiencies for Upstream Instrumentation
        sol_sys = Upstream_Throughput(w,sol_sys)
        stellar_ref = Upstream_Throughput(w,stellar_ref)
    else:
        pass
    
    observation=np.array([w,sol_sys,stellar_ref])
    observation=np.transpose(observation)
    return observation

"""
      LINE LIST FUNCTION
This Function creates a list of 
wavelengths where CH4 lines are
present. Takes line intensity
and transition energy criteria
"""
def Line_List(E_lim,S_lim):
    #***IMPORT LINE LIST***#
    methane = './Line_Lists/Hargreaves 2012 CH4 Hot Line List.txt'
    CH4_l_full = pd.read_csv(methane,header=None,sep=' ').values
    #Import Columns from Line List
    CH4_l = (CH4_l_full[:,1]).astype(float)
    CH4_S = (CH4_l_full[:,2]).astype(float)
    CH4_E = (CH4_l_full[:,3]).astype(float)
    CH4_Q = CH4_l_full[:,4]
    #Select for Quality
    CH4_l = CH4_l[(CH4_Q == 'H') | (CH4_Q=='1')]
    CH4_E = CH4_E[(CH4_Q == 'H') | (CH4_Q=='1')]
    CH4_S = CH4_S[(CH4_Q == 'H') | (CH4_Q=='1')]
    #Apply criterion to line catalogue
    CH4_l = CH4_l[(CH4_S >S_lim) & (CH4_E<E_lim)]
    CH4_l = (1/(CH4_l*100))/1E-9 #cm-1 to nm
    CH4_l = CH4_l[(CH4_l > 3800) & (CH4_l<4200)]
    CH4_l = np.unique(CH4_l)
    return CH4_l

"""
    CONSTRUCT GRATING FUNCTION
This function outputs a function over
the wavelength array from a line list
boolean and a Kernel convolution.
Kernel used is Super Gaussian.
Takes wavelength array and array of 
line indices from .line_list as input
"""
def Construct_Grating(w,CH4_l,grating_width):
    #***GRATING CONSTRUCTION***#
    spectral_reflectivity = 1
    #Kernel
    base = np.arange( -0.1, 0.1, step = 0.0001)
    base_res = base[1]-base[0]
    Super_Gaussian=np.exp(-1*((base/(m.sqrt(2)*grating_width))**6))
    grating_space = 2*(grating_width / base_res)
    #Oversample and Kernel Convolution
    CH4_l_oversample = np.round(CH4_l,3)
    w_oversample = np.linspace(3800,4200,num= 400000,endpoint=False)
    lines_oversample = np.zeros(shape=len(w_oversample))
    indices_oversample = np.where(np.isin(w_oversample, CH4_l_oversample))[0]
    #Grating Spaces
    line_dif = np.ediff1d(indices_oversample)
    line_dif = np.append(line_dif, 0) #make vectors equal length
    indices_oversample = indices_oversample[line_dif>2*grating_space]
    indices_oversample = indices_oversample[0::1]
    lines_oversample[indices_oversample] = spectral_reflectivity
    grating_oversample = np.convolve(Super_Gaussian,lines_oversample,mode='full')
    grating_oversample = grating_oversample[1000:-999]
    #Downsample
    grating = np.interp(w,w_oversample,grating_oversample)
    l_n = len(indices_oversample)
    return grating, l_n



"""
        GRATING STRETCH FUNCTION
This is function filters the spectral data 
through the constructed grating. It works
by doing element-wise multiplication using
partitioned rectangular matrices and 
sampling unique poisson noise for every 
measurement bin
"""
@nb.jit
def Grating_Stretch(data,stretch_2d,grating,n_grains,iterator):
        #Multidimensionalise Grating
        stretch_slice = (stretch_2d[iterator,:]).astype(int)
        grating_stacked=np.broadcast_to(grating, (len(stretch_slice), len(grating)))
        rows, column_indices = np.ogrid[:grating_stacked.shape[0], :grating_stacked.shape[1]]
        stretch_slice[stretch_slice < 0] += grating_stacked.shape[1]
        column_indices = column_indices - stretch_slice[:,np.newaxis]
        grating_stretched = grating_stacked[rows, column_indices]
        #Stack spectra
        data_stacked=np.broadcast_to(data, (len(stretch_slice), len(data)))
        #Create 2D noise matrices
        if n_grains<100: #Use the Poisson distribution for low integration time
            noise_matrix=np.random.poisson(n_grains, size=(len(stretch_slice), len(data)))
        else:
            noise_matrix=np.random.normal(n_grains, size=(len(stretch_slice), len(data)))
        noise_post_measure = 'y'
        if noise_post_measure == 'y':
            #Take the Hadamard product of stacked spectra and grating matrix
            measurement_data=np.multiply(grating_stretched,data_stacked)
            #Take the Hadamard product of the stretched grating matrix and the noise matrices
            detection_data=np.multiply(noise_matrix,measurement_data)
        else:
            #Take the Hadamard product of stacked spectra and noise spectra
            measurement_data=np.multiply(noise_matrix,data_stacked)
            #Take the Hadamard product of the noisy spectra and the stretched grating
            detection_data=np.multiply(grating_stretched,measurement_data)    
        #Flatten to a single column by summing elements of rows
        output=np.sum(detection_data, axis=1)
        return output

"""
   DETECTOR THROUGHPUT FUNCTION
This function considers the thermal
baseline, the number of dark events
per ms, and the quantum efficiency.
Detector is simulated as an Avalanche
photo-diode detector,
"""
def detector_throughput(data,n_grains):
    #Time Independent Thermal Baseline
    RN = 3.0 #e-1 , Readout Noise
    readout_noise = np.random.poisson(RN, len(data)) 
    #Detector counts with no input per ms
    dark_current = 0.01 #e/s
    dark_current = 0.01/1000 #e/ms
    dark_events = dark_current * np.random.normal(n_grains, size=(len(data)))
    #Quantum Efficiency of Avalanche Photodiode-based detector
    quantum_efficiency = 0.98
    
    data = data*quantum_efficiency + readout_noise + dark_events
    return data 

"""
        SIGNAL ANALYSIS FUNCTION
This function fits an upside-down Voigt 
Profile to the data, which is the convolution
of a Gaussian and a Lorentzian. After 
fitting the profile, the data is smoothed
by a boxcar filter to 1/3rd of the fit
size and the SNR is calculated using 
the peak/std_dev_noise 
"""
def Signal_Analysis(x,y,params):
    #Create fit using initial parameters
    #Stop fit from wandering onto random spikes of noise
    bound_centre = (params[0]-15,params[0]+15)
    bound_width = (params[3]-2,params[3]+30)
    #bound_amp = (0,params[1]),'amplitude_L': bound_amp
    bound_parameters = {'x_o': bound_centre,'fwhm_G': bound_width}
    fit_init = models.Voigt1D(params[0], params[1], params[2], params[3],bounds=bound_parameters)
    fit = fitting.LevMarLSQFitter()
    fitted_model = fit(fit_init, x, y)
    #Get value for fit peak (amplitude_L is not applicable)
    peak=abs(min(fitted_model(x)))
    #Post-Processing
    #Formula for voight width is found Olivero 1977
    fwhm_V = (fitted_model.fwhm_L/2) + m.sqrt((fitted_model.fwhm_L**2)/4 + fitted_model.fwhm_G**2)
    window_length = (1/3) * fwhm_V
    if window_length > 1:
        box_window = convolution.Box1DKernel(window_length)
        y = convolution.convolve(y,box_window)
    else:
        window_length = 2
        box_window = convolution.Box1DKernel(window_length)
        y = convolution.convolve(y,box_window)
    
    #Signal-to-Noise
    baseline = y - fitted_model(x) #not cheating....just to find the sigma of the non-peak reason lazily
    noise = stats.mad_std(baseline)
    snr = abs(peak)/noise
    if snr > 4:
        peak = min(y)
        snr = abs(peak)/noise
    else:
        pass
    
    return y, fitted_model, snr

