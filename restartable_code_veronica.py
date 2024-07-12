### Importing modules ###

import sys
import os
import glob 
import copy
import json
from os import listdir
from pathlib import Path

# Working with arrays 
import numpy as np

# Working with Data Frames
import pandas as pd

# use afino library
#sys.path.append("/Users/veronicaestrada/Downloads/Kazachenko/scripts/afino_release_version")
#from qpp import afino

#additional libraries needed for AFINO
import scipy.optimize as opt 
import scipy.signal as sig

# additional libraries needed for AFINO
#sys.path.append("/Users/veronicaestrada/Downloads/Kazachenko/scripts/afino_release_version")
#from afino import afino_start
#from afino import afino_spectral_models

# import stats library 
import scipy.stats as stats 
from scipy.stats import gamma
from scipy.special import gammaincc, gammainccinv
from scipy.signal import savgol_filter

# import the wavelet functions
sys.path.append("/Users/veronicaestrada/Downloads/Kazachenko_Lab/Project2/wavelets/wave_python")
import waveletFunctions as waveF


#This will be used to embed images in Jupyter Notebook

from IPython.display import Image
#While importing this is important to capitalize the I and P in IPython.
#if this is not done then this might be why there is an error



import sys 
#this is used to manipulate variable and functions. 
#According to online resources this is helpful mostly for run time

import os
#This is useful when dealing with differnent directories in python.
##It can remove, change, or help accese directories

import glob
#This can be helpful when trying to return a file path that follows 
#a specific parttern

from os import listdir
#Shows a file or list of all files and directories that is in a working directory

from os import path
#I am not sure what this one does



#This is for arrays
import numpy as np

#For fit files and WCS objects
import astropy.units as u
from astropy.time import Time


#plotting and drawing modules

from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns


#additional libraroes needed for AFINO

from scipy.io import readsav as idlsave

#import stats librarry

import scipy.stats as stats

### Basic Functions ###

def flx_check (flare_data):
    
    flag_flx = False

    for key in flare_data.keys():
        if key == "flx8":
            flag_flx = True
    return flag_flx


def recrate(flx, h): 
    '''compue the difference formula for f'(a) with step size h.
    
    parameters 
    -----------
    flx : array that contains reconnection flux 
    h : contant cadance between observationss
    
    Returns
    ---------
    recreate: reconnection flux(a+h) - reconnection flux(a-h)/2h 
            [negative reconnection rates, posative reconnection rate]
    '''
    
    recreate  = np.zeros((2, len(flx[1,:])))
    
    nflux       = flx[2]
    pflux      = flx[0]
    recreate[0] = np.gradient(nflux, h, edge_order = 2)
    recreate[1] = np.gradient(pflux, h, edge_order = 2) 
    
    return recreate


# Check if .sav file exists 
def check(filename,bstr,wvstr):
    '''
    
    inputs: 
            filename -- name of the reconnection flux idl save file
            bstr     -- 
                        BLOS - normal component taken as the line of sight
                        BRAD - normal component taken as the radial magnatic field (B_r)
            wvstr    -- 
    output: 
            savname  -- full filename with extension of the reconnection flux
    '''
    for i in range(len(bstr)): 
        if path.exists(filename+bstr[i]+wvstr+".sav" ) == True:
            return  filename+bstr[i]+wvstr+".sav"
        else: 
            print('File not found!')
            
            
def extract_ribbon_vars(idlstrct):
    '''
    Written by Marcel
    Oct. 5, 2020
    Inputs  --
    Outputs --  OPTIONAL (*) 
                *dosday --?
                flx6 -- magnetic reconnection flux for pixel area with I >= median(I)*6
                flx8 -- magnetic reconnection flux for pixel area with I >= median(I)*8
                flx10 -- magnetic reconnection flux for pixel area with I >= median(I)*10
                * area6 --?
                * area8 --? 
                * area10 --?
                * ar_area --? 
                * mflx --?
                tim -- UTC time from HMI 
                * cut --? 
                * sathr --? 
                * str_brad --? 
                * tim_maxnegrecrate --?    
                * tim_maxposrecrate --?       
                * maxnegrecrate --?        
                * maxposrecrate --?         
                n_o_satpix -- number of saturated points for a given AIA image          
                * bcenx6 --?               
                * bceny6 --?             
                * bcenx8 --?            
                * bceny8 --?            
                * bcenx10 --?             
                * bceny10 --?            
                * dst6 --?                
                * dst8 --?               
                * dst10 --?               

                
    
    Extract all the idl variables 
    '''
    
    flx6                = idlstrct.flx6 
    flx8                = idlstrct.flx8
    flx10               = idlstrct.flx10 
    tim                 = list(idlstrct.tim) #object arrays are not suported by python  
    n_o_satpix          = idlstrct.n_o_satpix
    # These have been commented out because they are not used in the qpp analysis 
    #dosday              = idlstrct.dosday
    area6               = idlstrct.area6 
    area8               = idlstrct.area8 
    area10              = idlstrct.area10 
    ar_area             = idlstrct.ar_area  
    #mflx                = idlstrct.mflx
    #cut                 = idlstrct.cut
    #satthr              = idlstrct.satthr
    #dflxdt              = idlstrct.dflxdt
    #str_brad            = idlstrct.str_brad

    #tim_maxnegrecrate   = idlstrct.tim_maxnegrecrate 
    #tim_maxposrecrate   = idlstrct.tim_maxposrecrate   
    #maxnegrecrate       = idlstrct.maxnegrecrate
    #maxposrecrate       = idlstrct.maxposrecrate 
    #bcenx6              = idlstrct.bcenx6
    #bceny6              = idlstrct.bceny6
    #bcenx8              = idlstrct.bcenx8
    #bceny8              = idlstrct.bceny8
    #bcenx10             = idlstrct.bcenx10
    #bceny10             = idlstrct.bceny10
    #dst6                = idlstrct.dst6
    #dst8                = idlstrct.dst8
    #dst10               = idlstrct.dst10
    
#    return dosday,flx8,flx6,flx10,area6,area8,area10,ar_area,mflx,tim,cut,satthr,dflxdt,str_brad,tim_maxnegrecrate,tim_maxposrecrate,maxnegrecrate,maxposrecrate,n_o_satpix,bcenx6,bceny6,bcenx8,bceny8,bcenx10,bceny10,dst6,dst8,dst10 
#    return dosday,flx8,flx6,flx10,area6,area8,area10,ar_area,mflx,tim,cut,satthr,str_brad,tim_maxnegrecrate,tim_maxposrecrate,maxnegrecrate,maxposrecrate,n_o_satpix,bcenx6,bceny6,bcenx8,bceny8,bcenx10,bceny10,dst6,dst8,dst10 
    return flx6,flx8,flx10,tim,n_o_satpix,area6,area8,area10,ar_area

def utc2jd(tim): 
    '''
    Written by Marcel 
    Oct. 2020
    
    Imput -- 
                tim -- UTC time from IDL sav file
    Output -- 
                jdt -- Julian days converted from UTC time  
                jst -- Julian days converted into seconds 
                js  -- Seconds from the start of the start date in Julian days
                to  -- Time Object dates in UTC-ISOT format
    '''
    jdt = list(np.zeros(len(tim)))
    jst = list(np.zeros(len(tim)))
    js  = list(np.zeros(len(tim)))
    sts = list(np.zeros(len(tim)))

    for i in range(len(tim)): 
        utcstr = str(tim[i])
        utcstr = utcstr.replace('b',"")
        utcstr = utcstr.replace("'",'') 
        sts[i] = utcstr #array to format date that is read by astropy
        jdt[i] = Time(utcstr,format='isot')
        jdt[i] = jdt[i].jd
        jst[i] = jdt[i] * 24 * 3600
        js[i]  = jst[i] - jst[0]
    jdt = np.array(jdt)
    jst = np.array(jst)
    js  = np.array(js)
    to  = Time(sts,scale='utc',format='isot') 
    return jdt,jst,js,to

def extract_RDB(flare_data):
    #extract all variables from RibbonDB IDL savefiles 
    rdbflx6,rdbflx8,rdbflx10,tim,n_o_satpix,area6,area8,area10,ar_area = extract_ribbon_vars(flare_data)
    rdbflx = (rdbflx6+rdbflx10)/2 # reconnection flux in Maxwells
   
    #changes from utc to julian calander (in seconds; see ribbon_functions.py)
    _,_,rdbt,rdbtt = utc2jd(tim) #time is stored in a time stamp from year, month, day, etc...
    
    # Evaluate the reconnection rate
    #   negative recflux = rflx[0,:]    
    #   positive recflux = rflx[1,:]
    rdbrflx  = recrate(rdbflx,rdbt[1]-rdbt[0]) # reconnection rate 
                                               # in units of Maxwells per second
    
    #conversion of time to minutes
    rdbt = rdbt/60
       
    return rdbflx,rdbrflx,rdbt,rdbtt,area8,n_o_satpix,ar_area


### Models ###  (possibly adapt into a function)
def power_law_model(a, f):
    return a[0] * f**a[1]

def colored_noise_model(a,f):
    return a[0] * f**a[1]  + a[2] 

def complex_noise_model(a,f): 
    
     return a[0] * f**a[1]  + a[2] +  a[3] * ( 1 + f**2/( a[4]*a[5]**2 ) )**( -(a[4]+1) / 2)
    
def power_law_decay_model(a, f):
    """
    Power Law + Constant Power + Power Law Decay Model
    Y = A_pl * f^a + A_cp + A_pd * f^(-a)
    log(Y) = log(A_pl * f^a + A_cp + A_pd * f^(-a))
    log(Y) = a * log(f) + log(A_pl + A_cp + A_pd * f^(-a))

    Parameters:
        frequencies: Frequency values [array]
        true_params: List containing model parameters[list]
              true_params[0]: Amplitude of power law
              true_params[1]: Power law index
              true_params[2]: Amplitude of constant power
              true_params[3]: Amplitude of power law decay
              true_params[4]: Decay frequency of power law decay

    Returns:
      Synthetic data following the power law + constant power + power law decay model[array]
    """
    return a[0] * f**a[1] + a[2] + a[3] / (1 + (f / a[4])**2)


def power_law_decay_gaussian_model(a, f):
    """
    Power Law + Constant Power + Power Law Decay + Gaussian Enhancement Model
    Y = A_pl * f^a + A_cp + A_pd * f^(-a) + A_g * exp(-(f - f_g)^2 / (2 * sigma_g^2))
    log(Y) = log(A_pl * f^a + A_cp + A_pd * f^(-a) + A_g * exp(-(f - f_g)^2 / (2 * sigma_g^2)))
    log(Y) = a * log(f) + log(A_pl + A_cp + A_pd * f^(-a) + A_g * exp(-(f - f_g)^2 / (2 * sigma_g^2)))


    Parameters:
        frequencies: Frequency values [array]
        true_params: List containing model parameters[list]
              true_params[0]: Amplitude of power law
              true_params[1]: Power law index
              true_params[2]: Amplitude of constant power
              true_params[3]: Amplitude of power law decay
              true_params[4]: Decay frequency of power law decay
              true_params[5]: Amplitude of Gaussian enhancement
              true_params[6]: Mean of Gaussian
              true_params[7]: Standard deviation of Gaussian

    Returns:
      Synthetic data following the power law + constant power + power law decay + Gaussian enhancement model [array]
      
      ( 1 + f**2/( a[4]*a[5]**2 ) )**( -(a[4]+1) / 2)
    """
    return a[0] * f**a[1] + a[2] + a[3] / (1 + (f / a[4])**2) + a[5] * np.exp(-(f - a[6])**2 / (2 * a[7]**2))

#### Noise Fitting Functions ####

# def noise fitting function
def randomize_initial_guess(): 
    plaw_amp = np.random.random(1) * (-10) + 10 # power law amplitude [0, 10]
    plaw_idx = np.random.random(1) * (-8) + 4   # power law index [-4, 4] 
    plaw_cts = np.random.random(1) * (-10) + 10 # power law constant [0, 10]
    kapp_amp = np.random.random(1) * (-20) + 10 # kappa function amplitude [-20, 10]
    kapp_idx = np.random.random(1) * (-4) + 4   # kappa function index [-4,4] 
    kapp_rho = np.random.random(1) - 1          # kappa function frequency normilizer [-1,1]
    
    #[0.05, 1d-4, -2.0, 20, 10.0, 0.1] 
    initial_guess = [plaw_amp[0], plaw_idx[0], plaw_cts[0], kapp_amp[0], kapp_idx[0], kapp_rho[0]]

    return initial_guess
    


def lnlike(a, f, data, model_function):
    model = model_function(a, f)
    epsilon = 1e-10  # Small offset to avoid zeros
    return -np.sum(np.log(np.abs(model) + epsilon)) - np.sum(data / (np.abs(model) + epsilon))



def BIC(k, variables, f, power, model_function, n):
    '''
    input:
    
    k: This is the number of free parameters in your model. It penalizes models with more parameters.

    variables: The parameters of your model. These are the values that the model uses to fit the data.

    f: The frequencies of your data.

    power: The power spectrum of your data.

    model_function: The function that defines your model. It takes variables and f as inputs and returns the model predictions.

    n: The number of data points.

    output:

    -2 * lnlike(variables, f, power, model_function): This is a common factor in BIC calculations. The factor of 2 comes from the assumption of normally distributed errors.

    k * np.log(n): This is the penalty term for the number of parameters in the model. It penalizes more complex models. The np.log(n) term accounts for the sample size.

    The entire expression is returned as the BIC score.'''
    return -2 * lnlike(variables, f, power, model_function) + k * np.log(n)

def rhoj(Sj, shatj):
    """
    Sample to Model Ratio (SMR) estimator (Eq. 5)

    Parameters
    ----------

    Sj
        random variables (i.e. data)
    shatj
        best estimate of the model. Should be same length as Sj

    Returns
    -------
    ndarray
        The Sample-to-Model ratio
    """
    return Sj / shatj

def rchi2_calc(m, nu, rhoj):
    """
    Goodness-of-fit estimator (Eq. 16)

    Parameters
    ----------

    m    - number of spectra considered
    nu   - degrees of freedom
    rhoj - sample to model ratio estimator

    Returns
    -------
    float - A chi-square like goodness of fit estimator
    """
    return (m / (1.0 * nu)) * np.sum((1.0 - rhoj) ** 2)


def prob_this_rchi2_or_larger(rchi2, m, nu):
    """
    :param rchi2: reduced chi-squared value
    :param m: number of spectra considered
    :param nu:  degrees of freedom
    :return:
    """
    a = (nu / 2.0) * np.float64(m) / (3.0 + np.float64(m))
    return gammaincc(a, a * rchi2)

def noise_fitting(f,power,model='colored', noise_prams = None, include_sinusoid = False):
    
    #---------------------------------------
    #use fitting methods from scipy optimize
    best_lnlike = -999999.0
    #best_param_vals = -999999.0
    # --------------------------------------------------------
    if model == 'colored':

        #don't have a good idea of starting guess parameters so randomize these and do multiple trials to cover more parameter space
        for i in range(0,20):

            #get randomized initial guess params to input into fit
            guess = randomize_initial_guess()
            #guess = [0,-4,0]
            

            #try 3 different fitting algorithms to ensure we maximize the likelihood
            for method in ['L-BFGS-B','TNC','SLSQP']:
                x0     = [guess[0],guess[1],guess[2]]
                nll    = lambda *args: -lnlike(*args)
                args   = (f,power,colored_noise_model)
                #bounds = [(-1e20,1e20),(-10,-1),(-1e20,1e20)]
                 
                res         = opt.minimize(nll, x0, args=args, method=method)
                param_vals  = res['x']
                jack_lnlike = lnlike(param_vals,f,power,colored_noise_model)
                
            
                if jack_lnlike > best_lnlike:
                    best_lnlike     = copy.deepcopy(jack_lnlike)
                    best_param_vals = copy.deepcopy(param_vals)
                    

        #calculate BIC and store best fit power spectrum and params
        jack_bic                = BIC(3,best_param_vals,f,power,colored_noise_model,len(power))
        best_fit_power_spectrum = colored_noise_model(best_param_vals,f)
        best_fit_params         = best_param_vals
        
    #-------------------------------------    
    if model == 'complex':

        #don't have a good idea of starting guess parameters so randomize these and do multiple trials to cover more parameter space
        for i in range(0,20):

            #get randomized initial guess params to input into fit
            guess = randomize_initial_guess()

            #try 3 different fitting algorithms to ensure we maximize the likelihood
            for method in ['L-BFGS-B','TNC','SLSQP']:
                x0   = guess
                nll  = lambda *args: -lnlike(*args)
                args = (f,power,complex_noise_model)
                
                res         = opt.minimize(nll, x0, args=args, method=method)
                param_vals  = res['x']
                jack_lnlike = lnlike(param_vals,f,power,colored_noise_model)
            
                if jack_lnlike > best_lnlike:
                    best_lnlike     = copy.deepcopy(jack_lnlike)
                    best_param_vals = copy.deepcopy(param_vals)

        #calculate BIC and store best fit power spectrum and params
        jack_bic                = BIC(6,best_param_vals,f,power,complex_noise_model,len(power))
        best_fit_power_spectrum = complex_noise_model(best_param_vals,f)
        best_fit_params         = best_param_vals
          
    #----------------------------------------
    
    #also want to find the goodness of fit for the best model
    smr = rhoj(power,best_fit_power_spectrum)
    if model == 'colored':
        deg_free = len(power) - 3
        rchi2    = rchi2_calc(1, deg_free, smr)
        prob     = prob_this_rchi2_or_larger(rchi2, 1, deg_free)
    elif model == 'complex':
        deg_free = len(power) - 6
        rchi2    = rchi2_calc(1, deg_free, smr)
        prob     = prob_this_rchi2_or_larger(rchi2, 1, deg_free)
    
    #----------------------------------------
    
    #now have the best fit, likelihood and BIC value. Return all these in a dictionary
    #will save combined fit results (both models) to a pickle file in the next level up.
    fitresults = {}
    fitresults['lnlike']                  = best_lnlike #likelihoods[selection_index]
    fitresults['model']                   = model
    fitresults['BIC']                     = jack_bic
    fitresults['best_fit_power_spectrum'] = best_fit_power_spectrum
    fitresults['frequencies']             = f
    fitresults['power']                   = power
    fitresults['params']                  = best_fit_params
    fitresults['rchi2']                   = rchi2
    fitresults['probability']             = prob
    
    return fitresults
            
    


### Wavelet Function ####
def WaveletAnalysis(time, Flux_p, Flux_n, Flare_name, save_path=None):
    ext = Flare_name
    #print(len(time))
    #print(len(Flux_p))
    # Validate input parameters
    if len(time) != len(Flux_p):
        raise ValueError("Input arrays 'Time' and 'Flux Positive' must have the same length.")
    if len(time) != len(Flux_n):
        raise ValueError("Input arrays 'Time' and 'Flux Negative' must have the same length.")
    
    dt = time[1] - time[0]
    pad = 1
    dj = 1/8
    s0 = 2*dt
    j1 = -1
    mother = 'MORLET'
    N = len(time)
    siglvl = 0.95
    k0 = 6

    # Apply filter to the noisy signal
    window_length = 7  # Window length for the filter
    polyorder = 3      # Polynomial order
    y_filtered_p = savgol_filter(Flux_p, window_length, polyorder)
    y_filtered_n = savgol_filter(Flux_n, window_length, polyorder)

    flux_p = Flux_p - y_filtered_p # Slow moving flux claculated to remove the backgrond
    flux_n = Flux_n - y_filtered_n # Slow moving flux claculated to remove the backgrond
    
    # normalize the reconnection rate, Fermi flux, GOES flux rate 
    norm_flux_p = (flux_p-np.mean(flux_p))/np.sqrt(np.var(np.abs(flux_p))) # norm. flux/rate
    norm_flux_n = (flux_n-np.mean(flux_n))/np.sqrt(np.var(np.abs(flux_n)))

    # Wavelet transform Analysis    
    Wcoef_p, Wperiod_p, Wscale_p, Wcoi_p = waveF.wavelet(norm_flux_p, dt, pad, dj, s0, j1, mother) # extract WT variables 
    Wpower_p                       = (np.abs(Wcoef_p)) ** 2                                  # compute wavelet power spectrum
    Wpower_tave_p                  = (np.sum(Wpower_p, axis=1) / N)                          # time-average Wavelet Power

    Wcoef_n, Wperiod_n, Wscale_n, Wcoi_n = waveF.wavelet(norm_flux_n, dt, pad, dj, s0, j1, mother) # extract WT variables 
    Wpower_n                       = (np.abs(Wcoef_n)) ** 2                                  # compute wavelet power spectrum
    Wpower_tave_n                  = (np.sum(Wpower_n, axis=1) / N)                          # time-average Wavelet Power

   # extract noise fit from time-series

    noise_fit_p = noise_fitting((1/Wperiod_p),Wpower_tave_p, model='colored')

    global_ws_p = (colored_noise_model(noise_fit_p['params'],(1/Wperiod_p)))
    global_ws_tave_p = (colored_noise_model(noise_fit_p['params'],(1/Wperiod_p)))

    # Negative #
    noise_fit_n = noise_fitting((1/Wperiod_n),Wpower_tave_n, model='colored')

    global_ws_n = (colored_noise_model(noise_fit_n['params'],(1/Wperiod_n)))
    global_ws_tave_n = (colored_noise_model(noise_fit_n['params'],(1/Wperiod_n)))

    # calculate the local significance for the power law noise model WT 
    local_powerlaw_signif_p = waveF.wave_signif(norm_flux_p, dt, Wscale_p, sigtest = 0, siglvl=siglvl, gws=global_ws_p)
    local_powerlaw_signif_p = local_powerlaw_signif_p[:, np.newaxis].dot(np.ones(N)[np.newaxis, :]) # make 2D matrix
    local_powerlaw_signif_p = Wpower_p / local_powerlaw_signif_p                                      # normalize, sig. > 1 is relevant 
                                   # normalize, sig. > 1 

    # local confidence level for the time-averaged wavelet spectrum
    tave_dof_p                   = N - Wscale_p/dt                                          # number of dof 
    tave_local_powerlaw_signif_p = waveF.wave_signif(norm_flux_p, dt, Wscale_p, sigtest = 1, siglvl=siglvl, dof=tave_dof_p, gws=global_ws_p)

    # Negative #
    # calculate the local significance for the power law noise model WT 
    local_powerlaw_signif_n = waveF.wave_signif(norm_flux_n, dt, Wscale_n, sigtest = 0, siglvl=siglvl, gws=global_ws_n)
    local_powerlaw_signif_n = local_powerlaw_signif_n[:, np.newaxis].dot(np.ones(N)[np.newaxis, :]) # make 2D matrix
    local_powerlaw_signif_n = Wpower_n / local_powerlaw_signif_n                                      # normalize, sig. > 1 is relevant 
                                   # normalize, sig. > 1 

    # local confidence level for the time-averaged wavelet spectrum
    tave_dof_n                   = N - Wscale_n/dt                                          # number of dof 
    tave_local_powerlaw_signif_n = waveF.wave_signif(norm_flux_n, dt, Wscale_n, sigtest = 1, siglvl=siglvl, dof=tave_dof_n, gws=global_ws_n)
   
    # define the power above significance levels
    localW_pwr_p  = Wpower_tave_p - tave_local_powerlaw_signif_p  # wavelet power rel. to local noise level

    # identify indices of peaks in the spectral transforms power
    Wpks_p = sig.find_peaks(Wpower_tave_p[Wperiod_p<Wcoi_p.max()], height=0.1)[0]
    Wpks_p = np.append(Wpks_p,sig.find_peaks(localW_pwr_p[Wperiod_p<Wcoi_p.max()],height=0.1)[0])

    # define the periods from significance level analysis    
    if len(Wpks_p) > 0:  
        locWi_p = np.argmin(np.abs(localW_pwr_p  - np.max(localW_pwr_p[Wpks_p])) )
    else:
        locWi_p = np.argmin(np.abs(localW_pwr_p  - np.max(localW_pwr_p)) )

    period_localW_p  = Wperiod_p[locWi_p] # period from local wavelet power
    
    Wimax_p = np.argmin(np.abs(Wperiod_p - period_localW_p))

    # initialize the flag variables      
    period_localW_flag_p  = 0 # period from local wavelet power flag

    # period from local wavelet power flag    
    if localW_pwr_p[locWi_p] > 0: 
        period_localW_flag_p  = 1 
    
    # Negative #

    # define the power above significance levels
    localW_pwr_n  = Wpower_tave_n - tave_local_powerlaw_signif_n  # wavelet power rel. to local noise level

    # identify indices of peaks in the spectral transforms power
    Wpks_n = sig.find_peaks(Wpower_tave_n[Wperiod_n<Wcoi_n.max()], height=0.1)[0]
    Wpks_n = np.append(Wpks_n,sig.find_peaks(localW_pwr_n[Wperiod_n<Wcoi_n.max()],height=0.1)[0])

    # define the periods from significance level analysis    
    if len(Wpks_n) > 0:  
        locWi_n = np.argmin(np.abs(localW_pwr_n  - np.max(localW_pwr_n[Wpks_n])) )
    else:
        locWi_n = np.argmin(np.abs(localW_pwr_n  - np.max(localW_pwr_n)) )

    period_localW_n  = Wperiod_n[locWi_n] # period from local wavelet power

    Wimax_n = np.argmin(np.abs(Wperiod_n - period_localW_n))

    # initialize the flag variables      
    period_localW_flag_n  = 0 # period from local wavelet power flag

    # period from local wavelet power flag    
    if localW_pwr_n[locWi_n] > 0: 
        period_localW_flag_n  = 1  
    # Graphing components
    if save_path:
        axd = plt.figure(figsize=(30, 30), constrained_layout=True).subplot_mosaic(
        """
        AAA...DDD.
        BBBC..EEEF
        BBBC..EEEF
        BBBC..EEEF
        """)

        fsize = 32
        bwth = 10


        sub1 = axd['A']
        sub2 = axd['B']
        sub3 = axd['C']
        sub4 = axd['D']
        sub5 = axd['E']
        sub6 = axd['F']
        #sub7 = axd['G']
        #sub8 = axd['H']

        # Time series
        sub1.plot(time, norm_flux_p, marker='D', color='black', linestyle='dashed', mfc='blue', label='Time Series')
        sub1.set_title('Time Series', fontsize=30)
        sub1.set_ylabel('Flux ', fontsize=25)
        sub1.set_xlabel('Time', fontsize=25)

        # Wavelet power spectrum
        CS = sub2.pcolormesh(time, Wperiod_p, Wpower_p, cmap='Blues', vmin=0, vmax=2)
        sub2.plot(time, Wcoi_p, 'r')  # cone of influence (The area's the signal is most reliable)
        sub2.set_xlabel('Time', fontsize=25)
        sub2.set_ylabel('Period', fontsize=25)
        sub2.set_title('Wavelet Power Spectrum', fontsize=30)
        sub2.fill_between(time, Wcoi_p * 0 + Wperiod_p[-1], Wcoi_p, facecolor="none",
                  edgecolor="#00000040", hatch='x')
        sub2.contour(time, Wperiod_p, local_powerlaw_signif_p,levels = [0.25, 0.5, 0.75, 1, 1.25, 1.50], vmin=0, vmax=2, cmap = 'Reds', linewidths= 2, zorder=3)

        sub2.set_ylim((1, 10))
        cbar = plt.colorbar(CS, location='bottom', ax=sub2)  # Pass ax=sub2
        cbar.set_label('Power ($\sigma^2$)', labelpad=10, fontsize=25)


        # Add the interpolated significance levels to the second subplot
        sub3.plot(tave_local_powerlaw_signif_p, Wperiod_p, 'k--', label='Significance Level')

        # Average wavelet power spectrum
        sub3.plot(Wpower_tave_p, Wperiod_p, marker='D', color='blue', linestyle='solid', mfc='blue', label='Average Wavelet Power')
        sub3.axhline(y = period_localW_p, xmin=0, xmax=1, color='k', linestyle='dashed',
             label='$\mathcal{P} =$' + "{:10.3f}".format(period_localW_p))
        sub3.set_ylim((1, 10))
        sub3.set_xlim((0, 5))
        sub3.set_title('Avg. Wavelet Power', fontsize=30)
        sub3.set_xlabel('Power ($\sigma^2$)', fontsize=25)
        sub3.set_ylabel('Period', fontsize=25)
        legend = sub3.legend(fontsize=13, loc='lower right', bbox_to_anchor=(1.0, 0.0))

        
        
        # Time series
        sub4.plot(time, norm_flux_n, marker='D', color='black', linestyle='dashed', mfc='blue', label='Time Series')
        sub4.set_title('Time Series', fontsize=30)
        sub4.set_ylabel('Flux ', fontsize=25)
        sub4.set_xlabel('Time', fontsize=25)


        # Wavelet power spectrum
        CS = sub5.pcolormesh(time, Wperiod_n, Wpower_n, cmap='Blues', vmin=0, vmax=2)
        sub5.plot(time, Wcoi_n, 'r')  # cone of influence (The area's the signal is most reliable)
        sub5.set_xlabel('Time', fontsize=25)
        sub5.set_ylabel('Period', fontsize=25)
        sub5.set_title('Wavelet Power Spectrum', fontsize=30)
        sub5.fill_between(time, Wcoi_n * 0 + Wperiod_n[-1], Wcoi_n, facecolor="none",
        edgecolor="#00000040", hatch='x')
        sub5.contour(time, Wperiod_n, local_powerlaw_signif_n,levels = [0.25, 0.5, 0.75, 1, 1.25, 1.50], vmin=0, vmax=2, cmap = 'Reds', linewidths= 2, zorder=3)
        sub5.set_ylim(np.min(Wperiod_n), np.max(Wcoi_n))


        cbar = plt.colorbar(CS, location='bottom', ax=sub5)  # Pass ax=sub2
        cbar.set_label('Power ($\sigma^2$)', labelpad=10, fontsize=25)

        # Add the interpolated significance levels to the second subplot
        sub6.plot(tave_local_powerlaw_signif_n, Wperiod_n, 'k--', label='Significance Level')

        # Average wavelet power spectrum
        sub6.plot(Wpower_tave_n, Wperiod_n, marker='D', color='blue', linestyle='solid', mfc='blue', label='Average Wavelet Power')
        sub6.axhline(y= period_localW_n, xmin=0, xmax=1, color='k', linestyle='dashed',
             label='$\mathcal{P} =$' + "{:10.3f}".format(period_localW_n))
        sub6.set_ylim((np.min(Wperiod_n), np.max(Wcoi_n)))
        sub6.set_xlim((np.min(Wpower_tave_n),np.max(Wpower_tave_n)))
        sub6.set_title('Avg. Wavelet Power', fontsize=30)
        sub6.set_xlabel('Power ($\sigma^2$)', fontsize=25)
        sub6.set_ylabel('Period', fontsize=25)
        legend = sub6.legend(fontsize=13, loc='lower right', bbox_to_anchor=(1.0, 0.0))

        # 2D Analysis
        #sub4 = axd['D']
        #sub4.plot(Wpower, color='green', linestyle='solid', label='2D Analysis')
        #sub4.axhline(y = period_localW, xmin=0, xmax=1, color='k', linestyle='dashed',
        #label='$\mathcal{P} =$' + "{:10.3f}".format(period_localW))
        #sub4.set_title('2D Analysis', fontsize=30)
        #sub4.set_xlabel('Period', fontsize=25)
        #sub4.set_ylabel('Power ($\sigma^2$)', fontsize=25)
        #sub4.set_xlim((1, 4))
        #legend2d = sub4.legend(fontsize=13, loc='upper right')

        # plt.show()

        plt.savefig(save_path + ext + '_Wavlets.tiff' , bbox_inches='tight')
        plt.close()
    else:
        axd = plt.figure(figsize=(30, 30), constrained_layout=True).subplot_mosaic(
        """
        AAA...DDD.
        BBBC..EEEF
        BBBC..EEEF
        BBBC..EEEF
        """)

        fsize = 32
        bwth = 10


        sub1 = axd['A']
        sub2 = axd['B']
        sub3 = axd['C']
        sub4 = axd['D']
        sub5 = axd['E']
        sub6 = axd['F']
        #sub7 = axd['G']
        #sub8 = axd['H']

        # Time series
        sub1.plot(time, norm_flux_p, marker='D', color='black', linestyle='dashed', mfc='blue', label='Time Series')
        sub1.set_title('Time Series ' + ext, fontsize=30)
        sub1.set_ylabel('Flux', fontsize=25)
        sub1.set_xlabel('Time', fontsize=25)

        # Wavelet power spectrum
        CS = sub2.pcolormesh(time, Wperiod_p, Wpower_p, cmap='Blues', vmin=0, vmax=2)
        sub2.plot(time, Wcoi_p, 'r')
        sub2.set_xlabel('Time', fontsize=25)
        sub2.set_ylabel('Period', fontsize=25)
        sub2.set_title('Wavelet Power Spectrum ' + ext, fontsize=30)
        sub2.fill_between(time, Wcoi_p * 0 + Wperiod_p[-1], Wcoi_p, facecolor="none", edgecolor="#00000040", hatch='x')
        sub2.contour(time, Wperiod_p, local_powerlaw_signif_p,levels = [0.25, 0.5, 0.75, 1, 1.25, 1.50], vmin=0, vmax=2, cmap = 'Reds', linewidths= 2, zorder=3)
        sub2.set_ylim((0, 4))
        cbar = plt.colorbar(CS, location='bottom', ax=sub2)
        cbar.set_label('Power ($\sigma^2$)', labelpad=10, fontsize=25)

        # Add the interpolated significance levels to the second subplot
        sub3.plot(tave_local_powerlaw_signif_p, Wperiod_p, 'k--', label='Significance Level')

        # Average wavelet power spectrum
        sub3.plot(Wpower_tave_p, Wperiod_p, marker='D', color='blue', linestyle='solid', mfc='blue', label='Average Wavelet Power')
        sub3.axhline(y=period_localW_p, xmin=0, xmax=1, color='k', linestyle='dashed', label='$\mathcal{P} =$' + "{:10.3f}".format(period_localW_p))
        sub3.set_ylim((1, 4))
        sub3.set_xlim((0, 5))
        sub3.set_title('Avg. Wavelet Power', fontsize=30)
        sub3.set_xlabel('Power ($\sigma^2$)', fontsize=25)
        sub3.set_ylabel('Period', fontsize=25)
        legend = sub3.legend(fontsize=13, loc='lower right', bbox_to_anchor=(1.0, 0.0))
        
        
        # Time series
        sub4.plot(time, norm_flux_n, marker='D', color='black', linestyle='dashed', mfc='blue', label='Time Series')
        sub4.set_title('Time Series ' + ext , fontsize=30)
        sub4.set_ylabel('Flux ', fontsize=25)
        sub4.set_xlabel('Time', fontsize=25)


        # Wavelet power spectrum
        CS = sub5.pcolormesh(time, Wperiod_n, Wpower_n, cmap='Blues', vmin=0, vmax=2)
        sub5.plot(time, Wcoi_n, 'r')  # cone of influence (The area's the signal is most reliable)
        sub5.set_xlabel('Time', fontsize=25)
        sub5.set_ylabel('Period', fontsize=25)
        sub5.set_title('Wavelet Power Spectrum ' + ext, fontsize=30)
        sub5.fill_between(time, Wcoi_n * 0 + Wperiod_n[-1], Wcoi_n, facecolor="none",
                  edgecolor="#00000040", hatch='x')
        sub5.contour(time, Wperiod_n, local_powerlaw_signif_n,levels = [0.25, 0.5, 0.75, 1, 1.25, 1.50], vmin=0, vmax=2, cmap = 'Reds', linewidths= 2, zorder=3)
        sub5.set_ylim(np.min(Wperiod_n), np.max(Wcoi_n))


        cbar = plt.colorbar(CS, location='bottom', ax=sub5)  # Pass ax=sub2
        cbar.set_label('Power ($\sigma^2$)', labelpad=10, fontsize=25)

        # Add the interpolated significance levels to the second subplot
        sub6.plot(tave_local_powerlaw_signif_n, Wperiod_n, 'k--', label='Significance Level')

        # Average wavelet power spectrum
        sub6.plot(Wpower_tave_n, Wperiod_n, marker='D', color='blue', linestyle='solid', mfc='blue', label='Average Wavelet Power')
        sub6.axhline(y= period_localW_n, xmin=0, xmax=1, color='k', linestyle='dashed',
                     label='$\mathcal{P} =$' + "{:10.3f}".format(period_localW_n))
        sub6.set_ylim((np.min(Wperiod_n), np.max(Wcoi_n)))
        sub6.set_xlim((np.min(Wpower_tave_n),np.max(Wpower_tave_n)))
        sub6.set_title('Avg. Wavelet Power ' + ext, fontsize=30)
        sub6.set_xlabel('Power ($\sigma^2$)', fontsize=25)
        sub6.set_ylabel('Period', fontsize=25)
        legend = sub6.legend(fontsize=13, loc='lower right', bbox_to_anchor=(1.0, 0.0))

        plt.show()
    result = {
        'period_localW_flag_p': period_localW_flag_p,
        'Wperiod_p': Wperiod_p,
        'Wpower_p': Wpower_p,
        'Wcoi_p': Wcoi_p,
        'local_powerlaw_signif_p': local_powerlaw_signif_p,
        'Wpower_tave_p': Wpower_tave_p,
        'period_localW_p': period_localW_p,
        'period_localW_flag_n': period_localW_flag_n,
        'Wperiod_n': Wperiod_n,
        'Wpower_n': Wpower_n,
        'Wcoi_n': Wcoi_n,
        'local_powerlaw_signif_n': local_powerlaw_signif_n,
        'Wpower_tave_n': Wpower_tave_n,
        'period_localW_n': period_localW_n
     }
    return result


def check_2(file): 
# Inputs  - file: path to the file that needs to be verified 
# Outputs - N/A
# Functionality - verifies if directory path exists and if not creates a new directory with the same path name 
    
# verify that path exists
    if os.path.isfile(file) == False: 
# if path does not exist we are running the code for the first time
        print(f'The file: {file} does not exist.')
        file_exist = False
    else: 
# File exists, no further actions needed
        file_exist = True
        
        print(f'The file: {file} exist, we will append results at the end.')
        
        return file_exist



# define the main analysis function so that it can be called externally by another script


def get_names(flare_directory):
    names = os.listdir(flare_directory)

    short_names = np.zeros(len(names), dtype = 'U30')
    for i in range(len(names)):
        short_names[i] = names[i][7:-26]
        exts = np.sort(np.array([*set(short_names)]))[1:]
    return exts

def qpps_main_analysis(i0,ie,csv_filename = False, csv_filename2 = False, rdb_directory = None): 
    
    #qpps_main_analysis(i0,ie,csv_filename = None, csv_filename2 = None, rdb_directory = None, save_path =False, load_path = False)

    '''Check if csv already existists'''
    file_exists = check_2(csv_filename)
    file_exists2 = check_2(csv_filename2)
    files = get_names(rdb_directory)
    # First thing we need to do is load all of the names in the RDB database
    # The func_get_names needs to provide with individual names of all flares in a sorted array. 
    # Save the files variable to a folder if save_path is specified
    # if save_path:
    #     # First thing we need to do is load all of the names in the RDB database
    #     # The func_get_names needs to provide with individual names of all flares in a sorted array. 
    #     files = get_names(rdb_directory)
        
    #     # Convert the list to a DataFrame
    #     df = pd.DataFrame(files, columns=['file_names'])
    #     # Ensure the directory exists
    #     os.makedirs(save_path, exist_ok=True)
    #     # Define the file path
    #     file_path = os.path.join(save_path, 'files.csv')
    #     # Save the DataFrame as a CSV file
    #     df.to_csv(file_path, index=False)
    #     print(f'Files variable saved to {file_path}')
    # else:
    #     # Load the files variable from the specified load path
    #     if load_path and os.path.exists(load_path):
    #         df = pd.read_csv(load_path)
    #         files = df['files'].tolist()
    #         print(f'Files variable loaded from {load_path}')
            
    #     else: 
    #         ValueError('If save_path is False then load_path must be used')
    # #print(files[:21], files[-21:])
    
    
    if i0 != 0 or ie != len(rdb_directory):
        files = files[i0:ie]

        
    return files, file_exists, file_exists2


#plotdir = '/Users/veronicaestrada/Downloads/Kazachenko/plots/'
#datadir ="/Users/veronicaestrada/Downloads/Kazachenko/"
#rdbdir  = datadir+'files'

### Event Analysis function ###

def multi_wavelet_analysis(i0, ie, plots_path = None, datadir = None, flare_directory = None, csv_filename = None, csv_filename2 = None):
    files, file_exists, file_exists2 = qpps_main_analysis( i0, ie, csv_filename = csv_filename, csv_filename2 = csv_filename2, rdb_directory = flare_directory)
    
    csv_file = []
    csv_file2 = []
    
    for i in range(len(files)):
        #print(files[i])
        
        ext = files[i]
        
        #print(f'the file is {ext}')
    
        sathr    = [5000.000,4500.000]
        bstr     = ['{:.2f}'.format(sathr[0]) + r"_brad",'{:.2f}'.format(sathr[1]) + r"_brad"]
        wvstr    = r""
        #print(f'bstr is {bstr}')
        flarename   = check(flare_directory + "/recfile"+ext+r"_cut08_sat",bstr,wvstr) 
        
        if  flarename != None:
            
            #print(f'the flare name is {flarename}')
        
        
            flare_data  = idlsave(flarename,verbose=0) 
            
            if flx_check(flare_data) == True:
                
            
                rdbflx,rdbrflx,rdbt,rdbtt,area8,n_o_satpix,ar_area = extract_RDB(flare_data)
            
                #### Flux Vs. Time ###
                flx_p = rdbrflx[0] #Posative Flux 
                flx_n = rdbrflx[1] #Negative Flux
                #print(len(flx_p))
                #### time ###
                tmin = rdbt # time series in minutes
                #print(len(tmin))
            
                results =  WaveletAnalysis(tmin, flx_p, flx_n, ext, save_path = plots_path +'main_plots/' + ext + '_wavelet_ana.tiff' )
            
                ####Flux Vs. Time 
                flx8_0 = flare_data.flx8[0] #Posative Flux 
                flx8_2 = flare_data.flx8[2] #Negative Flux
            
                plt.plot(flx8_0/1e21, label = 'Negative RF',color ='blue' )
                plt.plot(flx8_2/1e21, label= 'Posative RF', color = 'red')
                plt.legend(loc = 'upper right')
                plt.title('Flux Vs. Time',fontsize = 20)
                plt.ylabel('Flux',fontsize = 15)
                plt.xlabel('Time (min)',fontsize = 15)
                plt.legend(fontsize = 15)
                plt.savefig(plots_path + 'flux_vs_time/' + ext + '_timeseries.tiff')
                plt.close('all')
                #before exiting plt.close('all')
            
                ###finds the peaks for the reconnection rate so we can save the Reconnection Rate Vs Time[min]
                posative_peaks = sig.find_peaks(flx_p)
                negative_peaks = sig.find_peaks(np.abs(flx_n))
            
                peaks_prate = flx_p[posative_peaks[0]]
                peaks_ptim = rdbt[posative_peaks[0]]
            
                peaks_nrate = flx_n[negative_peaks[0]]
                peaks_ntim = rdbt[negative_peaks[0]]
            
                ###Plot for Reconnection Rate( Flux Vs. Time (min)
            
                plt.plot(rdbt,flx_p/1e19,color = 'r', alpha = 0.5, linewidth = 2,zorder = 2)
                plt.plot(peaks_ptim,peaks_prate/1e19, 'ro',zorder = 1)
                plt.plot(rdbt,flx_n/1e19,color = 'b', alpha = 0.5, linewidth = 2,zorder = 2)
                plt.plot(peaks_ntim,peaks_nrate/1e19, 'bo',zorder = 1)
                plt.title('Reconnection Rate Vs. Time Negative',fontsize = 20)
                plt.ylabel('Reconnection Rate ($\\frac{Mx}{s}$)',fontsize = 15)
                plt.xlabel('Time (min)', fontsize =15)
                plt.savefig(plots_path +'timeseries/' + ext + '_timeseries.tiff')
                plt.close('all')
            
            
                #plotting oscilation period
            
                plt.hist(peaks_ptim[1:-1] - peaks_ptim[0:-2],color = 'red',histtype = 'stepfilled',linewidth = 3,alpha = 0.5, label = 'postive')
                plt.hist(peaks_ntim[1:-1] - peaks_ntim[0:-2],color = 'blue',histtype = 'step',linewidth = 3, linestyle = 'dashed',label = 'negative')
                plt.ylabel('Counts [#]',fontsize = 15)
                plt.xlabel('Period [min]',fontsize = 15)
                plt.title('Counting Peaks', fontsize = 20)
                plt.savefig(plots_path +'counting_peaks/' + ext + '_counting_peaks.tiff')
                plt.legend(fontsize = 15)
                plt.close('all')
            
                total_nflux = np.min(flx8_0)
                total_pflux = np.max(flx8_2)
            
                total_parea = np.max(area8[0]) #double check 
                total_narea = np.max(area8[2])
            
                total_ar_parea = ar_area[0] #this might be wrong check with marcel
                total_ar_narea = ar_area[1]
            
                total_n_o_satpix = np.sum(n_o_satpix)
            
                qpp_ext = []
                no_qpp_ext = []
            
            
                if results['period_localW_flag_p'] == 1:
                    qpp_ext.append(ext)
                    #current_row = [qpp_ext, results, total_pflux, total_nflux, total_parea, total_narea, total_ar_parea, total_ar_narea, total_n_o_satpix]
                    current_row = [qpp_ext, results['period_localW_flag_p'], results['Wperiod_p'], results['Wpower_p'], results['Wcoi_p'], results['local_powerlaw_signif_p'], results['Wpower_tave_p'], results['period_localW_p'], results['period_localW_flag_n'], results['Wperiod_n'], results['Wpower_n'], results['Wcoi_n'], results['local_powerlaw_signif_n'], results['Wpower_tave_n'], results['period_localW_n'], total_pflux, total_nflux, total_parea, total_narea, total_ar_parea, total_ar_narea, total_n_o_satpix]
                    csv_file.append(current_row)
                    #print(f'this is a CSV file input {csv_file}')
            
            
                else: 
                    no_qpp_ext.append(ext)
                    non_qpp_current_row = [no_qpp_ext, results['period_localW_flag_p'], results['Wperiod_p'], results['Wpower_p'], results['Wcoi_p'], results['local_powerlaw_signif_p'], results['Wpower_tave_p'], results['period_localW_p'], results['period_localW_flag_n'], results['Wperiod_n'], results['Wpower_n'], results['Wcoi_n'], results['local_powerlaw_signif_n'], results['Wpower_tave_n'], results['period_localW_n'], total_pflux, total_nflux, total_parea, total_narea, total_ar_parea, total_ar_narea, total_n_o_satpix]
                    csv_file2.append(non_qpp_current_row)
                    
            else:
                print(f'File {ext} is Incomplete Check with Marcel')
 
        else:
            print(ext, i)
    
    Wqpp_Tabel = pd.DataFrame(csv_file, columns = ['Filename','FlagP','WperiodP','WpowerP','WcoiP', 'Local_SignifP', 'Wpower_TaveP', 'Period_LocalWP','FlagN', 'WperiodN', 'WpowerN','WcoiN','Local_SignifN','Wpower_TaveN','Period_LocalWN','Total_FluxP','Total_FluxN','Total_AreaP','Total_AreaN','Total_AR_AreaP','Total_AR_AreaN','Total_SatPix'])


    if file_exists:
        # df.to_csv(csv_filename, mode='a',header=True,index = False)
        Wqpp_Tabel.to_csv(csv_filename, mode='a', index = False) #indexbool, defaul True : Write row names (index).        

    else:    
        #df.to_csv(csv_filename, index = False)
        Wqpp_Tabel.to_csv(csv_filename, index = False) #indexbool, defaul True : Write row names (index).        


    #Wqpp_Tabel = pd.DataFrame(Wset, columns = ['Filename','FlagP','WperiodP','WpowerP','WcoiP', 'Local_SignifP', 'Wpower_TaveP', 'Period_LocalWP','FlagN', 'WperiodN', 'WpowerN','WcoiN','Local_SignifN','Wpower_TaveN','Period_LocalWN','Total_FluxP','Total_FluxN','Total_AreaP','Total_AreaN','Total_AR_AreaP','Total_AR_AreaN','Total_SatPix'])
    #Wqpp_Tabel.to_csv("Wqpp_Tabel.csv",index = True) #indexbool, defaul True : Write row names (index).        

    #Wqpp_Tabel
    
 
        
    try:
        csv_file2
    except NameError:
        print('All events detected a QPP')
    else:
        No_Wqpp_Tabel = pd.DataFrame(csv_file2, columns = ['Filename','FlagP','WperiodP','WpowerP','WcoiP', 'Local_SignifP', 'Wpower_TaveP', 'Period_LocalWP','FlagN', 'WperiodN', 'WpowerN','WcoiN','Local_SignifN','Wpower_TaveN','Period_LocalWN','Total_FluxP','Total_FluxN','Total_AreaP','Total_AreaN','Total_AR_AreaP','Total_AR_AreaN','Total_SatPix'])
        if file_exists2:
            No_Wqpp_Tabel.to_csv(csv_filename2, mode='a', index = False) #indexbool, defaul True : Write row names (index). 
            
        else:
            No_Wqpp_Tabel.to_csv(csv_filename2, index = False) #indexbool, defaul True : Write row names (index). 

### Running code ###

i0 = int(sys.argv[1]) # it is the first variable in your script call 
                  # example.sh -> python restartable_code_veronica.py 0 100 path/qpps.csv
ie = int(sys.argv[2]) # it is the second variable in your script call example.sh
#csv_filename = sys.argv[3] # it is the third variable in your script call example.sh
#csv_filename2 = sys.argv[4]

    
multi_wavelet_analysis(i0, ie, plots_path = '/Users/veronicaestrada/Downloads/Kazachenko_Lab/Project2/Test_plots/', datadir = "/Users/veronicaestrada/Downloads/Kazachenko_Lab/RDB_copy/", flare_directory = "/Users/veronicaestrada/Downloads/Kazachenko_Lab/RDB_copy/", csv_filename = "Wqpp_Tabel.csv", csv_filename2 = "No_Wqpp_Tabel.csv")



            # rdbflx,rdbrflx,rdbt,rdbtt,area8,n_o_satpix,ar_area = extract_RDB(flare_data)
            
            # #### Flux Vs. Time ###
            # flx_p = rdbrflx[0] #Posative Flux 
            # flx_n = rdbrflx[1] #Negative Flux
            # #print(len(flx_p))
            # #### time ###
            # tmin = rdbt # time series in minutes
            # #print(len(tmin))
            
            # results =  WaveletAnalysis(tmin, flx_p, flx_n, ext, save_path = plots_path +'main_plots/' + ext + '_wavelet_ana.tiff' )
            
            # ####Flux Vs. Time 
            # flx8_0 = flare_data.flx8[0] #Posative Flux 
            # flx8_2 = flare_data.flx8[2] #Negative Flux
    
            # plt.plot(flx8_0/1e21, label = 'Negative RF',color ='blue' )
            # plt.plot(flx8_2/1e21, label= 'Posative RF', color = 'red')
            # plt.legend(loc = 'upper right')
            # plt.title('Flux Vs. Time',fontsize = 20)
            # plt.ylabel('Flux',fontsize = 15)
            # plt.xlabel('Time (min)',fontsize = 15)
            # plt.legend(fontsize = 15)
            # plt.savefig(plots_path + 'flux_vs_time/' + ext + '_timeseries.tiff')
            # plt.close('all')
            # #before exiting plt.close('all')
            
            # ###finds the peaks for the reconnection rate so we can save the Reconnection Rate Vs Time[min]
            # posative_peaks = sig.find_peaks(flx_p)
            # negative_peaks = sig.find_peaks(np.abs(flx_n))
    
            # peaks_prate = flx_p[posative_peaks[0]]
            # peaks_ptim = rdbt[posative_peaks[0]]
    
            # peaks_nrate = flx_n[negative_peaks[0]]
            # peaks_ntim = rdbt[negative_peaks[0]]
    
            # ###Plot for Reconnection Rate( Flux Vs. Time (min)
    
            # plt.plot(rdbt,flx_p/1e19,color = 'r', alpha = 0.5, linewidth = 2,zorder = 2)
            # plt.plot(peaks_ptim,peaks_prate/1e19, 'ro',zorder = 1)
            # plt.plot(rdbt,flx_n/1e19,color = 'b', alpha = 0.5, linewidth = 2,zorder = 2)
            # plt.plot(peaks_ntim,peaks_nrate/1e19, 'bo',zorder = 1)
            # plt.title('Reconnection Rate Vs. Time Negative',fontsize = 20)
            # plt.ylabel('Reconnection Rate ($\\frac{Mx}{s}$)',fontsize = 15)
            # plt.xlabel('Time (min)', fontsize =15)
            # plt.savefig(plots_path +'timeseries/' + ext + '_timeseries.tiff')
            # plt.close('all')
        
        
            # #plotting oscilation period
        
            # plt.hist(peaks_ptim[1:-1] - peaks_ptim[0:-2],color = 'red',histtype = 'stepfilled',linewidth = 3,alpha = 0.5, label = 'postive')
            # plt.hist(peaks_ntim[1:-1] - peaks_ntim[0:-2],color = 'blue',histtype = 'step',linewidth = 3, linestyle = 'dashed',label = 'negative')
            # plt.ylabel('Counts [#]',fontsize = 15)
            # plt.xlabel('Period [min]',fontsize = 15)
            # plt.title('Counting Peaks', fontsize = 20)
            # plt.savefig(plots_path +'counting_peaks/' + ext + '_counting_peaks.tiff')
            # plt.legend(fontsize = 15)
            # plt.close('all')
            
            # total_nflux = np.min(flx8_0)
            # total_pflux = np.max(flx8_2)
            
            # total_parea = np.max(area8[0]) #double check 
            # total_narea = np.max(area8[2])
            
            # total_ar_parea = ar_area[0] #this might be wrong check with marcel
            # total_ar_narea = ar_area[1]
            
            # total_n_o_satpix = np.sum(n_o_satpix)
            
            # qpp_ext = []
            # no_qpp_ext = []
            
            
            # if results['period_localW_flag_p'] == 1:
            #     results
            #     qpp_ext.append(ext)
            #     #current_row = [qpp_ext, results, total_pflux, total_nflux, total_parea, total_narea, total_ar_parea, total_ar_narea, total_n_o_satpix]
            #     current_row = [qpp_ext, results['period_localW_flag_p'], results['Wperiod_p'], results['Wpower_p'], results['Wcoi_p'], results['local_powerlaw_signif_p'], results['Wpower_tave_p'], results['period_localW_p'], results['period_localW_flag_n'], results['Wperiod_n'], results['Wpower_n'], results['Wcoi_n'], results['local_powerlaw_signif_n'], results['Wpower_tave_n'], results['period_localW_n'], total_pflux, total_nflux, total_parea, total_narea, total_ar_parea, total_ar_narea, total_n_o_satpix]
            #     Wset.append(current_row)
                
    
            # else: 
            #     no_qpp_ext.append(ext)
            #     non_qpp_current_row = [no_qpp_ext, results['period_localW_flag_p'], results['Wperiod_p'], results['Wpower_p'], results['Wcoi_p'], results['local_powerlaw_signif_p'], results['Wpower_tave_p'], results['period_localW_p'], results['period_localW_flag_n'], results['Wperiod_n'], results['Wpower_n'], results['Wcoi_n'], results['local_powerlaw_signif_n'], results['Wpower_tave_n'], results['period_localW_n'], total_pflux, total_nflux, total_parea, total_narea, total_ar_parea, total_ar_narea, total_n_o_satpix]
            #     Wset_no_qpp.append(non_qpp_current_row)