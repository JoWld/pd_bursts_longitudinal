
# script runs beta burst detection from sensorimotor label using a percentile threshold approach
# the output will be a new numpy array of the time series in which all time points that do not "contain" beta bursts are set to zero
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:42:30 2023

@author: joswal
"""
import os
import numpy as np
import mne
import scipy as scipy
import os.path as op
import csv
import matplotlib.pyplot as plt
from mne.io import read_raw_fif
# %% Paths

#%% Settings
usr = 'joswal'
study = 'parkinson_motor_2024'#'parkinsons_longitudinal'
overwrite = True
# no loop  for hemisphere and session yet!
hemi = 'lh'
cond = 'ses1'

lim = 50  # lower duration limit for beta burst, in ms
percentile = 75 # setting individual amplitude threshold at 75% of the band-pass filtered beta amplitude envelope for each channel and condition

#burst_range = 15 #define range as peak frequency +/- burst_range, only needed for narrow-band bursts, otherwise: 
beta_lo = 13
beta_hi = 30
######  GENERAL ANALYSIS SETTINGS
# other analysis parameters
#n_fft = 1024
#fmin, fmax = 2, 45  # lower and upper band pass frequency for data filtering
#dwn = 200 # downsampling target frequency (Hz) for burst analysis

# %% Paths
# select study population 

if study == 'parkinson_motor_2024':
    archive_nr='21055'# '20055' # '20079'
else: 
    archive_nr='20079'# for parkinsons_longitudinal
    
tfr_path = '/archive/'+archive_nr + '_' + study +'/TFR'
tc_path = '/archive/'+archive_nr + '_' + study +'/TC'

# the following code up to line 72 read in a .csv file containing sub IDs and prepared it to be looped through afterwards
# I assume you might have some other input here, so change accordingly
subj_data_path = '/home/'+ usr+'/parkinson_motor/subj_data/' # where the list with sub IDs is stored
subj_file = op.join(subj_data_path, 'pdsubjects_and_dates.csv')
# load subject list
with open(subj_file, newline='') as csvfile:
    tmp = csv.reader(csvfile, delimiter=',', quotechar='"')
    subjid, date, include = [], [], []
    for ii, row in enumerate(tmp):
        subjid.append(row[1])
        date.append(row[2])
        include.append(row[3])
        
include = [x == '1' for x in include]       # Make boolean array
tt = [x == '' for x in date]                # Find empty cells to remove
idxer = [a and not b for a, b in zip(include, tt)]
    
subjects = [i for (i, v) in zip(subjid, idxer) if v]
date_flt = [i for (i, v) in zip(date, idxer) if v]
# unify subject list to 4-digit numbers
subjects2 =[]
for ii in subjects:
    ii = (f"{int(ii):04d}")
    subjects2.append(ii)
subjects_and_dates = [os.path.join('NatMEG_'+s, d) for (s, d) in zip(subjects2, date_flt)]

#load info on peak beta channels and frequencies, only needed if not running broad-band
#beta_info_f = []
#with open (subj_file2, 'r') as fin: 
#    for row in csv.reader(fin):
#        beta_info_f.extend([[row[0]]+[int(row[1])]])

#summary_all = []

for subj in subjects_and_dates:
    
    subid = subj[-11:-7]
    subj_path   = op.join(tc_path)
    #fig_path = op.join(subj_path, 'plots/')
    out_fname = subj_data_path+'/'+subid+'_bbthres-tc_sensorimotor_'+hemi+'_'+cond
    # loaed sensorimotor label time course 
    tc_fname = op.join(tc_path, subid+'_tc_sensorimotor_'+hemi+'_'+cond+'.npy')
    tfr_fname = op.join(tfr_path, subid+'_tfr_sensorimotor_'+hemi+'_'+cond+'.npy')
    data1 = np.load(tc_fname, allow_pickle = True)
    data1 = np.float64(np.squeeze(data1,axis=0))
    #data1 = np.squeeze(data1,axis=0)
    #data1, times = lab_tc[0], lab_tc[1]
    #   load information about peak beta frequencies, only needed for narrow-band bursts
    #for j in range(0, len(beta_info_f)):
    #    if beta_info_f[j][0] == subj:
    #        print('patient found')
    #        beta_lo = beta_info_f[j][1] - burst_range
    #        beta_hi = beta_info_f[j][1] + burst_range
    # Resampling
    ###  mne.filter.resample includes filters to avoid aliasing
    sfreq = 200
    #down = sfreq / dwn
    # goal: downsampling to frequency of 333 Hz, i.e. multiply by factor sfreq/1000
    #out1 = mne.filter.resample(data1, down=down, npad='auto', n_jobs=16, pad='reflect_limited',
    #                           verbose=None)  # additional options: window='boxcar', npad=100,
    # split data into consecutive epochs
    #sfreq = sfreq / down
    #ws = int(20 * sfreq / fmin)  # number of samples per window
    #overlap = 1 - 0  # set amount of overlap for consecutive FFT windows (second number sets amount of overlap)
    # separate data into consecutive data chunks (episode-like, because spectral_connectivity expects epochs)
    #array1 = list()
    #start = 0
    #stop = ws
    #step = int(ws * overlap)
    #while stop < out1.shape[1]:
    #    tmp = out1[:, start:stop]
    #    start += step
    #    stop += step
    #    array1.append(tmp)

    # define frequencies of interest
    freqs = np.arange(7., 45., 1.)
    #n_cycles = freqs / 2.

    #power = mne.time_frequency.tfr_array_morlet(array1, sfreq=sfreq, freqs=freqs,
    #                                            n_cycles=n_cycles, output='complex', n_jobs=16)
    power = np.load(tfr_fname, allow_pickle = True)
    freq_lo = beta_lo - int(min(freqs))
    freq_hi = beta_hi - int(min(freqs))
    cutoff = np.ceil(lim / 1000 * sfreq)  ### --> multiply with sfreq to get value in data points

    amplitude = []
    for k in range(0, len(power)):
        tmp = power[k][0][freq_lo:freq_hi + 1]
        tmptmp = np.mean(tmp, axis=0)
        amplitude = np.concatenate((amplitude, tmptmp), axis=None)
    rec_amp = np.absolute(amplitude)  # , /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])
    #fwhm = sfreq / (5 * down)
    fwhm = sfreq /5
    def fwhm2sigma(fwhm):
        return fwhm / np.sqrt(8 * np.log(2))
    sigma = fwhm2sigma(fwhm)
    filt1 = scipy.ndimage.filters.gaussian_filter1d(rec_amp, sigma, axis=-1, order=0, mode='reflect', cval=0.0,truncate=4.0)
    val = np.percentile(filt1, percentile)
    bin_burst = (filt1 > val).astype(np.int_)  # outputs binarized data, 1 for values above threshold
  
    ### copied from stackoverflow:
    ### https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
    def rle(inarray):
        ia = np.asarray(inarray)  # force numpy
        n = len(ia)
        if n == 0:
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)  # must include last element posi
            z = np.diff(np.append(-1, i))  # run lengths
            p = np.cumsum(np.append(0, z))[:-1]  # positions
            return (z, p, ia[i])  # return(z, p, ia[i])

   # burst_dur = []
   # burst_dur_ms = []
   # burst_amp = []
   # burst_onset = []
   # burst_offset = []

    burst_info = rle(bin_burst)
#    for l in range(0, len(burst_info[0])):
#        if burst_info[2][l] > 0:
#            if burst_info[0][l] >= cutoff:
#                tmp = burst_info[0][l]  # burst duration
#                tmp1 = burst_info[1][l]  # burst onset
#               tmp2 = tmp1 + tmp  # burst offset
#                tmp3 = np.max(filt1[tmp1:tmp1 + tmp])  # burst amplitude
#                burst_dur = np.concatenate((burst_dur, tmp), axis=None)
#                burst_onset = np.concatenate((burst_onset, tmp1), axis=None)
#                burst_amp = np.concatenate((burst_amp, tmp3), axis=None)
#                burst_offset = np.concatenate((burst_offset, tmp2), axis=None)
#    burst_dur_ms = (burst_dur / sfreq) * 1000

#    bbi = (np.diff(burst_onset)) / sfreq * 1000

#    ibi = []
#    for l in range(1, len(burst_offset)):
#        tmp = burst_onset[l] - burst_offset[l - 1]
#        ibi = np.concatenate((ibi, tmp), axis=None)
#    ibi = (ibi / sfreq) * 1000
    # binarized & temporally thresholded time series (bursts > lim)
    zeros = [0] * len(bin_burst)
    
# binarized timeseries, all bursts
    burst_binary_all=[]
    for k in range(0, len(burst_info[0])):
        tmp4=[]
        bdur=burst_info[0][k]
        bonset=burst_info[1][k]
        if burst_info[2][k]>0:
            if burst_info[0][k]>=cutoff:
                tmp2=zeros[bonset:bonset+bdur] 
                for x in tmp2:
                    tmp4.append(1)   # binarized & duration thresholded trace                            
            else:
                tmp4=zeros[bonset:bonset+bdur] 
        else:
            tmp4=zeros[bonset:bonset+bdur]   
        burst_binary_all=np.concatenate((burst_binary_all,tmp4), axis=None)

# apply this to the time serias data to get a time series were all time points with an amplitude < chosen percentile are set to 0 while those > percentile threshold have their true value
burst_tc = data1[:len(burst_binary_all)] * burst_binary_all
non_zero_indices = np.nonzero(burst_tc)[0]

np.save(out_fname, burst_tc)
 #   tmax = 300
 #   burst_count = len(burst_amp)
 #   burst_rate = burst_count / tmax
 #   burst_dur_mean = burst_dur_ms.mean()
 #   burst_dur_med = np.median(burst_dur_ms)
 #   burst_dur_sd = burst_dur_ms.std()
 #   burst_amp_mean = burst_amp.mean()
 #   ibi_mean = ibi.mean()
    # get bins fr burst duration
 #   dur_100 = len(list(filter(lambda burst_dur_ms: burst_dur_ms < 100, burst_dur_ms)))/tmax
 #   dur_200 = len(list(filter(lambda burst_dur_ms: 100 < burst_dur_ms < 200, burst_dur_ms)))/tmax
 #   dur_300 = len(list(filter(lambda burst_dur_ms: 199 < burst_dur_ms < 300, burst_dur_ms) ))     /tmax
 #   dur_500 = len(list(filter(lambda burst_dur_ms: 399 < burst_dur_ms < 500, burst_dur_ms)))/tmax
 #   dur_400 = len(list(filter(lambda burst_dur_ms: 299 < burst_dur_ms < 400, burst_dur_ms) ))  /tmax
 #   dur_600 = len(list(filter(lambda burst_dur_ms: 499 < burst_dur_ms < 600, burst_dur_ms)))/tmax
 #   dur_700 = len(list(filter(lambda burst_dur_ms: 599 < burst_dur_ms < 700, burst_dur_ms)))/tmax
 #   dur_800 = len(list(filter(lambda burst_dur_ms: burst_dur_ms >= 700, burst_dur_ms)))/tmax
    
 #   summary = [subj, cond, hemi, burst_rate, burst_dur_mean, burst_dur_med, burst_dur_sd, burst_amp_mean, ibi_mean,  dur_100, dur_200, dur_300, 
 #              dur_400, dur_500, dur_600, dur_700, dur_800]
    
 #   with open(subj_path + output_file, 'w', ) as myfile:
  #          wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
  #          for word in summary:
  #              wr.writerow([word])

#    summary_all.append(summary)
    
    #### Plot data 
    ### GENERAL PLOTTING SETTINGS
    
    # Font specifications
    # font = {
    #     'color':  'black',
    #     'weight': 'normal',
    #     'size': 20,
    #     }
    
    # t_from=0 # start time (in seconds)
    # t_to=10 # end time (in seconds)
    # start=int(t_from*sfreq)
    # stop=int(t_to*sfreq)
    # time=np.arange(t_from, t_to, 1/sfreq)
    # time=time[0:(stop-start)]
    
    # ylim=np.max(filt1[start:stop])+(np.max(filt1[start:stop]))/20
    # y1=burst_binary_all[start:stop]                
    
    # fig, axs = plt.subplots(3, 1, figsize=(12, 7))
    # fig.tight_layout()
    # plt.subplots_adjust(bottom=0.12, left=0.02, right=0.95, top=0.92, wspace=0.7)
    
    # # raw signal: out1
    # ax1=plt.subplot(311)
    # plt.plot(time, np.squeeze(out1)[start:stop], label='raw data')
    # plt.title('raw data', loc='left', fontdict=font)
    # frame1 = plt.gca()
    # frame1.axes.get_yaxis().set_ticks([])
    # frame1.axes.get_xaxis().set_ticklabels([])
    
    # # narrow band filtered signal: amplitude
    # ax2=plt.subplot(312)
    # plt.plot(time, amplitude[start:stop], label='band pass filtered data')
    # plt.title('band pass filtered data', loc='left', fontdict=font)
    # frame1 = plt.gca()
    # frame1.axes.get_yaxis().set_ticks([])
    # frame1.axes.get_xaxis().set_ticklabels([])
    
    # # smoothed amplitude envelope: filt 1
    # ax3=plt.subplot(313)
    # plt.plot(time, filt1[start:stop], label='amplitude envelope')
    # ax3.axhline(val, color='magenta', lw=2, alpha=0.5)
    # ax3.fill_between(time, 0, ylim, y1 > 0,
    #                 facecolor='magenta', alpha=0.3)
    # ax3.fill_between(time, val, filt1[start:stop], y1 > 0,
    #                 facecolor='magenta', alpha=0.4)
    # plt.title('amplitude envelope', loc='left', fontdict=font)
    # plt.xticks(fontsize=20)            
    # plt.xlabel('time (s)', fontdict=font)
    # frame1 = plt.gca()
    # frame1.axes.get_yaxis().set_ticks([])

    # ftitle=('raw_data_processing_%s_%s.png' % (beta_fre, hemi))   
    # fname  = fig_path + ftitle
    # plt.savefig(fname)
    # plt.show()


#with open(subj_data_path + output_file, 'w', ) as myfile:
#        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#        for word in summary_all:
#          wr.writerow([word])
