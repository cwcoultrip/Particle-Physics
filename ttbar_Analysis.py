import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,AutoMinorLocator
from scipy.stats import chisquare
import math
import os
import time

### Physics 121W Report 2 Code (ttbar Analysis)
### Cody Coultrip

### Parameters ###

luminosity = 10064

nbins = 40 # number of bins
xmin = 100. # Lower energy cutoff (fixed)
xmax = 250. # Upper energy cutoff (fixed)

bin_edges = np.linspace(xmin, xmax, num=nbins+1) # Array with bin edges
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2. # Array with bin centers


### Data Processing ###

def calc_mjjj(pts, Es, etas, phis):
    # Find 3-jet combination with largest vector pt and calculate the invariant mass mjjj. Given arrays for jet
    # pts, energies, etas, and phis; returns the 3-jet invariant mass for the three jets with highest vector pt.
    
    # initialize px, py, pz, and E
    px = 0
    py = 0
    pz = 0
    E = 0
    
    # find the 3-jet combination with the largest vector pt
    temp_ptsum = 0
    max_ptsum = 0
    indices = [0, 0, 0]
    
    # cycle through combinations of 3-jet vector pt and returns the greatest value
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            for k in range (j+1, len(pts)):
                for n in [i, j, k]:
                    px += pts[n]*math.cos(phis[n])
                    py += pts[n]*math.sin(phis[n])
                temp_ptsum = (px*px + py*py)
                if temp_ptsum > max_ptsum:
                    max_ptsum = temp_ptsum
                    indices = [i, j, k]
    max_ptsum = math.sqrt(max_ptsum)
    
    # re-initialize px, py, pz, and E
    px = 0
    py = 0
    pz = 0
    E = 0    
    
    # Calculate the invariant mass of the 3-jet combination mjjj
    for i in indices:
        px += pts[i]*math.cos(phis[i])
        py += pts[i]*math.sin(phis[i])
        pz += pts[i]*math.sinh(etas[i])
        E += Es[i]
    
    e2 = E*E
    p2 = px*px + py*py + pz*pz
    
    if (e2 > p2):
        mjjj = math.sqrt(e2-p2)
    else:
        mjjj = -math.sqrt(p2-e2)

    return mjjj


def count(files):
    # Counts number of events that pass the filter in the specified path; returns a 2D array with each row corresponding
    # to a different cut of MV2c10 and each column corresponding to a different bin. For 40 bins and 12 cuts of MV2c10,
    # there are 40 columns and 12 rows.
    
    fields = ['mcWeight', 'SumWeights', 'XSection', 'trigE', 'trigM', 'lep_pt', 'lep_ptcone30',
              'lep_etcone20', 'lep_eta', 'lep_phi', 'met_et', 'met_phi', 'jet_n', 'jet_E',
              'jet_pt', 'jet_eta', 'jet_phi', 'jet_MV2c10', 'scaleFactor_PILEUP', 'scaleFactor_ELE',
              'jet_jvt', 'scaleFactor_MUON', 'scaleFactor_BTAG', 'scaleFactor_LepTRIGGER',
              'lep_isTightID', 'lep_type', 'lep_trackd0pvunbiased', 'lep_tracksigd0pvunbiased',
              'lep_z0']
    
    start_time = time.time()
    
    total_events = 0 # Total events processed
    n_unfiltered = 0 # Number of photons that pass the filter
    
    mv2c10_cuts = np.linspace(0.4244273, 0.9744273, 12) # Evenly spaced array of cuts for MV2c10
    
    counts = np.zeros((12, nbins)) # Initialize array for number of counts per bin for different mv2c10
    
    if str(files) == 'E:\\ATLAS\\1lep\\Data\\*':
        MC = False
    else:
        MC = True
    
    for array in uproot.iterate(files, fields, step_size = 10000, library = 'np'):
    
        nread = len(array['jet_n'])
        total_events += nread #print('total_events: ', total_events)
    
        for i in range(nread):
            lep_pt = array['lep_pt'][i]
            lep_eta = array['lep_eta'][i]
            lep_phi = array['lep_phi'][i]
            trigE = array['trigE'][i]
            trigM = array['trigM'][i]
            lep_etcone20 = array['lep_etcone20'][i]
            lep_ptcone30 = array['lep_ptcone30'][i]
            lep_isTightID = array['lep_isTightID'][i]
            lep_type = array['lep_type'][i]
            lep_trackd0pvunbiased = array['lep_trackd0pvunbiased'][i]
            lep_tracksigd0pvunbiased = array['lep_tracksigd0pvunbiased'][i]
            lep_z0 = array['lep_z0'][i]
            mc_weight = array['mcWeight'][i]
            x_section = array['XSection'][i]
            sum_weights = array['SumWeights'][i]
            met_et = array['met_et'][i]
            met_phi = array['met_phi'][i]
            jet_n = array['jet_n'][i]
            jet_E = array['jet_E'][i]
            jet_pt = array['jet_pt'][i]
            jet_eta = array['jet_eta'][i]
            jet_phi = array['jet_phi'][i]
            jet_MV2c10 = array['jet_MV2c10'][i]
            # jet_jvt = array['jet_jvt'][i]
            
            SF_pileup = array['scaleFactor_PILEUP'][i]
            SF_ele = array['scaleFactor_ELE'][i]
            SF_muon = array['scaleFactor_MUON'][i]
            SF_btag = array['scaleFactor_BTAG'][i]
            SF_leptrig = array['scaleFactor_LepTRIGGER'][i]
            
            if MC == True:
                scalefactor = SF_pileup*SF_ele*SF_muon*SF_btag*SF_leptrig
                event_weight = scalefactor*mc_weight # Event reweighting
                if event_weight == 0:
                    continue
                rescale = luminosity * (x_section/sum_weights) # Luminosity renormalization
                weight = event_weight*rescale
                if weight == 0:
                    continue
            elif MC == False:
                weight = 1
            
            # Track isolation must be less than 0.15
            if lep_ptcone30/lep_pt > 0.15:
                continue
            
            # Calorimeter isolation must be less than 0.15
            if lep_etcone20/lep_pt > 0.15:
                continue
            
            # Lepton must satisfy single-electron or single-muon trigger
            if trigE == False and trigM == False:
                continue
            
            # Lepton must satisfy "tight" identification criteria
            if lep_isTightID == False:
                continue
            
            # Lepton transverse momentum greater than 30 GeV (30000 MeV)
            if lep_pt < 30000:
                continue
            
            # Missing transverse energy greater than 30 GeV (30000 MeV)
            if met_et < 30000:
                continue
            
            # Pseudorapidity must be less than 2.5
            if abs(lep_eta) > 2.5:
                continue
            
            lep_theta = 2*math.atan(math.exp(-lep_eta)) # Calculate theta for lepton
            
            # Electron selection criteria
            if lep_type == 11:
                if lep_trackd0pvunbiased/lep_tracksigd0pvunbiased >= 5:
                    continue
                if lep_eta > 2.47:
                    continue
                if lep_z0*math.sin(lep_theta) >= 0.5:
                    continue
            
            # Muon selection criteria
            if lep_type == 13:
                if lep_trackd0pvunbiased/lep_tracksigd0pvunbiased >= 3:
                    continue
                if lep_eta > 2.5:
                    continue
                if lep_z0*math.sin(lep_theta) >= 0.5:
                    continue
                
            mtw = math.sqrt(2*lep_pt*met_et*(1-math.cos(lep_phi-met_phi)))/1000 # Calculate W boson transverse mass
    
            # W boson transverse mass greater than 30 GeV
            if mtw < 30.0:
                continue
            
            # Number of total jets must be equal to or greater than 4
            if jet_n < 4:
                continue

            # Filter "good" jets by jet_pt           
            n_pt_jets = (jet_pt > 30000) # Number of good jets
            
            good_jet_pt = jet_pt[(jet_pt > 30000)] # Transverse momentum for good jets
            good_jet_MV2c10 = jet_MV2c10[(jet_pt > 30000)] # MV2c10 for good jets
            good_jet_E = jet_E[(jet_pt > 30000)] # Energy for good jets
            good_jet_eta = jet_eta[(jet_pt > 30000)] # Eta for good jets
            good_jet_phi = jet_phi[(jet_pt > 30000)] # Phi for good jets
            
            # Number of good jets must be equal to or greater than 4
            if n_pt_jets.sum() < 4:
                continue

            mjjj = calc_mjjj(good_jet_pt, good_jet_E, good_jet_eta, good_jet_phi)/1000
            for i in range(len(mv2c10_cuts)):
                n_btagged_jets = good_jet_MV2c10 > mv2c10_cuts[i]
                if n_btagged_jets.sum() < 2:
                    continue
                if mjjj < xmin or mjjj >= xmax:
                    continue                
                
                count_index = np.digitize(mjjj, bin_edges)-1
                counts[i][count_index] += weight
            
            n_unfiltered += 1
            
        end_time = time.time()
        print('Total events: ', total_events, 'Unfiltered events: ', n_unfiltered, ' Elapsed time: ', end_time - start_time)
    
    return counts


### Data for plots ###

path = 'E:\\ATLAS\\1lep\\'

ttbar = path + 'MC-S\\ttbar\\mc_410000.ttbar_lep.1lep.root'
diboson = path + 'MC-B\\Diboson\\*'
single_top = path + 'MC-B\\Single_top\\*'
Vp_jets = path + 'MC-B\\V+jets\\*'
data = path + 'Data\\*'

ttbar_counts = count(ttbar)
diboson_counts = count(diboson)
single_top_counts = count(single_top)
Vp_jets_counts = count(Vp_jets)
data_counts = count(data)


### Plotting ###

signal_counts = ttbar_counts
background_counts = diboson_counts + single_top_counts + Vp_jets_counts
MC_counts = signal_counts + background_counts
total_counts = signal_counts + background_counts
mv2c10_cuts = np.linspace(0.4244273, 0.9744273, 12)

total_sig = np.zeros(len(signal_counts))
total_bkg = np.zeros(len(background_counts))
total_MC = np.zeros(len(signal_counts))
total_data = np.zeros(len(data_counts))

for i in range(len(background_counts)):
    total_sig[i] = signal_counts[i].sum() # Total number of signal counts for each value of MV2c10 (sum across all bins)
    total_bkg[i] = background_counts[i].sum() # Total number of background counts for each value of MV2c10
    total_data[i] = data_counts[i].sum() # Total number of events in data for each value of MV2c10

total_MC = total_sig + total_bkg # Total MC counts for each value of MV2c10, sum of signal ttbar and background
sig_bkg_ratio = np.divide(total_sig,total_bkg) # Ratio of signal to background for each value of MV2c10


##### Plot 1 - m_jjj for each cut of MV2c10 #####

data_counts_err = np.sqrt(data_counts)

for i in range(len(signal_counts)):
    plt.axes([0.1,0.2,0.65,0.65])
    axes = plt.gca()
        
    plt.errorbar(bin_centers, data_counts[i], yerr=data_counts_err[i], fmt='ko', label='data')
    axes.hist(bin_centers, bin_edges, weights=ttbar_counts[i], label='ttbar', color='orange')
    axes.hist(bin_centers, bin_edges, weights=diboson_counts[i], label='diboson', color='steelblue')
    axes.hist(bin_centers, bin_edges, weights=single_top_counts[i], label='single top', color='deepskyblue')
    axes.hist(bin_centers, bin_edges, weights=Vp_jets_counts[i], label='V+ jets', color='purple')
    
    # Limits
    axes.set_xlim([xmin,xmax])
    axes.set_ylim(bottom=0,top=(np.amax(ttbar_counts[i])+math.sqrt(np.amax(ttbar_counts)))*1.5)
    
    # Labels
    axes.set_ylabel(r'Events / '+'bin',fontname='sans-serif',horizontalalignment='right',y=1.0,fontsize=11)
    axes.set_xlabel(r'$m_{jjj}$ (GeV)',fontname='sans-serif',horizontalalignment='right',y=1.0,x=0.95,fontsize=1)
    
    # Tick marks
    axes.yaxis.set_minor_locator(AutoMinorLocator())
    axes.yaxis.get_major_ticks()[0].set_visible(False)
    axes.tick_params(which='both',direction='in',top=True,labeltop=False,labelbottom=True,right=True,labelright=False)
    axes.xaxis.set_minor_locator(AutoMinorLocator())
    
    # Text
    plt.text(0.05,0.95,'ATLAS Open Data',ha="left",va="top",transform=axes.transAxes,fontsize=13)
    plt.text(0.05,0.86,'$\sqrt{s}=13$ TeV',ha="left",va="top",transform=axes.transAxes)
    plt.text(0.05,0.77,'MV2c10 = ' + '{0:.6f}'.format(mv2c10_cuts[i]),ha="left",va="top",transform=axes.transAxes)
    
    # Legends
    axes.legend()
    
    plt.show()
 

##### Plot 2 - Weighted background counts vs weighted signal counts #####

axes2 = plt.gca()
plt.plot(total_bkg, total_sig, 'ko')

# Labels
axes2.set_ylabel(r'Weighted Signal Events',fontname='sans-serif',horizontalalignment='right',y=1.0,fontsize=11)
axes2.set_xlabel(r'Weighted Background Events',fontname='sans-serif',horizontalalignment='right',y=1.0,x=0.95,fontsize=11)

# Tick marks
axes2.yaxis.set_minor_locator(AutoMinorLocator())
axes2.yaxis.get_major_ticks()[0].set_visible(False)
axes2.tick_params(which='both',direction='in',top=True,labeltop=False,labelbottom=True,right=True,labelright=False)
axes2.xaxis.set_minor_locator(AutoMinorLocator())

# Text
plt.text(0.05,0.95,'ATLAS Open Data',ha="left",va="top",transform=axes2.transAxes,fontsize=13)
plt.text(0.05,0.86,'$\sqrt{s}=13$ TeV',ha="left",va="top",transform=axes2.transAxes)

plt.show()


##### Plot 3 - Chi-squared between total signal+background events and data #####

chi_squared = np.zeros(len(total_data))

def chisqr(obs, exp, err):
    return np.sum((obs-exp)**2 / exp)

for i in range(len(data_counts)):
    chi_squared[i] = chisqr(data_counts[i], total_counts[i], data_counts_err[i])

axes3 = plt.gca()
plt.plot(mv2c10_cuts, chi_squared, 'ko')

# Labels
axes3.set_ylabel(r'Chi-squared',fontname='sans-serif',horizontalalignment='right',y=1.0,fontsize=11)
axes3.set_xlabel(r'MV2c10',fontname='sans-serif',horizontalalignment='right',y=1.0,x=0.95,fontsize=11)

# Tick marks
axes3.yaxis.set_minor_locator(AutoMinorLocator())
axes3.yaxis.get_major_ticks()[0].set_visible(False)
axes3.tick_params(which='both',direction='in',top=True,labeltop=False,labelbottom=True,right=True,labelright=False)
axes3.xaxis.set_minor_locator(AutoMinorLocator())

# Text
plt.text(0.63,0.95,'ATLAS Open Data',ha="left",va="top",transform=axes3.transAxes,fontsize=13)
plt.text(0.63,0.86,'$\sqrt{s}=13$ TeV',ha="left",va="top",transform=axes3.transAxes)

plt.show()


##### Plot 4 - Ratio of signal to background for each MV2c10 #####

axes3 = plt.gca()
plt.plot(mv2c10_cuts, sig_bkg_ratio, 'ko')

# Labels
axes3.set_ylabel(r'Signal/Background',fontname='sans-serif',horizontalalignment='right',y=1.0,fontsize=11)
axes3.set_xlabel(r'MV2c10',fontname='sans-serif',horizontalalignment='right',y=1.0,x=0.95,fontsize=11)

# Tick marks
axes3.yaxis.set_minor_locator(AutoMinorLocator())
axes3.yaxis.get_major_ticks()[0].set_visible(False)
axes3.tick_params(which='both',direction='in',top=True,labeltop=False,labelbottom=True,right=True,labelright=False)
axes3.xaxis.set_minor_locator(AutoMinorLocator())

# Text
plt.text(0.05,0.95,'ATLAS Open Data',ha="left",va="top",transform=axes3.transAxes,fontsize=13)
plt.text(0.05,0.86,'$\sqrt{s}=13$ TeV',ha="left",va="top",transform=axes3.transAxes)

plt.show()

########################
### Acknowledgements ###
########################

# Credit to the ATLAS Outreach data & tools github ttbar Analysis C++ code found at:
# https://github.com/atlas-outreach-data-tools/atlas-outreach-cpp-framework-13tev/tree/master/Analysis/TTbarAnalysis
# Using this framework, I was able to find an expression for W boson transverse momentum. I also used some of the filters
# used in the C++ code.

# Credit to the ROOT documentation available at:
# https://root.cern.ch/doc/master/index.html
# This helped me while I looked through the ROOT code to see how to calculate the 3-jet invariant mass. By following the
# ROOT implementation of lorentz vectors in C++, I was able to create my own function in Python to find the three highest
# vector pt jets and calculate the invariant mass.

# Additional thanks to the ATLAS Collaboration for the 13 TeV ATLAS Open Data release, whose data was used for this analysis.