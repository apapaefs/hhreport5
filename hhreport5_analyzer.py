import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hhreport5_plotting import *

##############################################
# functions
##############################################
# read Hua-Sheng's results and return an entry for the dictionary
def read_HS(directory, result_type, order, energy, pdfset, MH):
    filetoread = directory + '/' + result_type + '/' + result_type + '_' + str(order) + '_' + str(energy) + 'TeV_' + str(pdfset) + '_MH' + str(MH) + 'GeV.dat'
    print('reading results from', filetoread)
    # define the columns
    # scale=(muR/mu0,muF/mu0):
    colnames = ["(1.,1.)", "(2.,1.)", "(0.5,1.)", "(1.,2.)", "(2.,2.)", "(1.,0.5)", "(0.5,0.5)"]
    # read the file
    df = pd.read_csv(filetoread, sep=' ', comment='#', names=colnames)
    return df

# read EW K-factors and return an entry for the dictionary
def read_EW(directory, result_type, energy, pdfset):
    filetoread = directory + '/' + result_type +  '_kfac-' + str(pdfset) + '-' + str(energy) + 'TeV_muf=mur=0.5Mhh.dat'
    print('reading EW results from', filetoread)
    # read the file
    df = pd.read_csv(filetoread, sep='\t')
    return df



###################################################
# read+plot Hua-Sheng's N3LO+NLL and NNLO results:
###################################################
HSresults = {} # dictionary to hold the cross section
HSresults_directory = 'N3LO' # directory for Hua-Sheng's results
HStypes = ['TotalXS', 'Mhh']
HSorders = ['NNLO', 'N3LON3LL', 'N3LO']
HSpdfsets = ['PDF4LHC21_40', 'MSHT20xNNPDF40_aN3LO_qed', 'MSHT20xNNPDF40_NNLO_qed']
HSMH = [125]
HSenergies = [13.6, 13, 14]
for result_type in HStypes:
    for order in HSorders:
        for energy in HSenergies:
            for pdfset in HSpdfsets:
                if (pdfset == 'MSHT20xNNPDF40_aN3LO_qed' or pdfset == 'MSHT20xNNPDF40_NNLO_qed') and energy != 14: # for now the approximate PDF is available at 14 TeV
                    continue 
                for MH in HSMH:
                    data = read_HS(HSresults_directory, result_type, order, energy, pdfset, MH)
                    HSresults[(result_type, order, energy, pdfset, MH)] = data

# generate the K-factors:
# N3LO+N3LL/NNLO
K3N3LL2 = {} # the N3LO+N3LL/NNLO K-factors
for result_type in HStypes:
    for energy in HSenergies:
        for pdfset in HSpdfsets:
            if (pdfset == 'MSHT20xNNPDF40_aN3LO_qed' or pdfset == 'MSHT20xNNPDF40_NNLO_qed') and energy != 14: # for now the approximate PDF is available at 14 TeV
                continue 
            for MH in HSMH:
                df1 = HSresults[(result_type, 'N3LON3LL', energy, pdfset, MH)]
                df2 = HSresults[(result_type, 'NNLO', energy, pdfset, MH)]
                if result_type != 'Mhh':
                    K3N3LL2[(result_type, energy, pdfset, MH)] = df1/df2
                else:
                    # Keep first two columns from df1
                    first_two = df1.iloc[:, :2]
                    # Divide the rest
                    divided = df1.iloc[:, 2:] / df2.iloc[:, 2:]
                    # Combine them back together
                    result = pd.concat([first_two, divided], axis=1)
                    K3N3LL2[(result_type, energy, pdfset, MH)]= result

# generate the Delta3PDF for the aN3LO vs NNLO pdf:
# Delta3PDF = | [N3LO+N3LL(aN3LO PDF) - N3LO+N3LL(NNLO PDF)] / N3LO+N3LL(aN3LO PDF) |
Delta3PDF = {} # dictionary for the result
for result_type in HStypes:
    for energy in HSenergies:
        if energy != 14: # for now the approximate PDF is available at 14 TeV
            continue 
        df1 = HSresults[(result_type, 'N3LON3LL', energy, 'MSHT20xNNPDF40_aN3LO_qed', MH)]
        df2 = HSresults[(result_type, 'N3LON3LL', energy, 'MSHT20xNNPDF40_NNLO_qed', MH)]
        if result_type != 'Mhh':
            Delta3PDF[(result_type, energy, MH)] = abs((df1-df2)/df1)
        else:
            # Keep first two columns from df1
            first_two = df1.iloc[:, :2]
            # construct the result
            divided = abs(df1.iloc[:, 2:] -  df2.iloc[:, 2:]  / df1.iloc[:, 2:])
            # Combine them back together
            result = pd.concat([first_two, divided], axis=1)
            Delta3PDF[(result_type, energy, MH)]= result

            
############################
# read+plot NLO EW K-factors
############################
EWresults = {} # dictionary to hold the cross section
EWresults_directory = 'EW' # directory for Hua-Sheng's results
EWtypes = ['pth', 'Mhh']
EWpdfset = 'PDF4LHC'
EWenergies = [13.6, 13, 14]
           
#Mhh_kfac-PDF4LHC-13.6TeV_muf=mur=0.5Mhh.dat
for result_type in EWtypes:
        for energy in EWenergies:
            data = read_EW(EWresults_directory, result_type, energy, EWpdfset)
            EWresults[(result_type, energy, 'PDF4LHC21_40')] = data

# EW total XSEC (the name conforms with the N3LO PDF to simplify plotting)
EWresults[('TotalXS', 13, 'PDF4LHC21_40')] = 0.959
EWresults[('TotalXS', 13.6, 'PDF4LHC21_40')] = 0.959
EWresults[('TotalXS', 14, 'PDF4LHC21_40')] = 0.958


####################################
# read+plot NNLO_FTapprox K-factors
####################################
            
###########
# TESTING #
###########

# N3LO TESTING ONLY:
# print N3LO/NNLO K-factors
print('TotalXS N3LO+N3LL/NNLO (1.,1.) @ 13.6 TeV=', K3N3LL2[('TotalXS', 13.6, 'PDF4LHC21_40', 125)]['(1.,1.)'].to_string(index=False))
#print('Mhh N3LO+N3LL/NNLO (1.,1.) @ 13.6 TeV=', K3N3LL2[('Mhh', 13.6, 'PDF4LHC21_40', 125)]['(1.,1.)'].to_string(index=False))

# print Delta3PDF (Error due to not using N3LO PDFs)
print('Delta3PDF (1.,1.) @ 14 TeV=', Delta3PDF[('TotalXS', 14, 125)]['(1.,1.)'].to_string(index=False))

# test plots (N3LO): 
plot_HS(HSresults, K3N3LL2, 'Mhh', 13.6, 'PDF4LHC21_40', 125, envelope=True, points=True)

# EW+N3LO TESTING (EW k-factor in the lower panel)
plot_HS_EW(HSresults, EWresults, K3N3LL2, 'Mhh', 13.6, 'PDF4LHC21_40', 125, envelope=True, points=True)

