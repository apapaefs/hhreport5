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


###################################################
# read+plot Hua-Sheng's N3LO+NLL and NNLO results:
###################################################
HSresults = {} # dictionary to hold the cross section
HSresults_directory = 'Data4YR5' # directory for Hua-Sheng's results
HStypes = ['TotalXS', 'Mhh']
HSorders = ['NNLO', 'N3LON3LL', 'N3LO']
HSpdfsets = ['PDF4LHC21_40']
HSMH = [125]
HSenergies = [13.6, 13, 14]
for result_type in HStypes:
    for order in HSorders:
        for energy in HSenergies:
            for pdfset in HSpdfsets:
                for MH in HSMH:
                    data = read_HS(HSresults_directory, result_type, order, energy, pdfset, MH)
                    HSresults[(result_type, order, energy, pdfset, MH)] = data

# generate the K-factors:
# N3LO+N3LL/NNLO
K3N3LL2 = {} # the N3LO+N3LL/NNLO K-factors
for result_type in HStypes:
    for energy in HSenergies:
        for pdfset in HSpdfsets:
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
                    
# test:
#print('TotalXS N3LO+N3LL/NNLO (1.,1.) @ 13.6 TeV=', K3N3LL2[('TotalXS', 13.6, 'PDF4LHC21_40', 125)]['(1.,1.)'].to_string(index=False))
#print('Mhh N3LO+N3LL/NNLO (1.,1.) @ 13.6 TeV=', K3N3LL2[('Mhh', 13.6, 'PDF4LHC21_40', 125)]['(1.,1.)'].to_string(index=False))


plot_HS(HSresults, K3N3LL2, 'Mhh', 13.6, 'PDF4LHC21_40', 125, envelope=True, points=True)
