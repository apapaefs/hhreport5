import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hhreport5_plotting import *

##############################################
# functions
##############################################
# read Hua-Sheng's results and return an entry for the dictionary
def read_HS(directory, result_type, order, energy, pdfset, MH):
    filetoread = (
        f"{directory}/{result_type}/"
        f"{result_type}_{order}_{energy}TeV_{pdfset}_MH{MH}GeV.dat"
    )
    print("reading results from", filetoread)

    scale_cols = ["(1.,1.)", "(2.,1.)", "(0.5,1.)", "(1.,2.)",
                  "(2.,2.)", "(1.,0.5)", "(0.5,0.5)"]

    # Mhh (and typically pth) have two leading bin-edge columns
    if result_type in ["Mhh", "pth"]:
        colnames = ["left-edge", "right-edge"] + scale_cols
    else:
        colnames = scale_cols

    df = pd.read_csv(
        filetoread,
        comment="#",
        sep=r"\s+",
        engine="python",
        names=colnames,
        header=None
    )

    # Force numeric
    for c in colnames:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def read_EW(directory, result_type, energy, pdfset):
    filetoread = (
        f"{directory}/{result_type}"
        f"_kfac-{pdfset}-{energy}TeV_muf=mur=0.5Mhh.dat"
    )
    print("reading EW results from", filetoread)

    df = pd.read_csv(
        filetoread,
        sep=r"\s+",
        engine="python",
        header=None,
        names=["left-edge", "K_EW"]
    )

    df["left-edge"] = pd.to_numeric(df["left-edge"], errors="coerce")
    df["K_EW"]      = pd.to_numeric(df["K_EW"], errors="coerce")
    df = df.dropna(subset=["left-edge", "K_EW"]).reset_index(drop=True)

    return df

def read_NNLO_FTapprox(directory, result_type, order, energy):
    filetoread = (
        directory + '/' + 'LHC' + str(energy) + '/'
        + '1dd.plot.' + str(result_type) + '..' + str(order) + '.QCD.dat'
    )
    print('reading results from', filetoread)

    colnames = ["left-edge", "right-edge", "scale-central", "central-error",
                "scale-min", "min-error", "scale-max", "max-error",
                "rel-down", "rel-up"]

    df = pd.read_csv(
        filetoread,
        comment="#",
        sep=r"\s+",          
        engine="python",     
        names=colnames,
        header=None
    )

    # turn "-6.38%" -> -0.0638 and "3.52%" -> 0.0352
    for c in ["rel-down", "rel-up"]:
        df[c] = (
            df[c].astype(str)
                 .str.replace("%", "", regex=False)
                 .astype(float) / 100.0
        )

    return df

import pandas as pd


# find the min and max of the N3LO K-factors:
def minmax_k3n3ll2_scales(K3N3LL2, key, colnames):
    """
    Compute min/max of K3N3LL2[key] over the provided scale-variation columns.

    Parameters
    ----------
    K3N3LL2 : dict-like
        Dictionary holding ratio DataFrames.
    key : tuple
        Example: ('TotalXS', 13, 'PDF4LHC21_40', 125)
    colnames : list[str]
        Scale-variation column names to consider.

    Returns
    -------
    results : dict
        {
          "min_per_col": pd.Series,
          "max_per_col": pd.Series,
          "global_min": float,
          "global_min_col": str,
          "global_max": float,
          "global_max_col": str,
          "used_cols": list[str],
        }
    """
    if key not in K3N3LL2:
        raise KeyError(f"Key {key} not found in K3N3LL2")

    df = K3N3LL2[key]
    used_cols = [c for c in colnames if c in df.columns]
    if not used_cols:
        raise KeyError(
            f"None of the requested columns found for key={key}. "
            f"Requested={colnames}, available={list(df.columns)}"
        )

    sub = df[used_cols].apply(pd.to_numeric, errors="coerce")

    min_per_col = sub.min(axis=0, skipna=True)
    max_per_col = sub.max(axis=0, skipna=True)

    global_min = float(min_per_col.min())
    global_max = float(max_per_col.max())
    global_min_col = str(min_per_col.idxmin())
    global_max_col = str(max_per_col.idxmax())

    return {
        "min_per_col": min_per_col,
        "max_per_col": max_per_col,
        "global_min": global_min,
        "global_min_col": global_min_col,
        "global_max": global_max,
        "global_max_col": global_max_col,
        "used_cols": used_cols,
    }


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
EWresults = {} # dictionary to hold the k-factors
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
# NNLO_FTapprox total XSEC:
#sqrts       mH                XS             ± QCD Scale Unc.    ± THU     ± αs Unc. ± PDF Unc.
#13 TeV      125 GeV           30.75 fb       ± 4.98%             ± 0.14%   ± 1.51%   ± 1.96%   
#13.6 TeV    125 GeV           34.01 fb       ± 4.85%             ± 0.19%   ± 1.49%   ± 1.92%   
#14 TeV      125 GeV           36.27 fb       ± 4.78%             ± 0.21%   ± 1.47%   ± 1.89%

NNLO_FTapprox_results = {} # dictionary to hold NNLO_FTapprox results
NNLO_FTapprox_results_directory = 'NNLO_FTapprox' # directory for Hua-Sheng's results
NNLO_FTapprox_types = ['pT_h1', 'm_hh', 'total_rate']
NNLO_FTapprox_orders = ['NNLO']
NNLO_FTapprox_energies = [13.6, 13, 14]
NNLO_FTapprox_types_convert = { 'pT_h1': 'pth', 'm_hh': 'Mhh', 'total_rate': 'TotalXS'} # convert the naming of types to the global one

# test reading
#dftest = read_NNLO_FTapprox(NNLO_FTapprox_results_directory, 'm_hh', 'NNLO', 13.6)
#print(dftest)

# read in all the distributions
for result_type in NNLO_FTapprox_types:
        for energy in NNLO_FTapprox_energies:
            for order in NNLO_FTapprox_orders:
                data = read_NNLO_FTapprox(NNLO_FTapprox_results_directory, result_type, order, energy)
                NNLO_FTapprox_results[(NNLO_FTapprox_types_convert[result_type], order, energy)] = data
                print(NNLO_FTapprox_types_convert[result_type], order, energy)


###########
# TESTING #
###########

print('\nTESTING:')
                
#####################
# N3LO TESTING ONLY:
#####################

# get N3LO+N3LL K-factors and uncertainties:
print("N3LO+N3LL K-factors and uncertainties:")
colnames = ["(1.,1.)", "(2.,1.)", "(0.5,1.)", "(1.,2.)", "(2.,2.)", "(1.,0.5)", "(0.5,0.5)"]
key = ('TotalXS', 13, 'PDF4LHC21_40', 125)
K13 = np.float64(K3N3LL2[key]['(1.,1.)'].to_string(index=False))
res13 = minmax_k3n3ll2_scales(K3N3LL2, key, colnames)
key = ('TotalXS', 13.6, 'PDF4LHC21_40', 125)
K136 = np.float64(K3N3LL2[key]['(1.,1.)'].to_string(index=False))
res136 = minmax_k3n3ll2_scales(K3N3LL2, key, colnames)
key = ('TotalXS', 14, 'PDF4LHC21_40', 125)
K14 = np.float64(K3N3LL2[key]['(1.,1.)'].to_string(index=False))
res14 = minmax_k3n3ll2_scales(K3N3LL2, key, colnames)

print("\tK(13) =", K13, "+-", res13["global_max"]-K13, K13 - res13["global_min"])
print("\tK(13.6) =", K136, "+-", res136["global_max"]-K136, K136 - res136["global_min"])
print("\tK(14) =", K14, "+-", res14["global_max"]-K14, K14 - res14["global_min"])


# print N3LO/NNLO K-factors
#print('TotalXS N3LO+N3LL/NNLO K-Factor (1.,1.) @ 13 TeV=', K3N3LL2[('TotalXS', 13, 'PDF4LHC21_40', 125)]['(1.,1.)'].to_string(index=False))
#print('TotalXS N3LO+N3LL/NNLO K-Factor (1.,1.) @ 13.6 TeV=', K3N3LL2[('TotalXS', 13.6, 'PDF4LHC21_40', 125)]['(1.,1.)'].to_string(index=False))
#print('TotalXS N3LO+N3LL/NNLO K-Factor (1.,1.) @ 14 TeV=', K3N3LL2[('TotalXS', 14, 'PDF4LHC21_40', 125)]['(1.,1.)'].to_string(index=False))
#print('Mhh N3LO+N3LL/NNLO (1.,1.) @ 13.6 TeV=', K3N3LL2[('Mhh', 13.6, 'PDF4LHC21_40', 125)]['(1.,1.)'].to_string(index=False))

# print Delta3PDF (Error due to not using N3LO PDFs) -> FOR NOW THE APPROXIMATE PDF IS ONLY AVAILABLE AT 14 TeV
#print('Delta3PDF (1.,1.) @ 13 TeV=', Delta3PDF[('TotalXS', 13, 125)]['(1.,1.)'].to_string(index=False))
#print('Delta3PDF (1.,1.) @ 13.6 TeV=', Delta3PDF[('TotalXS', 13.6, 125)]['(1.,1.)'].to_string(index=False))
print('Delta3PDF (1.,1.) @ 14 TeV=', Delta3PDF[('TotalXS', 14, 125)]['(1.,1.)'].to_string(index=False))

# test plots (N3LO): 
plot_HS(HSresults, K3N3LL2, 'Mhh', 13.6, 'PDF4LHC21_40', 125, envelope=True, points=True)

###################
# EW+N3LO TESTING #
###################

# print EW K-factor
print('TotalXS EW K-Factor @ 13.6 TeV =', EWresults[('TotalXS', 13.6, 'PDF4LHC21_40')])

# plot EW+N3LO, as above but with EW k-factor in the lower panel
plot_HS_EW(HSresults, EWresults, K3N3LL2, 'Mhh', 13.6, 'PDF4LHC21_40', 125, envelope=True, points=True)

#########################
# NNLO_FTapprox TESTING #
#########################

# print total cross section K-factor (from FILE):
print('TotalXS NNLO_FTapprox @ 13.6 TeV (scale central) =', NNLO_FTapprox_results[('TotalXS', 'NNLO', 13.6)]['scale-central'].to_string(index=False))
# plot NNLO_FTapprox with scale variation envelope
plot_NNLOFTapprox(NNLO_FTapprox_results, 'Mhh', 'NNLO', 13.6)


#####################
# TEST COMBINATIONS #
#####################

# Plot individually (Mhh)
plot_HS_EW_NNLOFTapprox(HSresults, EWresults, NNLO_FTapprox_results, K3N3LL2,
                            'Mhh', 13.6, 'PDF4LHC21_40', 125,
                            envelope=True, points=True,
                            ft_order="NNLO")

# Plot the combination
plot_combination(HSresults, EWresults, NNLO_FTapprox_results, K3N3LL2,
                            'Mhh', 13, 'PDF4LHC21_40', 125,
                            envelope=True, points=True,
                            ft_order="NNLO", xmax=1500, ylog=False, include_unrescaled_FTapprox=True)

plot_combination(HSresults, EWresults, NNLO_FTapprox_results, K3N3LL2,
                            'Mhh', 13.6, 'PDF4LHC21_40', 125,
                            envelope=True, points=True,
                            ft_order="NNLO", xmax=1500, ylog=False, include_unrescaled_FTapprox=True)


plot_combination(HSresults, EWresults, NNLO_FTapprox_results, K3N3LL2,
                            'Mhh', 14, 'PDF4LHC21_40', 125,
                            envelope=True, points=True,
                            ft_order="NNLO", xmax=1500, ylog=False, include_unrescaled_FTapprox=True)
