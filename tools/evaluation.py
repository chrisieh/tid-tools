import numpy as np
from root_numpy import root2array
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d


def root_roc(sigf, bkgf, branch, tree="CollectionTree"):
    """Returns fpr, tpr, thr for two root-files containing classifier scores"""
    sig = root2array(sigf, treename=tree, branches=[branch, "weight"])
    bkg = root2array(bkgf, treename=tree, branches=[branch, "weight"])

    sig_score = sig[branchname]
    sig_weight = sig["weight"]
    sig_truth = np.ones_like(sig_score)

    bkg_score = bkg[branchname]
    bkg_weight = sig["weight"]
    bkg_truth = np.zeros_like(bkg_score)

    score = np.concatenate([sig_score, bkg_score])
    weight = np.concatenate([sig_weight, bkg_weight])
    truth = np.concatenate([sig_truth, bkg_truth])

    return roc_curve(truth, score, sample_weight=weight)


def tmva_roc(rootf):
    """Returns fpr, tpr, thr for a TMVA-style root-file"""
    data = root2array(rootf, treename="TestTree",
                      branches=["classifier", "classID", "weight"])
    
    # Convert tmva scheme (sig: 0, bkg: 1) to sig: 1, bkg: 0
    truth = data["classID"]
    sig = (truth == 0)
    bkg = np.logical_not(sig)
    truth[sig] = 1
    truth[bkg] = 0

    return roc_curve(truth, data["classifier"], sample_weight=data["weight"])


def eff_rej(fpr, tpr):
    """Calculate eff and rej from fpr and tpr"""
    nonzero = (fps == 0)
    return tpr[nonzero], 1.0 / fps[nonzero]


def rej_fixed_eff(eff, rej, feff):
    """Calculates the rejection at fixed efficiencies 'feff'"""
    f = interp1d(eff, rej, copy=False)
    return f(feff)