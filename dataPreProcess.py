import os
import glob

import pandas as pd
import numpy as np
import uproot

CAF_FHC_fName = "/storage/shared/cvilela/CAF/mcc11_v4/ND_FHC_FV_*.root"
CAF_RHC_fName = "/storage/shared/cvilela/CAF/mcc11_v4/ND_RHC_FV_*.root"

CAF_files = glob.glob(CAF_FHC_fName)
CAF_files += glob.glob(CAF_RHC_fName)

def main() :
    storeName = '/storage/shared/cvilela/MuonEff/MuonEff.h5'
    
    trainFraction = 0.8
    
    try :
        os.remove(storeName) 
    except OSError :
        pass

    store = pd.HDFStore(storeName)

    varList = ['LepMomX',
               'LepMomY', 
               'LepMomZ',
               'vtx_x',
               'vtx_y',
               'vtx_z',
               'muon_contained',
               'muon_tracker', 
               'isCC',
               'LepPDG']

    for f in CAF_files :
        try :
            fUproot = uproot.open(f)
            t = fUproot['caf']
        except :
            continue
        
        d = t.arrays(varList, outputtype = pd.DataFrame)
        
        # Remove non-CC events
        d.drop(d[d.isCC != 1].index, inplace = True)
        
        # Remove non-numu events
        d.drop(d[abs(d.LepPDG) != 13].index, inplace = True)

        # Don't need these variables anymore
        d.drop(columns=['isCC', 'LepPDG'], inplace = True)
        
        # Randomly select events into "Training" and "Test" samples
        mask = np.random.rand(len(d)) < trainFraction
        
        store.append('MuonEff_train', d[mask])
        store.append('MuonEff_test', d[~mask])
        
        del d
        del t
        del fUproot
        

if __name__ == "__main__" :
    main()
