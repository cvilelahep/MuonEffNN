import torch
import muonEffModel

import os
import glob

import pandas as pd
import numpy as np
import uproot

import math

CAF_FHC_fName = "/storage/shared/cvilela/CAF/mcc11_v4/ND_FHC_FV_*.root"
CAF_RHC_fName = "/storage/shared/cvilela/CAF/mcc11_v4/ND_RHC_FV_*.root"

CAF_files = glob.glob(CAF_FHC_fName)
CAF_files += glob.glob(CAF_RHC_fName)

# Process bachSize events at a time
batchSize = 1000

outDir = "/storage/shared/cvilela/MuonEff/FirstTry"

try :
    os.makedirs(outDir+"/"+"outTrees")
except(FileExistsError) :
    pass

# Get saved model
net = muonEffModel.muonEffModel().cuda()
net.load_state_dict(torch.load(outDir+"/"+"muonEff.nn"))
net.eval()

for f in CAF_files :
    
    # Read in file
    varList = ['LepMomX',
               'LepMomY', 
               'LepMomZ',
               'vtx_x',
               'vtx_y',
               'vtx_z',
               'muon_contained',
               'muon_tracker', 
               'isCC',
               'LepPDG',
               'Ev']
    try :
        print(f)
        fUprootIn = uproot.open(f)
        tIn = fUprootIn['caf']
    except :
        print("Didn't find caf tree. Skipping...")
        continue
    
    dIn = tIn.arrays(varList, outputtype = pd.DataFrame)
    
    features = dIn[['LepMomX', 'LepMomY', 'LepMomZ', 'vtx_x', 'vtx_y', 'vtx_z']]
    
    N_Events = len(dIn)
    N_Batches = math.ceil(float(N_Events)/batchSize)

    predictions = []

    for i in range(0, N_Batches) :
        print("Processing batch "+str(i)+" out of "+str(N_Batches))
        N_First = i*batchSize
        if (i+1)*batchSize <= N_Events :
            N_Last = (i+1)*batchSize
        else :
            N_Last = N_Events
        
        data = torch.as_tensor(features.iloc[N_First:N_Last].values).type(torch.FloatTensor).cuda()
        
        prediction = net(data)
        
        prediction = torch.nn.functional.softmax(prediction)
        
        predictions.append(prediction.cpu().detach().numpy())
        
        del data
        
    predictions = np.concatenate(predictions)
    print(predictions.shape)
    outFname = outDir+"/"+"outTrees"+"/"+os.path.splitext(os.path.basename(f))[0]+"_MuonEff.root"

    with uproot.recreate(outFname) as f:
        f["muonEfficiency"] = uproot.newtree({'LepMomX' : np.float64,
                                              'LepMomY' : np.float64,
                                              'LepMomZ' : np.float64,
                                              'vtx_x' : np.float64,
                                              'vtx_y' : np.float64,
                                              'vtx_z' : np.float64,
                                              'muon_contained' : np.int32,
                                              'muon_tracker' : np.int32,
                                              'isCC' : np.int32,
                                              'LepPDG' : np.int32,
                                              'Ev' : np.float64,
                                              'nnEffContained' : np.float64,
                                              'nnEffTracker' : np.float64,
                                              'nnEffNotAcc'  : np.float64})


        extend_dict = {}
        for var in varList :
            extend_dict[var] = dIn[var].values
        extend_dict['nnEffContained'] = predictions[:,0]
        extend_dict['nnEffTracker'] = predictions[:,1]
        extend_dict['nnEffNotAcc'] = predictions[:,2]
        
        f["muonEfficiency"].extend(extend_dict)
