import pandas as pd
import numpy as np

a = pd.read_hdf('muonfile/MuonEff.h5',key = 'MuonEff_train')
b = pd.read_hdf('muonfile/MuonEff.h5',key = 'MuonEff_test')

idxa_con = np.zeros(len(a.index))
idxa_tra = np.zeros(len(a.index))

idxb_con = np.zeros(len(b.index))
idxb_tra = np.zeros(len(b.index))

a = a.reset_index(drop = True)
b = b.reset_index(drop = True)


for i in range(0,len(a.index)-1):
    if (a['muon_endpoint[0]'][i]>-350)&(a['muon_endpoint[0]'][i]<350)&(a['muon_endpoint[1]'][i]>-150)&(a['muon_endpoint[1]'][i]<150)&(a['muon_endpoint[2]'][i]>0)&(a['muon_endpoint[2]'][i]<500): idxa_con[i] = 1
    if (a['muon_endpoint[0]'][i]==0)&(a['muon_endpoint[1]'][i]==0)&(a['muon_endpoint[2]'][i]==0): idxa_con[i] = 0
    if (a['muon_tracker'][i]==1): idxa_tra[i] = 1
    if (a['muon_endpoint[0]'][i]==0)&(a['muon_endpoint[1]'][i]==0)&(a['muon_endpoint[2]'][i]==0): idxa_tra[i] = 0

for i in range(0,len(b.index)-1):
     if (b['muon_endpoint[0]'][i]>-350)&(b['muon_endpoint[0]'][i]<350)&(b['muon_endpoint[1]'][i]>-150)&(b['muon_endpoint[1]'][i]<150)&(b['muon_endpoint[2]'][i]>0)&(b['muon_endpoint[2]'][i]<500): idxb_con[i] = 1
     if (b['muon_endpoint[0]'][i]==0)&(b['muon_endpoint[1]'][i]==0)&(b['muon_endpoint[2]'][i]==0): idxb_con[i] = 0    
     if (b['muon_tracker'][i]==1): idxb_tra[i] = 1
     if (b['muon_endpoint[0]'][i]==0)&(b['muon_endpoint[1]'][i]==0)&(b['muon_endpoint[2]'][i]==0): idxb_tra[i] = 0


a['new_contained'] = idxa_con
a['new_tracker'] = idxa_tra

b['new_contained'] = idxb_con
b['new_tracker'] = idxb_tra

a.to_hdf('muonfile/muoneff_train.h5',key = 'h')
b.to_hdf('muonfile/muoneff_test.h5',key = 'h')


