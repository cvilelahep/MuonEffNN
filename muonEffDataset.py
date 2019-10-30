import numpy as np
import pandas as pd

from torch.utils import data

class muonEffDataset(data.Dataset):
      def __init__(self, archive, datasetName):
            self.store = pd.HDFStore(archive, 'r')
            self.df = pd.read_hdf(self.store, datasetName)

      def __getitem__(self, index):
            return self.df.iloc[index, [self.df.columns.get_loc(col) for col in ['LepMomX', 'LepMomY', 'LepMomZ', 'vtx_x', 'vtx_y', 'vtx_z']]], self.df.iloc[index, [self.df.columns.get_loc(col) for col in ['muon_contained', 'muon_tracker']]]
      
      def __len__(self):
            return len(self.df)

      def close(self):
            del self.df
            self.store.close()

def collate(batch) :
      result = []
      for i in range(len(batch[0])):
            result.append(np.array([sample[i] for sample in batch]))
      return tuple(result)
