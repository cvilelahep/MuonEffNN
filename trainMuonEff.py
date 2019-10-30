import muonEffModel
import muonEffDataset
import torch
import time
import numpy as np
import os

class BLOB :
    pass

def main() :

    outDir = "/storage/shared/cvilela/MuonEff/FirstTry"

    try :
        os.makedirs(outDir)
    except (FileExistsError) :
        pass
    
    blob = BLOB()
    blob.net = muonEffModel.muonEffModel().cuda()
    blob.criterion = torch.nn.CrossEntropyLoss()
    blob.optimizer = torch.optim.Adam(blob.net.parameters())
    blob.data = None
    blob.label = None
    
    train_loader = torch.utils.data.DataLoader(muonEffDataset.muonEffDataset('/storage/shared/cvilela/MuonEff/MuonEff.h5', 'MuonEff_train'), batch_size=200, shuffle=True, num_workers=4, collate_fn = muonEffDataset.collate)
    test_loader =  torch.utils.data.DataLoader(muonEffDataset.muonEffDataset('/storage/shared/cvilela/MuonEff/MuonEff.h5', 'MuonEff_test'), batch_size=200, shuffle=True, num_workers=4, collate_fn =  muonEffDataset.collate)
    
    # Training loop
    TRAIN_EPOCH = 0.01
    blob.net.train()
    epoch = 0.
    iteration = 0.

    fTrainLossTracker = open(outDir+"/"+'trainLoss.log', 'w', 2048)
    fValidationLossTracker = open(outDir+"/"+'validationLoss.log', 'w', -1)

    while epoch < TRAIN_EPOCH :
        print('Epoch', epoch, int(epoch+0.5), 'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for i, data in enumerate(train_loader) :
            muonEffModel.FillLabel(blob,data)
            muonEffModel.FillData(blob,data)
            
            res = muonEffModel.forward(blob, True)
            
            muonEffModel.backward(blob)

            epoch += 1./len(train_loader)
            iteration += 1
            
            fTrainLossTracker.write(str(epoch)+","+str(iteration)+","+str(res['loss'])+'\n')

            if i == 0 or (i+1)%10 == 0 :
                print('TRAINING', 'Iteration', iteration, 'Epoch', epoch, 'Loss', res['loss'])
                
            
            if (i+1)%100 == 0 :
                with torch.no_grad() :
                    blob.net.eval()
                    test_data = next(iter(test_loader))
                    muonEffModel.FillLabel(blob,test_data)
                    muonEffModel.FillData(blob,test_data)
                    res = muonEffModel.forward(blob, False)

                    fValidationLossTracker.write(str(epoch)+","+str(iteration)+","+str(res['loss'])+'\n')

                    print('VALIDATION', 'Iteration', iteration, 'Epoch', epoch, 'Loss', res['loss'])

            if epoch >= TRAIN_EPOCH :
                break
    fTrainLossTracker.close()
    fValidationLossTracker.close()
    torch.save(blob.net.state_dict(), outDir+"/"+"muonEff.nn")
if __name__ == '__main__' :
    main()
