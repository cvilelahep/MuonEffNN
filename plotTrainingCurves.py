import pandas as pd
import matplotlib.pyplot as plt

outDir = "/storage/shared/cvilela/MuonEff/FirstTry"

trainDF = pd.read_csv(outDir+"/"+"trainLoss.log", header = None)
validationDF = pd.read_csv(outDir+"/"+"validationLoss.log", header = None)

ax = trainDF.plot(x = 0, y = 2, label = "Training")
validationDF.plot(x = 0, y = 2, label = "Validation", ax = ax)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(outDir+"/"+"trainingCurves.png")
plt.show()

