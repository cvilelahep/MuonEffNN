# Neural network to capture muon selection efficiency for DUNE-PRISM

 - dataPreProcess.py: reads CAF files and saves muon kinematics and selection variables for numu CC events to HDF5 file
 - muonEffDataset.py: defines the pytorch dataset using the files produced with dataPreProcess.py
 - muonEffModel.py: neural network architecture and useful functions
 - trainMuonEff.py: script to train the neural network model and save training log and model to disk
 - plotTrainingCurves.py: script to plot training curves
 - makeTree.py: script to produce ROOT files with TTrees that include the output of the NN in addition to muon kinematics
