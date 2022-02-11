import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deepchem as dc
from deepchem.splits import RandomSplitter
from deepchem.data import NumpyDataset
from deepchem.feat import AtomicConvFeaturizer
from deepchem.models import AtomicConvModel
import sklearn
import tensorflow as tf
import time

start_time = time.time()

raw_dataset = pd.read_pickle('featurized_dataset')
# max num atoms instead of exact.
frag1_num_atoms = 3000  # for ligand atoms
frag2_num_atoms = 3000  # for protein atoms
complex_num_atoms = frag1_num_atoms + frag2_num_atoms  # in total
max_num_neighbors = 5
# Cutoff in angstroms
neighbor_cutoff = 8


ids = raw_dataset['complex_id']
X = raw_dataset[['frag1_coords', 'frag1_neighbor_list', 'frag1_z', 'frag2_coords',
         'frag2_neighbor_list', 'frag2_z', 'complex_coords', 'complex_neighbor_list', 'complex_z']] # converting to nparray
y = np.asarray(raw_dataset['DASA'])
y = np.asarray(y).astype('float32')
w=np.full(np.array(y.shape), 1)
w = np.asarray(w).astype('float32')

dataset = NumpyDataset(ids=ids, X=X, y=y, n_tasks=1, w=w)
train, valid, test = RandomSplitter().train_valid_test_split(dataset, seed=777)
print ('Train: ', train)
print ('Valid: ', valid)
print ('Test: ', test)


acm = AtomicConvModel(n_tasks=1,
                      frag1_num_atoms=frag1_num_atoms,
                      frag2_num_atoms=frag2_num_atoms,
                      complex_num_atoms=frag1_num_atoms+frag2_num_atoms,
                      max_num_neighbors=max_num_neighbors,
                      batch_size=1,
                      layer_sizes=[32, 32, 16],
                      learning_rate=0.001,
                      dropouts=0.2
                      )

"""
###############################################################
##### Grid Hyperparameter Optimization using Ray package ######
###############################################################

#JOSIP
import ray

@ray.remote(num_gpus=1)
class RayNetWrapper:
  def __init__(self, net):
    self.net = net
  
  def train(self):
    return self.net.train()

ray.init()
actors = [RayNetWrapper.remote(acm) for _ in range(25)]
results = ray.get([actor.train.remote() for actor in actors])


#END
"""

"""
######################################
#### Fit model with hard-coded values
######################################
print ("Training model with predefined values...")
losses, validation_losses = [], []
max_epochs = 50

for epoch in range(max_epochs):
  loss = acm.fit(train, nb_epoch=1, max_checkpoints_to_keep=1, all_losses=losses)
  metric = dc.metrics.Metric(dc.metrics.score_function.rms_score)
  validation_losses.append(acm.evaluate(valid, metrics=[metric])['rms_score']**2)  # L2 Loss

print ("--- %s minutes ---" % ((time.time() - start_time)/60))
print ("Losses: ", validation_losses)

##############################
#### Evaluate model on train, test and validation datasets
###############################
datasets = train, valid, test
pearson_score = dc.metrics.Metric(dc.metrics.score_function.pearson_r2_score)
rms_score = dc.metrics.Metric(dc.metrics.score_function.rms_score)
for tvt, ds in zip(['train', 'val', 'test'], datasets):
  print(tvt, acm.evaluate(ds, metrics=[pearson_score, rms_score]))

print ("--- %s minutes ---" % ((time.time() - start_time)/60))
"""



#########################################################
##### Grid Hyperparameter Optimization ######
#########################################################
print ("Generating layer_sizes hyperparameter combinations...")
layer_sizes = []
for layer1, layer2, layer3, layer4 in zip(range(1,16), range(1,16), range(1,16), range(1,16)):
  layer_sizes.append([layer1,layer2,max(round(layer3/2),1)])

print (layer_sizes)
layer_sizes = [[24,24,12],
                [8,8,4],
                [32,32,16],
                [16,16,8]]

#layer_sizes = [[8,8,4],[4,4,2],[3,3,2],[6,6,3]]
grid_search_params = {'layer_sizes': layer_sizes,
          'dropouts': [0.2,0.3,0.4,0.5],
          'learning_rate':[0.001,0.002,0.003]}
metric = dc.metrics.Metric(dc.metrics.score_function.rms_score)

optimizer = dc.hyper.GridHyperparamOpt(lambda **p: AtomicConvModel(n_tasks=1,
                                                                  frag1_num_atoms=frag1_num_atoms,
                                                                  frag2_num_atoms=frag2_num_atoms,
                                                                  complex_num_atoms=frag1_num_atoms+frag2_num_atoms,
                                                                  max_num_neighbors=max_num_neighbors,
                                                                  batch_size=1,
                                                                  **p))
print ("Starting hyperparameter optimization...")
#try:
best_model, best_hyperparams, all_scores = optimizer.hyperparam_search(grid_search_params,train, 
                                                                      test, 
                                                                      metric, 
                                                                      nb_epoch=1, 
                                                                      max_checkpoints_to_keep=1)
#except:
#  print ("Failed hyperparameter optimization.")
#  print ("--- %s minutes ---" % ((time.time() - start_time)/60))
#  exit()
print ("--- %s minutes ---" % ((time.time() - start_time)/60))
#print ("Best model: ", best_model)
print ("Best hyperparams: ", best_hyperparams)
print ("All scores: ", all_scores)



"""
#########################################################
##### Gaussian Process Hyperparameter Optimization ######
#########################################################
gauss_search_params = {'learning_rate': 0.002}
search_range = {'learning_rate': 0.001}
metric = dc.metrics.Metric(dc.metrics.score_function.rms_score)
optimizer = dc.hyper.GaussianProcessHyperparamOpt(lambda **p: AtomicConvModel(n_tasks=1,
                                                                  frag1_num_atoms=frag1_num_atoms,
                                                                  frag2_num_atoms=frag2_num_atoms,
                                                                  complex_num_atoms=frag1_num_atoms+frag2_num_atoms,
                                                                  max_num_neighbors=max_num_neighbors,
                                                                  batch_size=1,
                                                                  layer_sizes=[16, 16, 8],
                                                                  **p))
best_hyperparams = optimizer.hyperparam_search(gauss_search_params,train, 
                                              test, 
                                              metric, 
                                              nb_epoch=1, 
                                              max_checkpoints_to_keep=1,                              
                                              search_range=search_range)
print ("Best hyperparams: ", best_hyperparams)
"""
