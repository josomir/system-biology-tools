sys.path.append(".")
from ACNNFeaturizer import ACNNFeaturizer
import numpy as np
import pandas as pd

import deepchem as dc
from deepchem.splits import RandomSplitter
from deepchem.data import NumpyDataset
from deepchem.models import AtomicConvModel

import time

import sys
import ray
from ray import tune

start_time = time.time()

#@ray.remote(num_gpus=2)
def custom_train(layer_sizes, learning_rate, dropouts):
  raw_dataset = pd.read_pickle('/scratch/IRB/jmesaric/featurized_dataset')
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
  print (valid)

  acm = AtomicConvModel(n_tasks=1,
                        frag1_num_atoms=frag1_num_atoms,
                        frag2_num_atoms=frag2_num_atoms,
                        complex_num_atoms=frag1_num_atoms+frag2_num_atoms,
                        max_num_neighbors=max_num_neighbors,
                        batch_size=4,
                        layer_sizes=layer_sizes,
                        learning_rate=learning_rate,
                        dropouts=dropouts
                        )

  losses, validation_losses = [], []
  # # Fit model
  max_epochs = 10
  for epoch in range(max_epochs):
    loss = acm.fit(train, nb_epoch=1, max_checkpoints_to_keep=1, all_losses=losses)
    print ("This is loss: ", loss)
    metric = dc.metrics.Metric(dc.metrics.score_function.rms_score)
    validation_losses.append(acm.evaluate(valid, metrics=[metric])['rms_score']**2)  # L2 Loss

  print ("--- %s minutes ---" % ((time.time() - start_time)/60))
  print ("Losses: ", validation_losses)

  score = dc.metrics.Metric(dc.metrics.score_function.pearson_r2_score)
  validation_score = acm.evaluate(valid, metrics=[score])['pearson_r2_score']
  print ('Pearson R2 validation score: ', validation_score)
  return (validation_score)


#custom_train(layer_sizes=[32, 32, 16], learning_rate=0.001)

############################### 
# HYPERPARAMETER OPTIMIZATION #
###############################
def main():
  ray.init(address='auto',_redis_password=sys.argv[1]) # IMPORTANAT, otherwise it won't work!!!
 
  #@ray.remote(num_gpus=2)
  def training_function(config):
      # Hyperparameters
      layer_sizes = config["layer_sizes"]
      learning_rate = config["learning_rate"]
      dropouts = config["dropouts"]
      # Iterative training function - can be any arbitrary training procedure.
      intermediate_score = custom_train(layer_sizes=layer_sizes, learning_rate=learning_rate, dropouts=dropouts)
      # Feed the score back back to Tune.
      tune.report(mean_loss=intermediate_score)
  

  analysis = tune.run(
      training_function,
      #num_samples=10,
      resources_per_trial={'cpu': 16, 'gpu': 2},
      config={
          "learning_rate": tune.grid_search([0.001, 0.002, 0.003]),
          "layer_sizes": tune.grid_search([[24,24,12],[32,32,16]]),
          #"layer_sizes": tune.grid_search([[8,8,4]]),
          "dropouts": tune.grid_search([0.1, 0.3, 0.4])
          #"dropouts": tune.grid_search([0.1])
      })

  print("Best config: ", analysis.get_best_config(
      metric="mean_loss", mode="max"))

  # Get a dataframe for analyzing trial results.
  df = analysis.results_df
  print (df)



if __name__ == '__main__':
  main()