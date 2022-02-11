import numpy as np
import pandas as pd

import deepchem as dc
from deepchem.utils import download_url, load_from_disk
from deepchem.splits import RandomSplitter
from deepchem.data import NumpyDataset
from deepchem.models import AtomicConvModel
import time

start_time = time.time()

raw_dataset = pd.read_pickle('new_dataset')
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


# # Modeling Binding Affinity
# 
# Docking is a useful, albeit coarse-grained tool for predicting protein-ligand binding affinities. However, it takes some time, especially for large-scale virtual screenings where we might be considering different protein targets and thousands of potential ligands. We might naturally ask then, can we train a machine learning model to predict docking scores? Let's try and find out!
# 
# Next, we'll need a way to transform our protein-ligand complexes into representations which can be used by learning algorithms. Ideally, we'd have neural protein-ligand complex fingerprints, but DeepChem doesn't yet have a good learned fingerprint of this sort. We do however have well-tuned manual featurizers that can help us with our challenge here.


class CustomAtomicConvModel(AtomicConvModel):
    def save():
        print ("bla")

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

losses, validation_losses = [], []



# # Fit model

max_epochs = 50

for epoch in range(max_epochs):
  loss = acm.fit(train, nb_epoch=1, max_checkpoints_to_keep=1, all_losses=losses)
  metric = dc.metrics.Metric(dc.metrics.score_function.rms_score)
  validation_losses.append(acm.evaluate(valid, metrics=[metric])['rms_score']**2)  # L2 Loss

print ("--- %s minutes ---" % ((time.time() - start_time)/60))
print ("Losses: ", validation_losses)

datasets = train, valid, test
score = dc.metrics.Metric(dc.metrics.score_function.pearson_r2_score)
for tvt, ds in zip(['train', 'val', 'test'], datasets):
  print(tvt, acm.evaluate(ds, metrics=[score]))

print ("--- %s minutes ---" % ((time.time() - start_time)/60))
