import os
import numpy as np
import pandas as pd
from deepchem.utils import download_url, load_from_disk
from deepchem.feat import AtomicConvFeaturizer
import time
from pdbfixer import PDBFixer

class ACNNFeaturizer():
    def __init__(self, frag1_num_atoms, frag2_num_atoms, max_num_neighbors, neighbor_cutoff, dataset_path, featurized_dataset_name, pdbs_dir):
        self.frag1_num_atoms = frag1_num_atoms
        self.frag2_num_atoms = frag2_num_atoms
        self.complex_num_atoms = frag1_num_atoms + frag2_num_atoms
        self.max_num_neighbors = max_num_neighbors
        self.neighbor_cutoff = neighbor_cutoff
        self.dataset_path = dataset_path
        self.featurized_dataset_name = featurized_dataset_name
        self.pdbs_dir = pdbs_dir


    def loadCSVDataset(self):
        raw_dataset = load_from_disk(self.dataset_path)
        raw_dataset = raw_dataset[['PDB_ID_1', 'PDB_ID_2', 'RMSD', 'DASA']]
        raw_dataset['PDB_ID_1'] = raw_dataset['PDB_ID_1'].map(lambda x: x.split('_')[0])
        raw_dataset['PDB_ID_2'] = raw_dataset['PDB_ID_2'].map(lambda x: x.split('_')[0])
        raw_dataset = raw_dataset.rename(columns={'PDB_ID_1': 'pdbid_1', 'PDB_ID_2': 'pdbid_2'})
        raw_dataset['complex_id'] = raw_dataset['pdbid_1'] + '-' + raw_dataset['pdbid_2']
        return raw_dataset

    def downloadPDBFiles(self, raw_dataset):
        for pdbid_1, pdbid_2 in zip(raw_dataset['pdbid_1'], raw_dataset['pdbid_2']):
            for pdbid in [pdbid_1, pdbid_2]:
                print ("Checking if {0}.pdb exists ".format(pdbid))
                pdb_path = "{}.pdb".format(os.path.join(self.pdbs_dir, pdbid))
                if not os.path.exists(pdb_path) and pdbid != '':
                    print ("Downloading PDB structure for", pdbid)
                    fixer = PDBFixer(pdbid=pdbid)
                    PDBFile.writeFile(fixer.topology, fixer.positions, open(pdb_path, 'w'))

    def featurize(self):
        total_count = 0
        failed_count = 0
        start_time = time.time()
        raw_dataset = self.loadCSVDataset()
        downloadPDBFiles = self.downloadPDBFiles(raw_dataset)
        # # Adding featurized complexs column to dataset
        complex_featurizer = AtomicConvFeaturizer(self.frag1_num_atoms, self.frag2_num_atoms,
                                                    self.complex_num_atoms,
                                                    self.max_num_neighbors, self.neighbor_cutoff)
        complex_featurizers = {}
        for pdbid_1, pdbid_2, complex_id in zip(raw_dataset['pdbid_1'], raw_dataset['pdbid_2'], raw_dataset['complex_id']):
            total_count += 1
            pdbid_1_file = "{0}.pdb".format(os.path.join(self.pdbs_dir, pdbid_1))
            pdbid_2_file = "{0}.pdb".format(os.path.join(self.pdbs_dir, pdbid_2))
            if pdbid_1 and pdbid_2:
                try:
                    print ("Complex: {0} and {1}".format(pdbid_1, pdbid_2))
                    (frag1_coords, frag1_neighbor_list, frag1_z, frag2_coords,
                        frag2_neighbor_list, frag2_z, complex_coords, complex_neighbor_list,
                        complex_z) = complex_featurizer._featurize((pdbid_1_file, pdbid_2_file)) # featurizing complex

                    complex_featurizers[str(complex_id)]= [frag1_coords, frag1_neighbor_list, frag1_z, frag2_coords,
                        frag2_neighbor_list, frag2_z, complex_coords, complex_neighbor_list,
                        complex_z]
                except:
                    failed_count += 1
                    print (" Featurizier couldn't get protein complex structure: ", complex_id)

        new = pd.DataFrame.from_dict(complex_featurizers, orient='index',  
                columns=['frag1_coords', 'frag1_neighbor_list', 'frag1_z', 'frag2_coords',
                'frag2_neighbor_list', 'frag2_z', 'complex_coords', 'complex_neighbor_list', 'complex_z'])

        raw_dataset = pd.merge(raw_dataset, new, left_on='complex_id', right_index=True, how='right')

        print ("--- %s seconds ---" % (time.time() - start_time))
        print ("{0}/{1} complexes featurized".format(total_count-failed_count, total_count))
        # Save dataset to CSV and pickle file
        print ("Saving featurized dataset.")
        raw_dataset.to_pickle(self.featurized_dataset_name)
        raw_dataset.to_csv(self.featurized_dataset_name + '.csv')
        return raw_dataset
