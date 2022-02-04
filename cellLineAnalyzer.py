import abc
import os
import csv
from numpy import split
import pandas as pd
import numpy as np


def split_csv():
    file = './CosmicCLP_MutantExport.tsv'
    df = pd.read_csv(file, sep='\t', chunksize=500000)
    for i, chunk in enumerate(df):
        chunk.to_csv('out{}.csv'.format(i))


def filterPrimarySite(file, primarysite):
    df = pd.read_csv(file,sep=',')
    #a = list(df.columns)
    a = df[df['Primary site'] == "ovary"]
    print (a)
    a.to_csv('%s_filtered.csv'% file.split('.')[0])

def getDistinctSampleNames(file):
    df = pd.read_csv(file,sep=',')
    names = df['Sample name'].unique()
    return names.tolist()


original_files = [f for f in os.listdir('.') if f.startswith('out') and f.endswith('.csv') and '_' not in f]
print (original_files)
for file in original_files:
    pass
    #filterPrimarySite(file,"bla")
    
filtered_files = [f for f in os.listdir('.') if f.startswith('out') and f.endswith('.csv') and '_' in f]
print (filtered_files)
cell_lines = []
for file in filtered_files:
    cell_lines.extend(getDistinctSampleNames(file))
#print (cell_lines)
x = np.array(cell_lines)
cell_lines_unique = np.unique(x)
pd.DataFrame(cell_lines_unique).to_csv('ovary_cell_lines.csv', header=['name'])
print (cell_lines_unique)
print (len(cell_lines_unique))

