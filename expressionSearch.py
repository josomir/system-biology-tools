from numpy import split
import pandas as pd
import numpy as np



def search(cell_lines):
    file = './CosmicCLP_CompleteGeneExpression.tsv'
    df = pd.read_csv(file, sep='\t', chunksize=10000000)
    for chunk in df:
        filtered_chunk = chunk[chunk['SAMPLE_NAME'].isin(cell_lines)]
        print (filtered_chunk)
        # mode='a' -> dodaj u postojeci file
        with open('CosmicExpressionsOvary.csv', 'a') as f: # nuzno je dodaje uvijek header kod appenda
            filtered_chunk.to_csv('CosmicExpressionsOvary.csv', mode='a', header=f.tell()==0)
        



cl = pd.read_csv('./ovary_cell_lines.csv')
cell_lines = cl['name'].to_numpy()
print (cell_lines)
search(cell_lines)
