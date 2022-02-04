import pandas as pd
import numpy as np


df_samples = pd.read_csv('CosmicSample.tsv',sep='\t', low_memory=False)
df_lines = pd.read_csv('QC_lines.csv',sep=',', low_memory=False)


print (df_samples.columns.tolist())
print (df_lines.columns.tolist())


df_all = pd.merge(df_lines, df_samples, how='left', left_on=['COSMIC_ID'], right_on=['sample_id'])


print (pd.isna(df_all['primary_site']).value_counts())

df_all.to_csv('allCellLines.csv')