# Copyright 2017 Goekcen Eraslan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import os
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
import torch
from sklearn.model_selection import train_test_split

class AnnSequence:
    def __init__(self, matrix: np.ndarray, batch_size: int, sf: np.ndarray = None):
        self.matrix = matrix
        self.size_factors = sf if sf is not None else np.ones((matrix.shape[0], 1), dtype=np.float64)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.matrix) // self.batch_size

    def __getitem__(self, idx: int):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch = self.matrix[start:end]
        batch_sf = self.size_factors[start:end]
        return {'count': batch, 'size_factors': batch_sf}, batch

def read_dataset(adata, transpose: bool = False, test_split: bool = False, copy: bool = False):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise TypeError("adata must be a scanpy AnnData object or a filepath to one.")

    norm_error = "Dataset must contain unnormalized count data in adata.X"
    assert 'n_count' not in adata.obs, norm_error

    if adata.X.size < 50e6:
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if transpose:
        adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        split_series = pd.Series(['train'] * adata.n_obs)
        split_series.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = split_series.values
    else:
        adata.obs['DCA_split'] = 'train'

    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    print(f'### Autoencoder: Successfully preprocessed {adata.n_vars} genes and {adata.n_obs} cells.')

    return adata

def normalize(adata, filter_min_counts: bool = True, size_factors: bool = True,
              normalize_input: bool = True, logtrans_input: bool = True):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    adata.raw = adata.copy() if (size_factors or normalize_input or logtrans_input) else adata

    if size_factors:
        sc.pp.normalize_total(adata, target_sum=1e4)
        adata.obs['n_counts'] = adata.X.sum(axis=1).A1 if sp.sparse.issparse(adata.X) else adata.X.sum(axis=1)
        adata.obs['size_factors'] = adata.obs['n_counts'] / np.median(adata.obs['n_counts'])
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

def read_genelist(filename: str):
    with open(filename, 'rt') as f:
        genelist = list(set(f.read().strip().split('\n')))
    assert genelist, 'No genes detected in genelist file'
    print(f'### Autoencoder: Subset of {len(genelist)} genes will be denoised.')
    return genelist


def write_text_matrix(matrix, filename: str, rownames=None, colnames=None, transpose: bool = False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    df = pd.DataFrame(matrix, index=rownames, columns=colnames)
    df.to_csv(filename, sep='\t',
              index=rownames is not None,
              header=colnames is not None,
              float_format='%.6f')
    
def read_pickle(inputfile: str):
    with open(inputfile, "rb") as f:
        return pickle.load(f)

def louvain_init_clustering(model, adata, knn=20, resolution=0.8):
    pretrain_latent = model.encodeBatch(torch.tensor(adata.X)).cpu().numpy()
    adata_latent = sc.AnnData(pretrain_latent)
    sc.pp.neighbors(adata_latent, n_neighbors=knn, use_rep="X")
    sc.tl.louvain(adata_latent, resolution=resolution)
    y_pred_init = np.asarray(adata_latent.obs['louvain'],dtype=int)
    features = pd.DataFrame(adata_latent.X,index=np.arange(0,adata_latent.n_obs))
    Group = pd.Series(y_pred_init,index=np.arange(0,adata_latent.n_obs),name="Group")
    Mergefeature = pd.concat([features,Group],axis=1)
    cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
    n_clusters = cluster_centers.shape[0]
    return cluster_centers, n_clusters, y_pred_init