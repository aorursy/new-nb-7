import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import time



from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor
FOLDER = '../input/'



# kaggle cloud has no output folder

# os.makedirs('output',exist_ok=True)

OUTPUT = '.'

os.listdir(FOLDER)
# df_mulliken_charges = pd.read_csv(FOLDER + 'mulliken_charges.csv')

# df_sample =  pd.read_csv(FOLDER + 'sample_submission.csv')

# df_magnetic_shielding_tensors = pd.read_csv(FOLDER + 'magnetic_shielding_tensors.csv')

df_train = pd.read_csv(FOLDER + 'train.csv')

# df_test = pd.read_csv(FOLDER + 'test.csv')

# df_dipole_moments = pd.read_csv(FOLDER + 'dipole_moments.csv')

# df_potential_energy = pd.read_csv(FOLDER + 'potential_energy.csv')

df_structures = pd.read_csv(FOLDER + 'structures.csv')

# df_scalar_coupling_contributions = pd.read_csv(FOLDER + 'scalar_coupling_contributions.csv')
def get_dist_matrix(df_structures_idx, molecule):

    df_temp = df_structures_idx.loc[molecule]

    locs = df_temp[['x','y','z']].values

    num_atoms = len(locs)

    loc_tile = np.tile(locs.T, (num_atoms,1,1))

    dist_mat = ((loc_tile - loc_tile.T)**2).sum(axis=1)

    return dist_mat



def assign_atoms_index(df_idx, molecule):

    se_0 = df_idx.loc[molecule]['atom_index_0']

    se_1 = df_idx.loc[molecule]['atom_index_1']

    if type(se_0) == np.int64:

        se_0 = pd.Series(se_0)

    if type(se_1) == np.int64:

        se_1 = pd.Series(se_1)

    assign_idx = pd.concat([se_0, se_1]).unique()

    assign_idx.sort()

    return assign_idx

def get_pickup_dist_matrix(df_idx, df_structures_idx, molecule, num_pickup=5, atoms=['H', 'C', 'N', 'O', 'F']):

    pickup_dist_matrix = np.zeros([0, len(atoms)*num_pickup])

    assigned_idxs = assign_atoms_index(df_idx, molecule) # [0, 1, 2, 3, 4, 5, 6] -> [1, 2, 3, 4, 5, 6]

    dist_mat = get_dist_matrix(df_structures_idx, molecule)

    for idx in assigned_idxs: # [1, 2, 3, 4, 5, 6] -> [2]

        df_temp = df_structures_idx.loc[molecule]

        locs = df_temp[['x','y','z']].values



        dist_arr = dist_mat[idx] # (7, 7) -> (7, )



        atoms_mole = df_structures_idx.loc[molecule]['atom'].values # ['O', 'C', 'C', 'N', 'H', 'H', 'H']

        atoms_mole_idx = df_structures_idx.loc[molecule]['atom_index'].values # [0, 1, 2, 3, 4, 5, 6]



        mask_atoms_mole_idx = atoms_mole_idx != idx # [ True,  True, False,  True,  True,  True,  True]

        masked_atoms = atoms_mole[mask_atoms_mole_idx] # ['O', 'C', 'N', 'H', 'H', 'H']

        masked_atoms_idx = atoms_mole_idx[mask_atoms_mole_idx]  # [0, 1, 3, 4, 5, 6]

        masked_dist_arr = dist_arr[mask_atoms_mole_idx]  # [ 5.48387003, 2.15181049, 1.33269675, 10.0578779, 4.34733927, 4.34727838]

        masked_locs = locs[masked_atoms_idx]



        sorting_idx = np.argsort(masked_dist_arr) # [2, 1, 5, 4, 0, 3]

        sorted_atoms_idx = masked_atoms_idx[sorting_idx] # [3, 1, 6, 5, 0, 4]

        sorted_atoms = masked_atoms[sorting_idx] # ['N', 'C', 'H', 'H', 'O', 'H']

        sorted_dist_arr = 1/masked_dist_arr[sorting_idx] #[0.75035825,0.46472494,0.23002898,0.23002576,0.18235297,0.09942455]



        target_matrix = np.zeros([len(atoms), num_pickup])

        for a, atom in enumerate(atoms):

            pickup_atom = sorted_atoms == atom # [False, False,  True,  True, False,  True]

            pickup_dist = sorted_dist_arr[pickup_atom] # [0.23002898, 0.23002576, 0.09942455]



            num_atom = len(pickup_dist)

            if num_atom > num_pickup:

                target_matrix[a, :num_pickup] = pickup_dist[:num_pickup]

            else:

                target_matrix[a, :num_atom] = pickup_dist

        

        pickup_dist_matrix = np.vstack([pickup_dist_matrix, target_matrix.reshape(-1)])

    return pickup_dist_matrix #(num_atoms, num_pickup*5)
# define index for faster computations

df_structures_idx = df_structures.set_index('molecule_name')

df_train_idx = df_train.set_index('molecule_name')    

    

# only 5 hydrogen atoms are considered as inverse squared distance

num = 5



mols = df_train['molecule_name'].unique()

num_div = len(mols) // 5

dist_mat = np.zeros([0, num*5])

atoms_idx = np.zeros([0], dtype=np.int32)

molecule_names = np.empty([0])

start = time.time()



# number of molecules to process

max_mol = 100000

k = 0

print ("Calculating ",max_mol," molecules.")



# from joblib import parallel_backend

# with parallel_backend('threading', n_jobs=2):

#    Parallel()(delayed)



for mol in mols[:max_mol]:

    k += 1

    if k and k % 1000 == 0:

        print(k, ': ' , mol)

    

    assigned_idxs = assign_atoms_index(df_train_idx, mol)

    dist_mat_mole = get_pickup_dist_matrix(df_train_idx, df_structures_idx, mol, num_pickup=num)

    mol_name_arr = [mol] * len(assigned_idxs) 



    molecule_names = np.hstack([molecule_names, mol_name_arr])

    atoms_idx = np.hstack([atoms_idx, assigned_idxs])

    dist_mat = np.vstack([dist_mat, dist_mat_mole])

    

col_name_list = []

atoms = ['H', 'C', 'N', 'O', 'F']

for a in atoms:

    for n in range(num):

        col_name_list.append('dist_{}_{}'.format(a, n))

        

se_mole = pd.Series(molecule_names, name='molecule_name')

se_atom_idx = pd.Series(atoms_idx, name='atom_index')

df_dist = pd.DataFrame(dist_mat, columns=col_name_list)

df_distance = pd.concat([se_mole, se_atom_idx,df_dist], axis=1)



elapsed_time = time.time() - start

print ("elapsed_time: {:4.2f}".format(elapsed_time) + " [sec]")

# kaggle cloud has no output folder

# os.makedirs('output')

df_distance.to_csv(OUTPUT + 'distance-XXX.csv', index=False)
# private dataset for speed purposes pre-calculted

# df_dist = pd.read_csv(OUTPUT + 'distance1000.csv')

df_dist.head()
# key index error in pandas

def merge_atom(df, df_distance):

    df_merge_0 = pd.merge(df, df_distance, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])

    df_merge_0_1 = pd.merge(df_merge_0, df_distance, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])

    del df_merge_0_1['atom_index_x'], df_merge_0_1['atom_index_y']

    return df_merge_0_1

df_train.head(15)
df_dist.head(15)
start = time.time()

# df_train_dist = merge_atom(df_train, df_dist)

# df_train_dist = pd.concat([df_train, df_dist], axis=1)

df_train_dist = merge_atom(df_train, df_distance) # corrected!: df_dist -> df_distance

elapsed_time = time.time() - start

print ("elapsed_time: {:4.2f}".format(elapsed_time) + " [sec]")

df_train_dist.head(15)
# takes a long time to write, unessecary if data is not processed further

# df_train_dist.to_csv(OUTPUT + 'train_dist-XXX.csv', index=False)
# private data included for speed purposes

# df_train_dist = pd.read_csv(OUTPUT + 'train_dist1000.csv')

df_train_dist.head(15)

df_1JHC = df_train_dist.query('type == "1JHC"')

y = df_1JHC['scalar_coupling_constant'].values

X = df_1JHC[df_1JHC.columns[6:]].values

print(X.shape)

print(y.shape)
# remove nan

X = np.nan_to_num(X)

X
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
start = time.time()

print('start training regressor')



# on the Kaggle cloud one should see 400% CPU use (quad thread)

# use hidden_layer_sizes=(100,50) for better results

# not sure this is best approach, depends on layer otpimization

# simple regression maybe better see https://scikit-learn.org/stable/modules/classes.html



## adam is a bit better but takes longer than 'sgd' solver

mlp = MLPRegressor(activation='relu', solver='sgd', hidden_layer_sizes=(100,50))

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_val)



# Ransac is much faster, but less accurate

## from sklearn.linear_model import RANSACRegressor

## from sklearn.datasets import make_regression

## ransac = RANSACRegressor(random_state=1234)

## ransac.fit(X_train, y_train)

## y_pred = ransac.predict(X_val)



## from sklearn.linear_model import TheilSenRegressor

## from sklearn.datasets import make_regression

## theilsen = TheilSenRegressor(random_state=1234)

## theilsen.fit(X_train, y_train)

## y_pred = theilsen.predict(X_val)



elapsed_time = time.time() - start

print ("elapsed_time: {:4.2f}".format(elapsed_time) + " [sec]")
print('len(y_val)  :',len(y_val))

print('len(y_pred) :', len(y_pred))
y_val.view()[:10]

y_val.view()[-10:]
plt.scatter(y_val, y_pred, marker='.')

plt.title('1JHC')

plt.plot([60, 220], [60, 220])

plt.show()

# not sure if correct, also axis labels crooked

from sklearn.linear_model import Ridge

from yellowbrick.regressor import ResidualsPlot



# Instantiate the linear model and visualizer

ridge = Ridge()

visualizer = ResidualsPlot(ridge)



visualizer.fit(X_train, y_train)   # Fit the training data to the model

visualizer.score(X_val, y_pred)    # Evaluate the model on the test data

visualizer.poof()                  # Draw/show/poof the data

y_pred.view()[:10]

y_pred.view()[-10:]
from sklearn.metrics import *

from math import sqrt



# current sklearn on kaggle version has no max_error

# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/regression.py

def max_error(y_true, y_pred):

    return np.max(np.abs(y_true - y_pred))



print("Mean squared error     : %.6f" %    mean_squared_error(y_val, y_pred))

print("Median absolute error  : %.6f" % median_absolute_error(y_val, y_pred))

print("Mean absolute error    : %.6f" %   mean_absolute_error(y_val, y_pred))

print("Maximum residual error : %.6f" %             max_error(y_val, y_pred))

print("                  RMSE : %.6f" % sqrt(mean_squared_error(y_val, y_pred)))

print("                    R2 : %.6f" %                 r2_score(y_val, y_pred))          


