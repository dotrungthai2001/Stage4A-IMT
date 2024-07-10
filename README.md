# Mini_projet_PAT
# See the related note ''Photoacoustic tomography (PAT) - Inverse problem''

We consider here collected signal s = Au (noiseless) or s = Au+e (noisy)
We developped various algorithms to recover u from measurements s

The PAT.py contains all the methods useful to construct the PAT system (direct model)

The model construction is made in run_model_construction.py

The algorithms.py groups all the algorithms that have been implemented to solve the inverse problem
1) Least Squares
Regularized least squares using:
2) Non negativity constraint
3) L2 constraint
4) l1 penalisation on wavelet coefs.
5) TV regularisation
6) Cauchy regularisation with FB splitting
7) Cauchy regularisation with BFGS

The quality measures (contrast, error, ...) are developed in error_measures.py file

The main file to be launched is run_reconstruction_tests.py
