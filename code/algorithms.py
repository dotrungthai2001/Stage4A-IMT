# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:19:08 2022

File containing all the algorithms solving the inverse problem

@authors: Caroline, Paul
"""

from scipy.sparse.linalg import lsqr
import scipy.optimize
import numpy as np
import pywt

# least squares

def moindre_carres(A, s):
    '''least squares from scipy'''
    recon = lsqr(A, s, atol=0, btol=0, conlim=0,iter_lim = 25)
    return recon
    
# Regularized least squares

# gradient function to be used in the following l2-regularized algorithms
def gradient(u, A, s):
    
    '''computes the gradient as a function of u(the image)
    of the data fidelity part that is the gradient of 
    (1\2)||s - Au||^2'''
    
    Aus = A@u.reshape(-1) - s
    return A.T@Aus

# non negative least squares
def moindre_carres_nonneg(A, s, gamma, n_iter):
    '''non negative least squares --> projected gradient'''
    recon = np.zeros(np.size(A,1))
    F = np.zeros(n_iter)
    F[0] = 1/2*np.linalg.norm((s-(A@recon.reshape(-1))))**2
    for i in range(n_iter):
        if not(i%10):
            print('Non neg Least square iteration : '+f'{i}/ {n_iter}')
        recon = np.maximum(recon-gamma*gradient(recon,A,s),0)
        F[i] = 1/2*np.linalg.norm((s-(A@recon.reshape(-1))))**2
    return recon, F

# l2 penalized least squares
def moindre_carres_l2(A, s, gamma, labda, n_iter):
    '''non negative least squares --> projected gradient'''
    recon = np.zeros(np.size(A,1))
    F = np.zeros(n_iter)
    F[0] = 1/2*np.linalg.norm((s-(A@recon.reshape(-1))))**2+labda*np.linalg.norm(recon)**2
    for i in range(n_iter):
        if not(i%10):
            print('l2 reg Least square iteration : '+f'{i}/ {n_iter}')
        recon = recon-gamma*(gradient(recon,A,s)+2*recon)
        F[i] = 1/2*np.linalg.norm((s-(A@recon.reshape(-1))))**2+labda*np.linalg.norm(recon)**2
    return recon, F

# wavelet domain l1 penalized least squares (Forward-Backward splitting <--> ISTA)
def prox_l1(x, l):
    '''function for computing the l1 prox of a vector u'''
    return np.sign(x)*np.maximum(np.abs(x)-l, 0)

def prox_wavelets(vect, dim1, dim2, wave_type, wave_mode, wave_level, gamma, labda):
    '''function that computes the prox of the l1 norm of an image under the wavelets transform '''
    
    coef = pywt.wavedec2(vect.reshape(dim1, dim2), wavelet=wave_type, mode = wave_mode, level=wave_level)
    arr, slices = pywt.coeffs_to_array(coef)
    
    diff = prox_l1(x = arr, l = gamma*labda)
    coeffs = pywt.array_to_coeffs(diff, slices, output_format='wavedec2')
    
    return pywt.waverec2(coeffs, wavelet=wave_type, mode = wave_mode).ravel()

def obj_l1wave(A, u, s, labda, wave, mo, levels):
    '''The objective of the l1-wavelets penalized least squares problem'''
    
    coef = pywt.wavedec2(u, wavelet=wave, mode = mo, level=levels)
    arr, _ = pywt.coeffs_to_array(coef)
    
    return labda*np.abs(arr).sum()+1/2*np.linalg.norm((s-(A@u.reshape(-1))))**2

def FBS_l1wave(A, s, gamma, labda, N1, N2, n_iter, wave='db1', wmode='periodization', wlevel=2, beta=0.01):
    '''Forward-Backward Splitting algorithm using l1-wavelets reg'''
    N = N1*N2
    u_prev = np.zeros(N)
    F = np.zeros(n_iter)
    
    F[0] = obj_l1wave(A=A, s=s, u=u_prev.reshape(N1, N2), 
                      labda=labda, wave=wave, mo=wmode, levels=wlevel)
    
    for i in range(n_iter):
        if not(i%10):
            print('l2-l1 iteration : '+f'{i}/ {n_iter}')
        gm = u_prev - gamma*gradient(A=A, s=s, u=u_prev).ravel()

        u_new = prox_wavelets(vect=gm, dim1=N1, dim2=N2
                        , wave_type=wave, wave_mode=wmode, wave_level=wlevel, gamma=gamma, labda=labda)
        u_prev = u_new
        F[i] = obj_l1wave(A=A, s=s, u=u_prev.reshape(N1, N2), 
                      labda=labda, wave=wave, mo=wmode, levels=wlevel)
    return u_new, F

# TV penalized least squares (Chambolle-Pock primal-dual algorithm)
def grad_col(u):
    '''This function compute the column gradient of a 2D image u'''
    gc = np.zeros(u.shape)
    gc[:, :-1] = u[:, 1:] - u[:, :-1]
    gc[:, -1] = u[:, 0] - u[:, -1]
    return gc

def grad_row(u):
    '''This function compute the row gradient of a 2D image u'''
    gr = np.zeros(u.shape)
    gr[:-1, :] = u[1:, :] - u[:-1, :]
    gr[-1, :] = u[0, :] - u[-1, :]
    return gr

def grad_tot(u):
    '''The total gradient of a 2D image u'''
    N1, N2 = u.shape
    d = u.ndim
    gt = np.zeros((N1, N2, d))
    gt[:, :, 0] = grad_row(u)
    gt[:, :, 1] = grad_col(u)
    return gt

def gradTrans_row(u):
    '''The adjoint of the row gradient'''
    gtr = np.zeros(u.shape)
    gtr[1:, :] = u[:-1, :] - u[1:, :]
    gtr[0, :] = u[-1,:] - u[0,:]
    return gtr

def gradTrans_col(u):
    '''The adjoint of the column gradient'''
    gtc = np.zeros(u.shape)
    gtc[:, 1:] = u[:, :-1] - u[:, 1:]
    gtc[:, 0] = u[:, -1] - u[:, 0]
    return gtc

def gradTrans_tot(grad):
    return gradTrans_row(grad[:, :, 0]) + gradTrans_col(grad[:, :, 1])

def obj_tv(A, u, s, labda, N1, N2):
    '''The Total variation objective'''
    norm = np.linalg.norm
    gradu = grad_tot(u.reshape(N1, N2))
    normu = np.sqrt(gradu[:, :, 0]**2 + gradu[:, :, 1]**2)
    return labda*(normu.sum()) + 0.5*norm(s-(A@u.reshape(-1)))**2

def dual_obj(s, y1):
    '''The dual of the TV objective'''
    norm=np.linalg.norm
    return 0.5*norm(y1)**2 + np.dot(s.ravel(),y1.ravel())

def prox_sigma_f1_star(y1, s, sigma):
    '''The prox of the conjugate of f1 in the augmented 
    TV-regularization reformulation '''
    return (y1 - sigma*s)/(1+sigma)

def prox_sigma_f2_star(labda, beta, y2):
    '''The prox of the conjugate of f2 in the augmented TV-regularization reformulation '''
    normy = np.sqrt(y2[:, :, 0]**2 + y2[:, :, 1]**2)
    dnomi = ((np.maximum((beta/labda)*
                         normy, 1))[:, :, np.newaxis])
    return y2/dnomi

def primal_dual_TV(A, sigma, beta, tau, N1, N2, s, labda, epsilon, n_iter, theta=1):
    '''Chambolle-Pock implementation using the augmented TV-regularization reformulation'''
    #intializing all variables
    N=N1*N2
    x = np.zeros(N)
    xbar = x
    y1 = A@x.reshape(-1)
    y2 = beta*grad_tot(x.reshape(N1, N2))
    F = np.zeros(n_iter)
    Fstar = np.zeros(n_iter)
    F[0] = obj_tv(A=A, u=x, s=s, labda=labda,N1=N1, N2=N2)
    Fstar[0] = dual_obj(s, y1)
    x_prev = x
    for k in range(n_iter):
        #y1 at k+1
        param1 = y1 + sigma*A@xbar.reshape(-1)
        y1 = prox_sigma_f1_star(y1=param1, s=s, sigma=sigma)
        #y2 at k+1
        param2 = y2 + sigma*beta*grad_tot(xbar.reshape(N1, N2))
        y2 = prox_sigma_f2_star(labda = labda, beta=beta, y2=param2)
        #x at k+1
        x_new = (x_prev.reshape(-1) - tau*(A.T@y1 + beta*(gradTrans_tot(y2)).reshape(-1))).reshape(1, -1)
        xbar = x_new + theta*(x_new - x_prev)
        if k<(n_iter-1):
            F[k+1] = obj_tv(A=A, u=x_new, s=s, labda=labda,N1=N1, N2=N2)
            Fstar[k+1] = dual_obj(s, y1)
        x_prev=x_new
        if n_iter<500:
            if not(k%10):
                print('TV iteration : '+f'{k}/ {n_iter}')
        else:
            if not(k%50):
                print('TV iteration : '+f'{k}/ {n_iter}')
        # if np.abs(Fstar[k]-F[k])<epsilon:
        #     break
    return x_new, F, Fstar

# Cauchy penalized least-squares (Forward-Backward splitting)
def obj_cauchy(A, s, u, beta, labda):
    '''The objective of the Cauchy penalized least squares problem'''
    return 0.5*np.linalg.norm(A@u.reshape(-1) - s)**2 - labda*(np.log(beta/(beta**2 + u**2))).sum()

def func_cauchy(u, xi, beta, labda, gamma):
    '''The function f(.)that will be used to compute the proximal operator related to Cauchy prior'''
    return (1/(2*labda*gamma))*(u - xi)**2 - np.log(beta/(beta**2 + u**2))

def solve_cubic(xi, beta, labda, gamma):
    ''' a function that solves the cubic equation:
     ui^3 - xi*ui^2 + (beta^2 + 2*labda*gamma)*ui - xi*beta^2  for ui ''' 
    
    z1 = np.zeros(xi.shape)
    z2 = np.zeros(xi.shape).astype(complex)
    z3 = np.zeros(xi.shape).astype(complex)

    a0 = -xi*beta**2
    a1 = beta**2 + 2*labda*gamma
    a2 = -xi 

    Q = (3*a1 - a2**2)/9
    R = (9*a1*a2 - 27*a0 - 2*a2**3)/54
    D = Q**3 + R**2
    D[np.abs(D)<np.finfo(np.float64).eps] = 0

    mask = (D==0)
    z1[mask] = (-a2[mask]/3) + 2*np.cbrt(R[mask])
    z2[mask] = (-a2[mask]/3).astype(complex) - np.cbrt(R[mask]).astype(complex)
    z3[mask] = (-a2[mask]/3).astype(complex) - np.cbrt(R[mask]).astype(complex)

    #D > 0

    mask = (D > 0)
    S = np.cbrt(R[mask] + np.sqrt(D[mask]))
    T = np.cbrt(R[mask] - np.sqrt(D[mask]))
    z1[mask] = (-a2[mask]/3) + (S + T)
    z2[mask] = ((-a2[mask]/3)).astype(complex) - ((S + T)/2).astype(complex) + (1j*np.sqrt(3)/2)*(S - T) 
    z3[mask] = ((-a2[mask]/3)).astype(complex) - ((S + T)/2).astype(complex) - (1j*np.sqrt(3)/2)*(S - T)

    #D<0

    mask = (D < 0)
    theta = np.arccos(R[mask]/np.sqrt(-Q[mask]**3))
    z1[mask] = (-a2[mask]/3) + 2*np.sqrt(-Q[mask])*np.cos(theta/3) 
    z2[mask] = ((-a2[mask]/3)).astype(complex) + (2*np.sqrt(-Q[mask])*np.cos((theta + 2*np.pi)/3)).astype(complex)
    z3[mask] = ((-a2[mask]/3)).astype(complex) + (2*np.sqrt(-Q[mask])*np.cos((theta + 4*np.pi)/3)).astype(complex)
    
    return z1, z2, z3

def prox_cauchy(xi, beta, labda, gamma):
    '''function that returns the prox operator of the cauchy regularization'''
    
    root1, root2, root3 = solve_cubic(xi = xi, beta = beta, labda = labda, gamma = gamma)
    
    if np.any((np.imag(root2) == 0)):
        mask1 = (np.imag(root2) == 0)
        maskf = ((mask1) & (func_cauchy(u=np.real(root2), xi=xi, beta=beta, labda=labda, gamma=gamma) <  
             func_cauchy(u=root1, xi=xi, beta=beta, labda=labda, gamma=gamma)))
        root1[maskf] = np.real(root2[maskf])
        
    if np.any((np.imag(root3) == 0)):
        mask1 = (np.imag(root3) == 0)
        maskf = ((mask1) & (func_cauchy(u=np.real(root3), xi=xi, beta=beta, labda=labda, gamma=gamma) <  
             func_cauchy(u=root1, xi=xi, beta=beta, labda=labda, gamma=gamma)))
        root1[maskf] = np.real(root3[maskf])
    return root1

def FBS_cauchy(A, s, gamma, labda, N1, N2, n_iter, wave='db1', wmode='periodization', wlevel=2, beta=0.01):
    '''FBS algorithm using for Cauchy Penalty'''
    N = N1*N2
    u_prev = np.zeros(N)
    F = np.zeros(n_iter)
    
    F[0] = obj_cauchy(A=A, s=s, u=u_prev, beta=beta, labda=labda)
    
    for i in range(n_iter):
        if not(i%10):
            print('l2-Cauchy FB iteration : '+f'{i}/ {n_iter}')
        gm = u_prev - gamma*gradient(A=A, s=s, u=u_prev).ravel()
        
        u_new = prox_cauchy(xi=gm, beta=beta, labda=labda, gamma=gamma)
        u_prev = u_new
        F[i]= obj_cauchy(A=A, s=s, u=u_prev, beta=beta, labda=labda)
    return u_new, F

# Cauchy penalized least-squares (BFGS) 
# Cautious BFGS as seen in https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.8382&rep=rep1&type=pdf
# on Page 4 Under the name Algorithm 1
# Also see https://hal.science/hal-03594202 page 10

def Armijo(direc, f, f_val, gfk, x, sig=0.01, tau=0.5, alph=1, maxit=200):
    '''This function implements the armijo rule linesearch for a descent step'''
    k = 0
    
    while True:
        print("Armijo it %d" % (k))
        
        if (((f(x + alph*direc) <= f_val + alph*sig*np.dot(gfk, direc))) or (k > maxit)):
            break
        else:
            alph = tau*alph
            k += 1

    return alph      

def MBFGS(A, s, n_iter, labda, beta, N1, N2, eps, normA):
    '''Implementation of the Modified BFGS as seen in the paper of Dong-Hui Li &
    Masao Fukushima Titled: On the Global Convergence of the BFGS Method For NonConvex
    Unconstrained Optimization Problems 
    (https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.8382&rep=rep1&type=pdf) on Page 4 
    Under the name Algorithm 1
    Also see https://hal.science/hal-03594202 page 10'''
    N = N1*N2
    #initial values
    k = 0
    # u0 = A.T @ s
    # u0 = u0 / np.max(u0)
    u0 = np.zeros(N)
    gradc = lambda x: ((A.T@(A@x - s)) + (2*labda*x)/(beta**2 + x**2))
    objc = lambda x: 0.5*np.sum((A@x - s)**2) - labda*np.sum((np.log(beta/(beta**2 + x**2))))
        
    gk = gradc(x=u0)
    I = np.eye(N)
    Hk = I
    uk = u0
    
    F = np.zeros(n_iter)
        
    while np.linalg.norm(gk) > eps and k < n_iter:
        if not(k%10):
            print('l2-Cauchy BFGS iteration : '+f'{k}/ {n_iter}')
        pk = -Hk@gk
        fk = objc(x=uk)
        F[k] = fk
        # stepk = Armijo(direc=pk, f=objc, f_val=fk, gfk=gk, x=uk)
        
        stepk = scipy.optimize.line_search(f=objc,myfprime=gradc,xk=uk,pk=pk,\
                                            gfk=gk,old_fval=fk,amax=10/(normA**2+2/beta),maxiter=100)
                                        
        uk1 = uk + stepk[0] * pk
        sk = uk1 - uk
        gk1 = gradc(x=uk1)
        yk = gk1 - gk
        if np.linalg.norm(gk) < 1:
            alpha = 3
        else:
            alpha = 0.01
        epsil = 1e-6
        rho = 1/np.dot(yk, sk)     
        
        if (np.dot(yk, sk)/(sk**2).sum()) >= (epsil * np.linalg.norm(gk)**alpha):
            skTyk = np.sum(sk*yk)
            Hkyk = Hk@yk
            HkykskT = np.tensordot(Hkyk, sk,axes=0)
            Hk = Hk + (skTyk + np.sum(yk*Hkyk))/(skTyk**2) * np.tensordot(sk,sk,axes=0) \
                - (HkykskT + HkykskT.T)/(skTyk)
    
        gk = gk1
        uk = uk1
        k += 1
    return uk,F


def LMBFGS(A, s, n_iter, labda, beta, N1, N2, eps, normA):
    '''Implementation of the Limited Memory Modified BFGS as seen in the paper of Dong-Hui Li &
    Masao Fukushima Titled: On the Global Convergence of the BFGS Method For NonConvex
    Unconstrained Optimization Problems 
    (https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.8382&rep=rep1&type=pdf) on Page 4 
    Under the name Algorithm 1
    Also see https://hal.science/hal-03594202 page 10'''
    N = N1*N2
    #initial values
    k = 0
    u0 = np.zeros(N)
    
    L = normA**2 + 2*labda/beta

    
    gradc = lambda x: ((A.T@(A@x - s)) + (2*labda*x)/(beta**2 + x**2))
    objc = lambda x: 0.5*np.sum((A@x - s)**2) - labda*np.sum((np.log(beta/(beta**2 + x**2))))
        
    gk = gradc(x=u0)
    uk = u0
    
    F = np.zeros(n_iter)
    
    list_sk = np.zeros((n_iter,N))
    list_yk = np.zeros((n_iter,N))
           
    while np.linalg.norm(gk) > eps and k < n_iter:
        if not(k%10):
            print('l2-Cauchy BFGS iteration : '+f'{k}/ {n_iter}')
        
        Hk= scipy.optimize.LbfgsInvHessProduct(list_sk[:k,:],list_yk[:k,:])
        pk = -Hk.matvec(gk)
        
        fk = objc(x=uk)
        F[k] = fk
        
        stepk = scipy.optimize.line_search(f=objc,myfprime=gradc,xk=uk,pk=pk,\
                                            gfk=gk,old_fval=fk,amax=10/(L),maxiter=100)
                                        
        uk1 = uk + stepk[0] * pk
        sk = uk1 - uk
        gk1 = gradc(x=uk1)
        yk = gk1 - gk
        if np.linalg.norm(gk) < 1:
            alpha = 3
        else:
            alpha = 0.01
        epsil = 1e-6
                
        if (np.dot(yk, sk)/(sk**2).sum()) >= (epsil * np.linalg.norm(gk)**alpha):
            list_sk[k,:] = sk
            list_yk[k,:] = yk
    
        gk = gk1
        uk = uk1
        k += 1
    return uk,F


def grad_cauchy_grad(x,beta):
    Dx = grad_tot(x)
    nDx2 = np.sum(Dx,axis=2)[:,:,np.newaxis]
    v = gradTrans_tot( 2 * Dx / (nDx2 + beta**2))
    return v

def LMBFGS_grad(A, s, n_iter, labda, beta, N1, N2, eps, normA):
    '''Implementation of the Limited Memory Modified BFGS as seen in the paper of Dong-Hui Li &
    Masao Fukushima Titled: On the Global Convergence of the BFGS Method For NonConvex
    Unconstrained Optimization Problems 
    (https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.8382&rep=rep1&type=pdf) on Page 4 
    Under the name Algorithm 1
    Also see https://hal.science/hal-03594202 page 10'''
    N = N1*N2
    #initial values
    k = 0
    u0 = np.zeros(N)
    
    L = normA**2 + (labda*2)*(4**2/beta**2)
    
    gradc = lambda x: (A.T@(A@x - s)) + labda*grad_cauchy_grad(x.reshape((N1,N2)), beta).reshape(-1)
    objc = lambda x: 0.5*np.sum((A@x - s)**2) - labda*np.sum((np.log(beta/(beta**2 + np.sum(grad_tot(x.reshape((N1,N2)))**2,axis=2)))))
        
    gk = gradc(x=u0)
    uk = u0
    
    F = np.zeros(n_iter)
    
    list_sk = np.zeros((n_iter,N))
    list_yk = np.zeros((n_iter,N))
           
    while np.linalg.norm(gk) > eps and k < n_iter:
        if not(k%10):
            print('l2-Cauchy BFGS iteration : '+f'{k}/ {n_iter}')
        
        Hk= scipy.optimize.LbfgsInvHessProduct(list_sk[:k,:],list_yk[:k,:])
        pk = -Hk.matvec(gk)
        
        fk = objc(x=uk)
        F[k] = fk
        
        stepk = scipy.optimize.line_search(f=objc,myfprime=gradc,xk=uk,pk=pk,\
                                            gfk=gk,old_fval=fk,amax=10/L,maxiter=100)
                                        
        uk1 = uk + stepk[0] * pk
        sk = uk1 - uk
        gk1 = gradc(x=uk1)
        yk = gk1 - gk
        if np.linalg.norm(gk) < 1:
            alpha = 3
        else:
            alpha = 0.01
        epsil = 1e-6
                
        if (np.dot(yk, sk)/(sk**2).sum()) >= (epsil * np.linalg.norm(gk)**alpha):
            list_sk[k,:] = sk
            list_yk[k,:] = yk
    
        gk = gk1
        uk = uk1
        k += 1
    return uk,F