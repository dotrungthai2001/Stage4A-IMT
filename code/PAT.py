# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:19:08 2022

File containing all the necessary functions to construct the direct model

@authors: Caroline, Paul
"""

import numpy as np
import scipy.sparse as sps

# definition de differents capteurs. Par convention, le centre du capteur est (0,0)
# la discretisation du capteur est stockée sous la forme d'un tableau de taille (d,N)
# ou d est la dimension du probleme (2 ou 3) et N est le nombre de points 

# capteur isotrope: un point en (0,0)
def capteur_iso():
    x = np.zeros((2,1))
    return x

# capteur en ligne : N points decrivant l'intervalle (-L,L,N) x {0}
def capteur_seg(L,N):
    x = np.zeros((2,N))
    x[0,:] = np.linspace(-L,L,N) 
    return x 
    
# capteur en arc de cercle : N points decrivant l'arc de cercle de rayon R
# et d'angle [-thetaMax,thetaMax]
def capteur_cercle(R, thetaMax,N):
    x = np.zeros((2,N))
    theta = np.linspace(-thetaMax,thetaMax,N)

    x[0,:] = R*np.cos(theta)
    x[1,:] = R*np.sin(theta)
    
    x = x - np.expand_dims(np.mean(x, axis=1),axis=1)
    return x
    

# translation et rotation des capteurs correspondant au systeme de jerome
# INPUT     x_capteur : discretisation du capteur
#           alphaMax : rotations du capteur dans (-alphaMax,alphaMax)
#           nalpha: nbr de rotations
#           txMax,tyMax : rotations dans [-txMax, txMax] x [-tyMax,tyMax]
#           nT : nbr de translation             
def systeme_jerome(x_capteur, alphaMax, nalpha, txMax, tyMax, nT):
    alpha = np.linspace(-alphaMax, alphaMax, nalpha)
    trans = np.zeros((2,nT))
    trans[0,:] = np.linspace(-txMax,txMax,nT)
    trans[1,:] = np.linspace(-tyMax,tyMax,nT)
    
    capteurs = np.zeros((x_capteur.shape[0],x_capteur.shape[1],nalpha*nT))
    
    idx = 0
    for i in range(nalpha):
        rotMat = np.array(((np.cos(alpha[i]), -np.sin(alpha[i])), (np.sin(alpha[i]), np.cos(alpha[i]))))
        rotatedSensor = rotMat @ x_capteur 
        
        for j in range(nT):
            capteurs[:,:,idx] = rotatedSensor - np.expand_dims(trans[:,j], axis=1)
            idx = idx + 1
            
    return capteurs 
            
            
        

# calcul de la matrice de tomographie photoaccoustique
# INPUT     capteurs : liste des capteurs de taille (2,Npt,Ncapteurs)
#           L : grille [-L/2,L/2] x [-L/2,L/2]
#           N : nbr de pts dans chaque direction
#           c : vitesse du son
#           Fs : frequence echantillonnage du capteur
#           tstart, tend : starting time and end time of the recording
def PAT(capteurs, L,N,c,Fs,tstart,tend):
    NN = N*N
    Dxy = L / (N-1)
    dt = 1/Fs
    time= np.arange(tstart,tend,dt) 
    timePdt = time + dt
    time = time - dt
    Nt = len(time)
    nrows = Nt*capteurs.shape[2]
    n_angles = 2*N
    
    A_mat = sps.csr_matrix(([], ([], [])), shape=(nrows, NN))
    
    dist_sensor = c*time
    dist_pdd_sensor = c*timePdt
    
    for i in range(capteurs.shape[2]):
        for j in range(capteurs.shape[1]):
        
            X = capteurs[:,j,i] 
            normX = np.sqrt(X[0]**2 + X[1]**2)
            
            angle_max = np.arcsin( (L + 2*Dxy)*np.sqrt(2) / (2 * normX) )
            k_1 = np.linspace(-angle_max,angle_max,n_angles) 
            angles = k_1.reshape(-1,1) @ np.ones((1, len(time)))
            
            x_pt = normX-(np.ones((n_angles, 1))*dist_sensor)*np.cos(angles)
            y_pt = (np.ones((n_angles, 1))*dist_sensor)*np.sin(angles)
            R_pt = np.ones((n_angles, 1))*dist_sensor
            
            xpdx_pt = normX-(np.ones((n_angles, 1))*dist_pdd_sensor)*np.cos(angles)
            ypdy_pt = (np.ones((n_angles, 1))*dist_pdd_sensor)*np.sin(angles)
            RpdR_pt = np.ones((n_angles, 1))*dist_pdd_sensor
    
            A_mat = A_mat + (1/(2*dt)) * (
                        -mat_proj(N, x_pt, y_pt, R_pt, X[0], X[1], L, nrows, i+1)\
                        +mat_proj(N,xpdx_pt,ypdy_pt,RpdR_pt,X[0], X[1], L, nrows, i+1)
                        )
        print("Sensor %d/ %d \n" % (i,capteurs.shape[2]))
        
    return A_mat
        

def mat_proj(N,x_pt,y_pt,R_pt,X,Z,image_width,n_rows,proj):
    """
    Get I, from equation (12) of the paper "Acceleration of Optoacoustic Model-Based
    Reconstruction Using Angular Image Discretization"

    Parameters
    ----------
    x_pt : TYPE
        DESCRIPTION.
    y_pt : TYPE
        DESCRIPTION.
    R_pt : TYPE
        DESCRIPTION.
    theta : TYPE 
        DESCRIPTION.
    image_width : TYPE
        DESCRIPTION.
    n_rows : TYPE
        DESCRIPTION.
    proj : TYPE
        DESCRIPTION.

    Returns
    -------
    A_mat_p : TYPE
        I.

    """
    nn = N*N    #number of columns of the matrix
    lt = np.size(x_pt,1)  #length of the time vector
    n_angles = np.size(x_pt,0)  #number of points of the curve
    Dxy = image_width/(N-1)   # sampling distance in x and y
    
    valeur_cos = X/np.sqrt(X**2 + Z**2)
    valeur_sin = Z/np.sqrt(X**2 + Z**2)
    

    x_pt_unrot = x_pt*valeur_cos - y_pt*valeur_sin  # horizontal position of the points of the curve in the original grid (not rotated)
    y_pt_unrot = x_pt*valeur_sin+ y_pt*valeur_cos  #vertical position of the points of the curve in the original grid (not rotated)
    
    # print(x_pt_unrot.shape)
    # print(n_angles)
    # x_aux_1 = np.zeros((2,lt))
    # x_aux_1[0,:] = xp_pt_unrot
    # y_aux_1 = np.zeros((2,lt))
    # y_aux_1[0,:] = yp_pt_unrot

    # x_aux_2 = np.zeros((2,lt))
    # y_aux_2 = np.zeros((2,lt))

    d_pt = np.sqrt( (x_pt_unrot[1:,:] - x_pt_unrot[:-1,:])**2 +  (y_pt_unrot[1:,:] - y_pt_unrot[:-1,:])**2 )
    
    
    # x_aux_1 =  np.vstack((x_pt_unrot, np.zeros((1,lt))))
    # y_aux_1 =  np.vstack((y_pt_unrot, np.zeros((1,lt))))
    # x_aux_2 =  np.vstack((np.zeros((1,lt)), x_pt_unrot))
    # y_aux_2 =  np.vstack((np.zeros((1,lt)), y_pt_unrot))
    # dist_aux = np.sqrt((x_aux_1-x_aux_2)**2+(y_aux_1-y_aux_2)**2)
    
    # d_pt = dist_aux[1:n_angles,:]  # length of the segments of the curve
    
    vec_int = (1/2)*(np.vstack((d_pt, np.zeros((1,lt)))) + np.vstack((np.zeros((1,lt)), d_pt)))/ R_pt  # vector for calculating the integral
    x_pt_pos_aux = (x_pt_unrot+(image_width/2))/Dxy+1  # horizontal position of the points of the curve in normalized coordinates
    y_pt_pos_aux = (y_pt_unrot+(image_width/2))/Dxy+1  # vertical position of the points of the curve in normalized coordinates
    
    
    
    x_pt_pos_bef = np.floor(x_pt_pos_aux)  # horizontal position of the point of the grid at the left of the point (normalized coordinates)
    x_pt_pos_aft = np.floor(x_pt_pos_aux+1)  # horizontal position of the point of the grid at the right of the point (normalized coordinates)
    x_pt_dif_bef = x_pt_pos_aux-x_pt_pos_bef
    
    y_pt_pos_bef = np.floor(y_pt_pos_aux)  # vertical position of the point of the grid below of the point (normalized coordinates)
    y_pt_pos_aft = np.floor(y_pt_pos_aux+1)  # vertical position of the point of the grid above of the point (normalized coordinates)
    y_pt_dif_bef = y_pt_pos_aux-y_pt_pos_bef
    
    
    in_pos_sq_1, Pos_sq_1_t_vec = pos_making(N,x_pt_pos_bef, y_pt_pos_bef,
                                             lt, n_angles)
    in_pos_sq_2, Pos_sq_2_t_vec = pos_making(N,x_pt_pos_aft, y_pt_pos_bef,
                                             lt, n_angles)
    in_pos_sq_3, Pos_sq_3_t_vec = pos_making(N,x_pt_pos_bef, y_pt_pos_aft,
                                             lt, n_angles)
    in_pos_sq_4, Pos_sq_4_t_vec = pos_making(N,x_pt_pos_aft, y_pt_pos_aft,
                                             lt, n_angles)
    
    
    weight_sq_1 = (1-x_pt_dif_bef)*(1-y_pt_dif_bef)*vec_int  # weight of the first point of the triangle
    weight_sq_2 = (x_pt_dif_bef)*(1-y_pt_dif_bef)*vec_int  # weight of the second point of the triangle
    weight_sq_3 = (1-x_pt_dif_bef)*(y_pt_dif_bef)*vec_int  # weight of the third point of the triangle
    weight_sq_4 = (x_pt_dif_bef)*(y_pt_dif_bef)*vec_int  # weight of the fourth point of the triangle
    weight_sq_1_t_vec = weight_sq_1.reshape(1,n_angles*lt)#, order ='F')  # weight_sq_1 in vector form
    weight_sq_2_t_vec = weight_sq_2.reshape(1,n_angles*lt)#, order ='F')  # weight_sq_1 in vector form
    weight_sq_3_t_vec = weight_sq_3.reshape(1,n_angles*lt)#, order ='F')  # weight_sq_1 in vector form
    weight_sq_4_t_vec = weight_sq_4.reshape(1,n_angles*lt)#, order ='F')  # weight_sq_1 in vector form
    
    k = np.linspace(1,lt,lt)
    Row_Matrix = (k.reshape(-1,1) @ np.ones((1,n_angles))).T  # rows of the sparse matrix
    Row_Matrix_vec = Row_Matrix.reshape(1, n_angles*lt)#, order ='F') # rows of the sparse matrix in vector form
    
    
    V = np.hstack((weight_sq_1_t_vec[in_pos_sq_1],weight_sq_2_t_vec[in_pos_sq_2],
                   weight_sq_3_t_vec[in_pos_sq_3],weight_sq_4_t_vec[in_pos_sq_4]))
    
    I = np.hstack((Row_Matrix_vec[in_pos_sq_1]+((proj-1)*lt),
                   Row_Matrix_vec[in_pos_sq_2]+((proj-1)*lt),
                   Row_Matrix_vec[in_pos_sq_3]+((proj-1)*lt),
                   Row_Matrix_vec[in_pos_sq_4]+((proj-1)*lt)))
    # I = I.astype(int)
    
    J = np.hstack((Pos_sq_1_t_vec[in_pos_sq_1],Pos_sq_2_t_vec[in_pos_sq_2],\
                   Pos_sq_3_t_vec[in_pos_sq_3],Pos_sq_4_t_vec[in_pos_sq_4]))
    # J = J.astype(int)
        
    A_mat_p = sps.csc_matrix((V, (I-1, J-1)), shape=(n_rows,nn))
    # A_mat_p = sps.csr_matrix(([], ([], [])), shape=(n_rows, nn))

    return A_mat_p


def pos_making(N, x_pos, y_pos, lt, n_angles):
   """
   Generate the positions of the squares inside the grid

   Parameters
   ----------
   x_pos : TYPE
       horizontal position of the points.
   y_pos : TYPE
       vertical position of the points.
   lt : TYPE
       length of the time vector.
   n_angles : TYPE
       number of points of the curve.

   Returns
   -------
   in_pos_sq : TYPE
       boolean determining if the given points of the square is inside the grid.
   Pos_sq_t_vec : TYPE
       one dimensional position of the given points of the squares in the grid.

   """
   pos_sq_x = x_pos
   pos_sq_y = y_pos
   
   in_pos_sq = (pos_sq_x>0)&(pos_sq_x<=N)&(pos_sq_y>0)&(pos_sq_y<=N)
   in_pos_sq = in_pos_sq.reshape(1,-1) #, order = 'F')
   
   Pos_sq_t = N*(pos_sq_x-1)+pos_sq_y
   
   Pos_sq_t_vec = Pos_sq_t.reshape(1,n_angles*lt) #, order ='F')
   
   return in_pos_sq, Pos_sq_t_vec
    
