import torch
import numpy as np
from FFDNET import FFDNet
from ckpt_manager import CheckpointManager
from helper_functions import read_transparent_png
import matplotlib.pyplot as plt
from tools import *
from scipy.sparse.linalg import svds
import scipy
from error_measures import SNR




save_path = './'
checkpoint_dir = save_path + 'checkpoints/'
dtype = torch.float32
torch.set_default_dtype(dtype )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
torch.cuda.empty_cache()

model = FFDNet(num_input_channels=1).to(device)
model.load_state_dict(torch.load(checkpoint_dir + 'model_83730.pt', map_location=torch.device('cpu'))["model"])
for p in model.parameters():
    p.requires_grad = True
# model.eval()
# with torch.no_grad():
    
    

u = read_transparent_png('vaisseau_photo_3.png')
u = 255 - np.mean(u, axis=2)
u = u[:-1,:-1]
# u = torch.tensor(u, dtype = dtype).to(device)[None, None] / 255
u = np.array(u,dtype=np.float32)

A = load_sparse_csr('Afwd')
normA = svds(A, k=1, return_singular_vectors = False)[0]
A = A / normA 
L = 1
s = A@u.reshape(-1)


def FBS_PnP(A, s, denoiser, gamma, labda, N1, N2, n_iter,u_exact):
    '''Forward-Backward Splitting algorithm using l1-wavelets reg'''
    N = N1*N2
    u_prev = A.T@s
    
    ## init tikhonov
    matvec = lambda x : A.T@(A @ x) + labda * x 
    linATA = scipy.sparse.linalg.LinearOperator(shape = (N,N), matvec = matvec, rmatvec = matvec)
    x, info = scipy.sparse.linalg.cg(linATA, u_prev, maxiter = 20)
    u_prev = x 
    
    for i in range(n_iter):
        gm = u_prev - gamma*(A.T@(A@u_prev - s))

        gm_t = torch.tensor(gm, dtype=dtype).to(device).reshape((N1,N2))[None,None]
        min_gm = torch.min(gm_t)
        max_gm = torch.max(gm_t)
        # print(min_gm)
        # print(max_gm)
        gm_t = (gm_t - min_gm) / (max_gm - min_gm)
        with torch.no_grad():
            u_new_t = denoiser(gm_t,torch.sqrt(torch.tensor(labda*gamma, device=device, dtype=dtype)))
            u_new_t = min_gm + (max_gm - min_gm)*gm_t
        u_new = u_new_t[0,0].reshape(-1).cpu().numpy()
        
        u_prev = u_new
        print('%d / %d -- SNR = %f' %(i,n_iter, SNR(u_exact,u_new.reshape((N1,N2)))))
    return u_new

def ADMM(A, s, denoiser, gamma, labda, N1, N2, n_iter,u_exact):
    
    N = N1*N2
    ATs = A.T@s
    x = ATs
    y = x 
    ## init tikhonov
    matvec = lambda x : A.T@(A @ x) + gamma * x 
    linATA = scipy.sparse.linalg.LinearOperator(shape = (N,N), matvec = matvec, rmatvec = matvec)
    # x, info = scipy.sparse.linalg.cg(linATA, ATs, maxiter = 20)
    mu = np.zeros(N)
    
    for i in range(n_iter):
        tmp = ATs + mu + gamma * y
        x, info = scipy.sparse.linalg.cg(linATA, tmp, maxiter = 20)
        
        tmtpt = torch.tensor((x - mu/gamma).reshape((N1,N2)),device=device,dtype=dtype)[None,None]
        with torch.no_grad():
            y_t = denoiser( tmtpt, torch.sqrt(torch.tensor(labda/gamma, dtype=dtype).to(device)))
            y = y_t[0,0].reshape(-1).cpu().numpy()
        
        mu = mu + gamma*(y - x)
        print('%d / %d -- SNR = %f' %(i,n_iter, SNR(u_exact,x.reshape((N1,N2)))))

    
    return x
    
    

n_iter = 100
u_FBPnP = FBS_PnP(A,s,model,L,0.0001,u.shape[0],u.shape[1],n_iter,u).reshape((u.shape[0],u.shape[1]))
u_ADMM = ADMM(A,s,model,1e-2,0.0001,u.shape[0],u.shape[1],n_iter,u).reshape((u.shape[0],u.shape[1]))

print('SNR FB PnP: '+f'{SNR(u,u_FBPnP)} dB')
print('####')

# show the results
plt.figure
plt.imshow(u_FBPnP)
plt.title('FB PnP')
plt.show()

print('SNR ADMM PnP: '+f'{SNR(u,u_ADMM)} dB')
print('####')

# show the results
plt.figure
plt.imshow(u_ADMM)
plt.title('ADMM')
plt.show()
