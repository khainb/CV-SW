import numpy as np
import torch
from torch.autograd import Variable
import ot

def compute_true_Wasserstein(X,Y,p=2):
    M = ot.dist(X.detach().numpy(), Y.detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)
def compute_Wasserstein(M,device='cpu',e=0):
    if(e==0):
        pi = ot.emd([],[],M.cpu().detach().numpy()).astype('float32')
    else:
        pi = ot.sinkhorn([], [], M.cpu().detach().numpy(),reg=e).astype('float32')
    pi = torch.from_numpy(pi).to(device)
    return torch.sum(pi*M)

def rand_projections(dim, num_projections=1000,device='cpu'):
    projections = torch.randn((num_projections, dim),device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections


def one_dimensional_Wasserstein_prod(X,Y,theta,p):
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    wasserstein_distance = torch.mean (torch.pow(wasserstein_distance, p), dim=0,keepdim=True)
    return wasserstein_distance



def SW(X, Y, L=10, p=2, device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L,device)
    sw=one_dimensional_Wasserstein_prod(X,Y,theta,p=p).mean()
    return  torch.pow(sw,1./p)


def LCVSW(X,Y,L=10,p=2,device="cpu"):
    dim = X.size(1)
    m_1 = torch.mean(X,dim=0)
    m_2 = torch.mean(Y,dim=0)
    diff_m1_m2= m_1-m_2
    G_mean = torch.mean((diff_m1_m2)**2) #+ (sigma_1-sigma_2)**2
    theta = rand_projections(dim, L, device)
    hat_G = torch.sum(theta*(diff_m1_m2),dim=1)**2 #+(sigma_1-sigma_2)**2
    diff_hat_G_mean_G = hat_G - G_mean
    hat_sigma_G_square = torch.mean((diff_hat_G_mean_G)**2)
    distances = one_dimensional_Wasserstein_prod(X,Y,theta,p=p)
    hat_A = distances.mean()
    hat_C_G = torch.mean((distances-hat_A)*(diff_hat_G_mean_G))
    hat_alpha = hat_C_G/(hat_sigma_G_square+1e-24)
    Z = hat_A - hat_alpha*torch.mean(diff_hat_G_mean_G)
    return torch.pow(torch.mean(Z),1./p)

def UCVSW(X,Y,L=10,p=2,device="cpu"):
    dim = X.size(1)
    m_1 = torch.mean(X,dim=0)
    m_2 = torch.mean(Y,dim=0)
    diff_m1_m2= m_1-m_2
    diff_X_m1 = X-m_1
    diff_Y_m2 = Y-m_2
    G_mean = torch.mean((diff_m1_m2)**2) +  torch.mean((diff_X_m1)**2)+  torch.mean((diff_Y_m2)**2)
    theta = rand_projections(dim, L, device)
    hat_G = torch.sum(theta*(diff_m1_m2),dim=1)**2 +torch.mean(torch.matmul(theta,diff_X_m1.transpose(0,1))**2,dim=1)+torch.mean(torch.matmul(theta,diff_Y_m2.transpose(0,1))**2,dim=1)
    diff_hat_G_mean_G = hat_G - G_mean
    hat_sigma_G_square = torch.mean((diff_hat_G_mean_G)**2)
    distances = one_dimensional_Wasserstein_prod(X,Y,theta,p=p)
    hat_A = distances.mean()
    hat_C_G = torch.mean((distances-hat_A)*(diff_hat_G_mean_G))
    hat_alpha = hat_C_G/(hat_sigma_G_square+1e-24)
    Z = hat_A - hat_alpha*torch.mean(diff_hat_G_mean_G)
    return torch.pow(torch.mean(Z),1./p)




