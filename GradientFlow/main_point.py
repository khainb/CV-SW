##############################################
# Setup
# ---------------------
import random
import time
from utils import *
import numpy as np
import torch
for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)
A = np.load("reconstruct_random_50_shapenetcore55.npy")
#32 33 and  2 42
ind1=32
ind2=33
target=A[ind2]*2+10
source=A[ind1]*2-10
seed=1
device='cpu'
f_type='exp'
learning_rate = 0.01
N_step=8000
eps=0
L=100
print_steps = [0,2999,3999,4999,5999,7999]
Y = torch.from_numpy(target)
N=target.shape[0]
# copy=False
X=torch.tensor(source)
Xdetach = X.detach()
Ydetach = Y.detach()
vars=[]
for L in [10]:
    for seed in [1]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("SW {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)

            optimizer.zero_grad()
            sw= SW(X,Y,L=L,p=2)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().data.numpy())
        np.save("saved/SW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/SW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/SW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")




for L in [10]:
    for seed in [1]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("LCSW {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)

            optimizer.zero_grad()
            sw= LCVSW(X,Y,L=L)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().data.numpy())
        np.save("saved/LCVSW_L{}_{}_{}_points_seed{}.npy".format(L, ind1, ind2, seed), np.stack(points))
        np.savetxt("saved/LCVSW_L{}_{}_{}_distances_seed{}.txt".format(L, ind1, ind2, seed), np.array(distances),
                   delimiter=",")
        np.savetxt("saved/LCVSW_L{}_{}_{}_times_seed{}.txt".format(L, ind1, ind2, seed), np.array(caltimes), delimiter=",")

for L in [10]:
    for seed in [1]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("UCVSW {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)

            optimizer.zero_grad()
            sw= UCVSW(X,Y,L=L)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().data.numpy())
        np.save("saved/UCVSW_L{}_{}_{}_points_seed{}.npy".format(L, ind1, ind2, seed), np.stack(points))
        np.savetxt("saved/UCVSW_L{}_{}_{}_distances_seed{}.txt".format(L, ind1, ind2, seed), np.array(distances),
                   delimiter=",")
        np.savetxt("saved/UCVSW_L{}_{}_{}_times_seed{}.txt".format(L, ind1, ind2, seed), np.array(caltimes), delimiter=",")



