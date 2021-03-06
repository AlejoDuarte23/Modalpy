import BSDTA as BS
import SSI_COV_AD as SSI
import numpy as np 
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.cluster import KMeans
#--------------- Loading Data ------------------------------------------------#
fs = 100
Acc =  np.loadtxt('Acc_trial.txt')
Nd,Nc = Acc[10000:].shape
#---------------------- 1.  Load data ----------------------------------------#
start = timer()
fn,zeta ,phi,fopt,dampopt = SSI.SSI_COV_AD(Acc,fs,10,Nc,100,90)
kmeans = KMeans(n_clusters=6)
kmeans.fit(fopt.reshape(-1,1))
clusters = np.sort(kmeans.cluster_centers_).T
ranges = np.abs(np.diff(clusters))
print(clusters)
end = timer()
print(start-end) 
print('******')
  
start = timer()

# for i in range(clusters.shape[1]-1):
#     fo=clusters[0,i]-ranges[0,i]
#     fi=clusters[0,i]+ranges[0,i]
#     Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)
#     print(Yxx.shape)
#     # opt,C = BS.Modal_id(Yxx,freq_id,Nc,N,clusters[0,i],0.01,fo,fi)
#     opt = [clusters[0,i],0.01,-9,-20]
#     samples=BS.walkers(opt,N,Nc,Yxx,freq_id,70,clusters[0,i])

fo=24
fi=26
Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)
print(Yxx.shape)
# opt,C = BS.Modal_id(Yxx,freq_id,Nc,N,clusters[0,i],0.01,fo,fi)
opt = [24.5,0.001,-9,-20]
samples=BS.walkers(opt,N,Nc,Yxx,freq_id,500,24)

"""
modeshapes needs to eb adjust with NC
fix modal determinsitic - Noise distribuiton
""

    

end = timer()
print(start-end)

# fo = 16#8#8.8#14#16
# fi = 19#10#10#16#19 
# Acc, Nc, Nd = FF.load_Data(fn,1,-1,0,-1)
# fsi = 95
# Ni = int(Acc.shape[0]*fsi/fs)
# from scipy import signal
# Acc = signal.resample(Acc, Ni)
# fn,zeta ,phi,fopt,dampopt = SSI.SSI_COV_AD(Acc,fsi,10,Nc,30,4)
# Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fsi,fo,fi)
# opt,C = BS.Modal_id(Yxx,freq_id,Nc,N,fopt[-1],dampopt[-1],fo,fi)
# samples=BS.walkers(opt,N,Nc,Yxx,freq_id,300)
