import BSDTA as BS
import SSI_COV_AD as SSI
import numpy as np 
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.cluster import KMeans
from tqdm import tqdm
import MDOF_LSQ as LSQ
#--------------- Loading Data ------------------------------------------------#
fs = 95
Ncl = 5
Lk_dist = 0.2
Acco =  np.loadtxt('Data/Sp_Acc_11_02_12_7_18.txt',delimiter = ',')
# Acc = Acco[:,3:6]
Acc = Acco[:,-6:]

Nd,Nc = Acc.shape
#---------------------- 1.  Load data ----------------------------------------#

# Yxx,freq_id,N =BS.PSD_FORMAT(Acc,fs,1,20,PL=True)


cases = 6
if cases == 0:
    fn,zeta ,phi,fopt,dampopt = SSI.SSI_COV_AD(Acc,fs,6,Nc,40,35,Ncl,Lk_dist)
 
if cases == 1:
    Optc,clusters,ranges = BS.Optimal_Cluster(fopt)
    BS.Det_Unc_OMA(fopt,dampopt,phi,Acc,fs,Nc,ranges,clusters)

if cases == 2:
    #Inital:
    fo = 10
    fi = 10.8
    phi= [0.1,0.1,0.1,0.9,0.9,0.9,0.1,0.1,0.1]

    Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)
    tetha = [10.5,0.001,-8,-5]
    Samples = BS.walkers(tetha,phi,fo,fi,N,Nc,Yxx,freq_id,3000)
if cases == 3:
    Nm = 3
    fo = 3.5
    fi = 5.8
    f = [3.88,4.8,5.6]
    z = [0.001,0.001,0.001]
    S = [-7,-8,-8]
    Se = [-10]
    xo = [*f,*z,*S,*Se]
    xopt,psd = LSQ.MDOF_LSQ(xo,Acc,fs,fo,fi,Nm)

if cases == 5:
    #Inital:
    fo = 3.6
    fi = 4.3
    phi= [1,1,1]

    Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)
    tetha = [3.88,0.004,-5,-6]
    Samples = BS.walkers(tetha,phi,fo,fi,N,Nc,Yxx,freq_id,2000)

if cases == 6:
    
    fo = 17
    fi = 18
    
    f=[17.11,17.25,17.67,17.89,18.11,18.2,18.3]
    # f = [17.1,17.24,17.57,17.67,17.8,17.9,18.1]
    z = [0.0001,0.0001,0.0001,0.0001,0.01,0.01,0.01]
    S = [-6,-6,-6,-6,-5,-5,-5]
    Se = [-15]
    xo = [*f,*z,*S,*Se]
    Nm = len(f)
    xopt,psd = LSQ.MDOF_LSQ(xo,Acc,fs,fo,fi,Nm)


##Inital:
#fo = 10
#fi = 10.8
#phi= [0.1,0.1,0.1,0.9,0.9,0.9,0.1,0.1,0.1]
#
#Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)
#tetha = [10.5,0.001,-8,-5]
#Samples = BS.walkers(tetha,phi,fo,fi,N,Nc,Yxx,freq_id,3000)
#    
    
##Inital:
#fo = 5.3
#fi = 6
#phi= [0.9,0.9,0.9,0.1,0.1,0.1,0.1,0.1,0.1]
#
#Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)
#tetha = [6.6,0.001,-6,-6]
#Samples = BS.walkers(tetha,phi,fo,fi,N,Nc,Yxx,freq_id,1000)
    
#fo = 3.6
#fi = 3.78
#phi= [0.9,0.9,0.9,0.1,0.1,0.1,0.1,0.1,0.1]
#
#Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)
#tetha = [3.6,0.0001,-6,-2]
#Samples = BS.walkers(tetha,phi,fo,fi,N,Nc,Yxx,freq_id,1000)        

#fo = 17
#fi = 19 
#phi= [0.01,0.01,0.01,0.1,0.1,0.1,0.9,0.1,0.9]
#
#Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)
#tetha = [17.5,0.001,-6,-5]
#Samples = BS.walkers(tetha,phi,fo,fi,N,Nc,Yxx,freq_id,2000)
#    
#    
#
#fo = 8.2
#fi = 9
#phi= [0,0,0,0.2,0.2,0.2,0.9,0.9,0.9]
#
#Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)
#tetha = [8.73,0.01,-6,-5]
#Samples = BS.walkers(tetha,phi,fo,fi,N,Nc,Yxx,freq_id,1500)

               
# i = 0
# fo=clusters[0,i]-ranges[0,i]/2
# fi=clusters[0,i]+ranges[0,i]/2
# Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)y
# plt.plot(freq_id,10*np.log10(Yxx))

 
 
# for i in range(clusters.shape[1]-7): 
#     fo=clusters[0,i]-ranges[0,i]/2
#     fi=clusters[0,i]+ranges[0,i]/2
#     Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)
#     # opt,C = BS.Modal_id(Yxx,freq_id,Nc,N,clusters[0,i],0.01,fo,fi)
#     opt = [clusters[0,i],0.01,-9,-10]
#     samples=BS.walkers(opt,N,Nc,Yxx,freq_id,3000)
#
#fo=24
#fi=26
#Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)
#print(Yxx.shape)
#opt,C = BS.Modal_id(Yxx,freq_id,Nc,N,clusters[0,1],0.01,fo,fi)
#opt = [24.5,0.001,-9,-20]
#samples=BS.walkers(opt,N,Nc,Yxx,freq_id,500)
#
#end = timer()
#print(start-end)

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
