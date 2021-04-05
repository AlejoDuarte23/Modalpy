from scipy.optimize import fmin
import BSDTA as BS
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')


def H(freq,f,z,S,Se,Nm,N):
    
    H = np.zeros((Nm,Nm,N),dtype=np.complex_)
    for i in range(0,Nm):
        for j in range(0,Nm): 
            if i==j:
                bki = f[i]/freq
                bkj = f[j]/freq
                ter1 = 1/((1-bki)+2j*z[i]*bki)
                ter2 = 1/((1-bkj)+2j*z[j]*bkj)
                H[i,j,:] = (10.**S[i,j])*(ter1)*(ter2)+10.**Se
    return H


# def like(Y,freq_id,f,z,S,Se,phi,Nm,N,Nc): 
#         Se = 10**Se
#         S  = 10**S
#         Li = 0
#         SDKi =  np.trace(H(freq_id,f,z,S,Se,Nm,N))
#         for i in range(N):
            
#             sk = Y[i,:].conj().reshape(Nc,1)*Y[i,:]
#             print(sk)
#             ti = 2*(SDKi[i]**2+2*Se*SDKi[i]+Se**2)
#             L = np.log(np.pi*ti)+(np.trace(sk)-(SDKi[i]+Se))**2/ti
#             Li = Li +L
#             print(Li)
#         return Li

def new_lik(Y,freq_id,x,Nm,N,Nc):
    # breakpoint()
    f = x[:Nm]
    z = x[ Nm:2*Nm]
    S= np.diag(x[ 2*Nm:3*Nm])
    Se=x[-1]    
    Li = 0
    SDKi =  np.abs(np.trace(H(freq_id,f,z,S,Se,Nm,N)))
    # Ho =  H(freq_id,f,z,S,Se,Nm,N)
    for i in range(N):
        sk = Y[i,:].conj().reshape(Nc,1)*Y[i,:]
        ti = 2*(SDKi[i]**2+2*(10.**Se)*SDKi[i]+(10.**Se)**2)
        L = np.log(np.pi*ti)+(np.trace(sk)-(SDKi[i]+(10.**Se)))**2/ti
        Li = Li +L
    return Li

def Modes(freq_id,x,Nm,N,Nc):
    # breakpoint()
    f = x[:Nm]
    z = x[ Nm:2*Nm]
    S= np.diag(x[ 2*Nm:3*Nm])
    Se=x[-1]    
    Li = 0
    SDKi=  np.abs(np.trace(H(freq_id,f,z,S,Se,Nm,N)))
    return SDKi

#------------------------------- Data ----------------------------------------#
fs = 95
Acco =  np.loadtxt('Data/Sp_Acc_11_02_12_7_18.txt',delimiter = ',')
Acc = Acco[:,:3]
Nd,Nc = Acc.shape
fo = 3
fi = 4.2
Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)
f=[3.72,3.89]
Nm = 2
z = [ 5.0e-03, 5.0e-03]
S = [-6,-6]
Se = [-5]
xo = [*f,*z,*S,*Se]

SDKi = Modes(freq_id,xo,Nm,N,Nc)
plt.figure()
plt.plot(freq_id,10*np.log10(SDKi))
plt.plot(freq_id,10*np.log10(Yxx))
likelyhood = lambda x:new_lik(Yxx,freq_id,x,Nm,N,Nc)
#---------------- Optimize likelyhood ----------------#

# opt = fmin(func=likelyhood ,x0=xo)
opt = fmin(func=likelyhood ,x0=xo,maxiter= 5000,xtol=1e-12, ftol=1e-12)
SDKi = Modes(freq_id,opt,Nm,N,Nc)

modelo = []
for i in range(N):
    sk = Yxx[i,:].conj().reshape(Nc,1)*Yxx[i,:]
    modelo = np.abs(np.append(modelo,np.trace(sk)/Nc))
        
        
plt.figure()
plt.plot(freq_id,10*np.log10(SDKi),color = 'red')
plt.plot(freq_id,10*np.log10(modelo),color= 'k')




