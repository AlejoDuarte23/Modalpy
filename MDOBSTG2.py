from scipy.optimize import fmin
import BSDTA as BS
import numpy as np
import matplotlib.pyplot as plt
import metropolis_hasting as MH

#------------------------------- 1. Theorerical Model ------------------------#

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
#------------------------------- 2. Likelyhood ------------------------------#

def MDOF_lkhood(Y,freq_id,x,Nm,N,Nc):
    f = x[:Nm]
    z = x[ Nm:2*Nm]
    S= np.diag(x[ 2*Nm:3*Nm])
    Se=x[-1]    
    Li = 0
    SDKi =  np.abs(np.trace(H(freq_id,f,z,S,Se,Nm,N)))
    for i in range(N):
        sk = Y[i,:].conj().reshape(Nc,1)*Y[i,:]
        ti = 2*(SDKi[i]**2+2*(10.**Se)*SDKi[i]+(10.**Se)**2)
        L = np.log(np.pi*ti)+(np.trace(sk)-(SDKi[i]+(10.**Se)))**2/ti
        Li = Li +L
    return Li

#------------------------------- 3. Model ------------------------------------#
def Model(freq_id,x,Nm,N,Nc):
    f = x[:Nm]
    z = x[ Nm:2*Nm]
    S= np.diag(x[ 2*Nm:3*Nm])
    Se=x[-1]    
    Li = 0
    SDKi=  np.abs(np.trace(H(freq_id,f,z,S,Se,Nm,N)))
    return SDKi
#------------------------------- 4. Map  -------------------------------------#
def ID_BSTA_MDOF(Yxx,freq_id,x,Nm,N,Nc,PL=True):
    likelyhood = lambda x:MDOF_lkhood(Yxx,freq_id,x,Nm,N,Nc)
    opt = fmin(func=likelyhood ,x0=xo,maxiter= 5000,xtol=1e-12, ftol=1e-12)
    if PL == True:
        modelo  = []
        for i in range(N):
            sk = Yxx[i,:].conj().reshape(Nc,1)*Yxx[i,:]
            modelo = np.abs(np.append(modelo,np.trace(sk)/Nc))
        SDKi = Model(freq_id,opt,Nm,N,Nc)
        plt.figure()
        plt.plot(freq_id,10*np.log10(SDKi))
        plt.plot(freq_id,10*np.log10(modelo))
    return opt


#------------------------------- Data ----------------------------------------#
fs = 95
Acco =  np.loadtxt('Data/Sp_Acc_11_02_12_7_18.txt',delimiter = ',')
Acc = Acco[:,:3]
Nd,Nc = Acc.shape
fo = 3
fi = 4.2
#------------------------------- t2frequencydomain----------------------------#
Yxx,freq_id,N = BS.PSD_FORMAT(Acc,fs,fo,fi)
# Target Frequencies
f=[3.72,3.89]
Nm = 2
z = [ 5.0e-03, 5.0e-03]
S = [-6,-6]
Se = [-5]
xo = [*f,*z,*S,*Se]
#------------------------------- Engine --------------------------------------#
ID_BSTA_MDOF(Yxx,freq_id,xo,Nm,N,Nc,PL=True)



std_tr = [0.001,0.001,0.00005,0.00005,0.1,0.1,0.1]
tr_model = lambda x: MH.transition_model(x,std_tr)

pri_lim = [[3.6,3.8],[3.7,4],[0,0.05],[0,0.05],[-10,0],[-10,0],[-20,-1]]
pri_model = lambda x: MH.prior(x,pri_lim)

lik= lambda x,y: -MDOF_lkhood(y,freq_id,x,Nm,N,Nc)
accepted,rejected,prob = MH.metropolis_hastings(lik,pri_model, tr_model,xo,3000,Yxx)


# plt.plot(accepted)
opt2 = np.mean(accepted,axis =0)
SDKi = Model(freq_id,opt2,Nm,N,Nc)
plt.plot(freq_id,10*np.log10(SDKi))
plt.plot(freq_id,10*np.log10(Yxx**2))