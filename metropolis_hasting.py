import numpy as np
from scipy import stats
from tqdm import tqdm




def transition_model(x,std):
    """
    Parameters
    ----------
    x : list
        Mean values for proposed distribution.
    std : list
        Standard deviation for proposed distribution.
    Returns
    -------
    Proposal PDF
    TYPE
        Numpy array.
    """
    return np.random.normal(x,std,(len(x),))

def prior(x,lim_pri):
    """
    Parameters
    ----------
    x : list
        current values markov chain.
    lim_pri : list
        boundaries for uniform prior distribution.

    Returns
    -------
    int
        DESCRIPTION.
    """
    sw =0
    i =0
    while sw == 0:
        if lim_pri[i][0]<=x[i]<=lim_pri[i][1]:
            if i < len(x)-1:
                sw = 0
                i = i+1
            else:
                return 1
        else:
            sw = 1
    return 0

def acceptance_rule(x, x_new):
    """
    

    Parameters
    ----------
    x : list
        current values markov chain.
    x_new : list
        next values markov chain.

    Returns
    -------
    TYPE
        bolean 

    """
    if x_new>x:      
        return True
    else:
        accept= np.random.uniform(0,1)
    return (accept < (np.exp(x_new-x)))

 
def likelyhood(x,y,model):
    """
    

    Parameters
    ----------
    x : list
        current value of MCMC.
    y : Array
        FFT values.
    model : Function
       Theoretical PSD.
    Returns
    -------
    lik : float
        likelyhood value

    """
    yhat = model(x[:-1])
    err = np.linalg.norm(yhat-y)
    print(err)
    lik = -err/((10**-12))
    return lik
    
def metropolis_hastings(likelihood_computer,prior, transition_model, param_init,iterations,data):
    x = param_init
    accepted = []
    rejected = []  
    prob =[]
    for i in tqdm(range(iterations)):
        x_new =  transition_model(x)    
        x_lik = likelihood_computer(x,data)
        x_new_lik = likelihood_computer(x_new,data) 
        
        if (acceptance_rule(x_lik + np.log(prior(x)),x_new_lik+np.log(prior(x_new)))):            
            x = x_new
            accepted.append(x_new[:])
        else:
            rejected.append(x_new)            
            prob.append(x_new_lik)  
            
    return np.array(accepted), np.array(rejected),prob

# from MDOF_LSQ import CPSD,Model
# import matplotlib.pyplot as plt

# Nm = 3
# fs = 95
# Acco =  np.loadtxt('Data/Sp_Acc_11_02_12_7_18.txt',delimiter = ',')
# Acc = Acco[:,:3]
# Nc = Acc.shape[1]
# Nm = 2

# ff,cpsd,N,Fc = CPSD(Acc,fs,Nc,3.4,4)

# f=[3.719,3.8870]

# z = [0.005,0.0053]
# S = [-8,-8]
# Se = [-6]
# std = [-5]
# xo = [*f,*z,*S,*Se,*std]
# m_model = lambda x: Model(x,ff,Nm,N,Fc)

# std_tr = [0.001,0.001,0.001,0.001,0.5,0.5,0.5,1]
# tr_model = lambda x: transition_model(x,std_tr)

# pri_lim = [[3.6,3.8],[3.7,4],[0,0.05],[0,0.05],[-20,0],[-20,0],[-20,-1],[-30,-1]]
# pri_model = lambda x: prior(x,pri_lim)

# lik= lambda x,y: likelyhood(x,y,m_model)



# accepted,rejected,prob = metropolis_hastings(lik,pri_model, tr_model,xo,1000,cpsd)




# plt.close('all')
# plt.figure()
# plt.plot(accepted)
# plt.figure()
# xopt = np.mean(accepted,axis = 0)[:-1]
# plt.plot(ff,10*np.log10(m_model(xopt)))
# plt.plot(ff,10*np.log10(cpsd))
# plt.plot(ff,m_model(xo))

# zo = np.linspace(0,10,100)
# error =np.random.normal(0,1,100)
# y = 2.23*zo**2+4.56 + error

# #------- Transition Model ---- #
# std_tr = [0.5,0.5,0.01]
# tr_model = lambda x: transition_model(x,std_tr)
# #------- Prior limits ---- #
# pri_lim = [[0,10],[0,10],[0,10]]
# pri_model = lambda x: prior(x,pri_lim)
# #------- model ---- #
# m_model = lambda x: model(x,zo)
# #------- lik ---- #
# lik= lambda x,y: likelyhood(x,y,m_model)

# xoo = [2,4,0.1]
# accepted,rejected,prob = metropolis_hastings(lik,pri_model, tr_model,xoo,1000,y,acceptance)



            
                
            
            
