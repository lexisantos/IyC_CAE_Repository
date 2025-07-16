# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:38:38 2024

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
# import sympy as sym
import pandas as pd
import glob 
# from decimal import Decimal

# def L(f):
#     return sym.laplace_transform(f, t, s, noconds=True)

# def invL(F):
#     return sym.inverse_laplace_transform(F, s, t)

def LocalToMin(my_time):
    """
    Parameters
    ----------
    my_time : hora en formato hh:mm:ss.

    Returns
    -------
    t1 : devuelve el equivalente en min
    """
    factors = (60, 1, 1/60)
    t1 = sum(i*j for i, j in zip(map(int, my_time.split(':')), factors))
    return t1

def load_data(spnd, signal, head):
    """
    Carga de datos

    Parameters
    ----------
    spnd : no. de SPND.
    signal : str, dice qué señal va a cortar, i.e. Rh_I0, V_I0, Comp, Comp_I0, V o Rh.
    head : lista de nombre de cols.

    Returns
    -------
    lista con: Tiempos (min), Corriente (pA).

    """
    try:
        path = glob.glob(r"V {}\{}*.txt".format(spnd, signal))[0]
        Data = pd.read_csv(path, sep=r'\s+', skiprows = 1, header=0, names=head)
    except:
        return(print('No data'))
    if spnd != 1947:
        Tiempo = np.array([LocalToMin(tt) for tt in Data['T local']])
    else:
        Tiempo = np.round(np.array(Data['T local']))
    Corriente = np.array(Data['Corriente [pA]'])        
    return (Tiempo, Corriente)

#%%
spnd = 1971
h = ['Offset', 'T', 'T local', 'Corriente [pA]']

# tV_cut, iV_cut = np.transpose(np.loadtxt("Datos_cortados\{}\SPND{}_I_cut.txt".format('V', spnd)))

tV, iV = load_data(spnd, 'V', h)
tV -= tV[0]
iV = iV*1e-12

#%%
N = len(iV)

Pg = 2.51e-21
Pb = 3.6e-20
Fp = Pg/(Pg+Pb)
Fq = 1-Fp
tv_v = 225.6 #seg
sigma = 4.9e-24 #cm2
sens = 8.7e-21
V51 = 3.999*6.022e23/50.9415 #nuclear number density, n/cm3

Ts = 1
flujo_est = np.mean(iV[-200:-100])/sens
K1 = (2*Fp*tv_v - Ts)/(2*Fp*tv_v + Ts)
K2 = (Ts - 2*tv_v)/(2*Fp*tv_v + Ts)
K3 = (2*tv_v + Ts)/(2*Fp*tv_v + Ts)

k = np.arange(0, N, 1)
phi_comp = np.zeros(N)

for j, kj in enumerate(k[:-1]):
    phi_comp[j+1]= K1*phi_comp[j] + (1/(sigma*V51*(Pg+Pb)))*(K2*iV[j] + K3*iV[j+1])


# i_comp = phi_comp * sens

plt.figure()
plt.scatter(tV, phi_comp, label= 'flujo comp.')
plt.plot(tV, iV/sens, 'r-', label= '$i_{Rh}/sens$')
plt.axhline(flujo_est, xmin=tV[0], xmax= tV[-1], color='k', label='phi calc')

plt.legend()
plt.xlabel('t [s]')
plt.ylabel('$\phi_c$ [nv]')

#%% #Approx of K
## K1
r = 0.1 #cm

S_abs = sigma*V51
S_sca = 5.088e-24*V51

S_t = S_abs + S_sca

K = 0.183*np.sqrt(S_abs)/r

f = sci.special.jv(1, K*r)/sci.special.jv(0, K*r)

K1_approx = (2/(K*r))*f

P1 = 0.94
P_es = 0.08
K1_calc = P1*r*S_t/(r*S_abs*(2-P1*r*S_t) + 2*r*S_sca*P_es*r*S_t)

## E mean calc y K2
Z = 23
E_max = 2.54 #MeV

E_mean = 0.33*E_max*(1-np.sqrt(Z)/50)*(1 + np.sqrt(E_max)/4) #MeV

R_beta = 0.542*E_max - 0.133  #g/cm2 - Feather lineal approx about beta range. Another one #412*E_mean**(1.265- 0.0954*np.log(E_mean)) #mg/cm2
dens_V = 3.999/(np.pi*21*(0.2/2)**2) #g/cm3

l_calc = R_beta / dens_V #cm

K2 = 0.5*(1 -(r/l_calc)*sci.special.exp1(r/l_calc))

# l = np.arange(1,20)*0.01
# K2 = [0.5*(1 - r/ll*sci.special.exp1(r/ll)) for ll in l]
