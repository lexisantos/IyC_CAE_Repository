# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:23:01 2024

@author: Alex
"""
import numpy as np
from numba import njit

sigmas_oiea = {'V': 4.90e-24, 'Rh': 143.5e-24}
ctes_SPND = {'A1': {'P103': [3.20e-21, 0.16e-21], 'P104': [2.112e-20, 0.015e-20]},
             'A0': {'P103': [2.97e-21, 0.17e-21], 'P104': [2.135e-20, 0.017e-20]}, 
             'DW9': {'P103': [2.228e-21, 0.024e-21], 'P104': [2.2644e-20, 0.0025e-20]},
             'V-210': {'Pg': [0.27e-20, 0.02e-20], 'Pb': [3.58e-20, 0.03e-20]}}

#%% Rodio

def Rh_ctes_min(Kb, Kb_err):
    lm, lg = np.log(2)/np.array([260.4, 42.3])
    Kp = 1 - Kb
    term_1 = lg*(Kp + 0.923*Kb) + lm*Kp
    term_2 = np.sqrt(term_1**2 - 4*Kp*lg*lm)
    a, b = [(term_1+term_2)/(2*Kp), (term_1-term_2)/(2*Kp)]
    
    alfa, alfa_prima = np.array([lg*(Kp + 0.923*Kb) + lm*Kp, -0.077*lg-lm])
    fterm = np.sqrt(alfa**2 - 4*Kp*lm*lg)
    a_prima, b_prima = np.array([1, -1])*lm*lg/(Kp*fterm) + (1/Kp + alfa_prima/fterm)*np.array([a, b])
    da, db = np.abs(np.array([a_prima, b_prima])*Kb_err)
    return [[a, da], [b, db]]

def Rh_ctes_may_err(may, may_min, otra_min, lm, lg):
    may_min, dmay_min = may_min
    otra_min, dotra_min = otra_min
    dmay_maymin = (2*may_min-lm-lg + may)/(otra_min - may_min)
    dmay_otra = -may/(otra_min - may_min)
    return np.sqrt((dmay_maymin*dmay_min)**2 + (dmay_otra*dotra_min)**2)

def Rh_ctes_may(a, b):
    lm, lg = np.log(2)/np.array([260.4, 42.3])
    Sm, Sg = np.array([0.077, 0.923])*sigmas_oiea['Rh']   
    A = (a[0] - lg)*(a[0] - lm)/(b[0] - a[0])
    B = (b[0] - lg)*(b[0] - lm)/(a[0] - b[0])
    dA = Rh_ctes_may_err(A, a, b, lm, lg)
    dB = Rh_ctes_may_err(B, b, a, lm, lg)
    return ([A, dA], [B, dB])

@njit
def Rh_Xj(A, a, Xaj_1, iRhj_1, Ts):
    a, da = a
    A, dA = A
    za = np.exp(-a*Ts)
    dXa_a = Ts*za*((A/a)*iRhj_1 - Xaj_1[0]) - (1 - za)*(A/a**2)*iRhj_1 
    dXa_A = (1 - za)*iRhj_1/a

    term_a = da*dXa_a
    term_A = dA*dXa_A
    
    Xa = za*Xaj_1[0] +(1-za)*(A/a)*iRhj_1
    dXa = np.sqrt((Xaj_1[1]*za)**2 + term_a**2 + term_A**2)
    return np.array([Xa, dXa])

@njit
def DIM_Rh(i, y0, Ndens, model, Ts):
    sigma = sigmas_oiea['Rh']
    P103 = []
    while P103 == []:
        try:
            P103, P104 = list(ctes_SPND[model].values())
        except:
            model = str(input('Ingrese un modelo de RhSPND válido (A0, A1, DW9):\n'))
    Fp = P103[0]/(P103[0]+P104[0])
    Fq = 1 - Fp
    dFq = (1/(P103[0]+P104[0]))*np.sqrt((P103[1]*Fq)**2 + (P104[1]*Fp)**2) #misma que dFp
    Fq = [Fq, dFq] 

    a, b = Rh_ctes_min(*Fq)
    A, B = Rh_ctes_may(a, b)
    N = len(i)

    Xa = np.zeros((N, 2))
    Xb = np.zeros((N, 2))
    phi_comp = np.zeros((N, 2))
    
    Xa[0] = (A[0]/a[0])*y0[0]*np.array([1, np.sqrt((A[1]/A[0])**2 + (a[1]/a[0])**2)])
    Xb[0] = (B[0]/b[0])*y0[0]*np.array([1, np.sqrt((B[1]/B[0])**2 + (b[1]/b[0])**2)])
    phi_comp[0] = [y0[1], 0]
    
    for jj in range(1, N):
        Xa[jj] = Rh_Xj(A, a, Xa[jj-1], i[jj-1], Ts)
        Xb[jj] = Rh_Xj(B, b, Xb[jj-1], i[jj-1], Ts)
        phi_comp[jj][0] = (1/(P103[0]*Ndens*sigma))*(i[jj] + Xa[jj][0] + Xb[jj][0])
        phi_comp[jj][1] = np.sqrt((phi_comp[jj][0]*P103[1]/P103[0])**2 + (1/(P103[0]*Ndens*sigma)**2)*(Xa[jj][1]**2 + Xb[jj][1]**2))
    return ([Xa, Xb, phi_comp])    

## Podría agregar las constantes en un dict al principio, y que las use para A0 o A1, dependiendo lo que se pida.

#%% Vanadio

def DIM_V(i, y0, Ndens, model, Ts):
    '''
    DIM para señales de vanadio

    Parameters
    ----------
    i : array,
        Corriente en A.
    y0 : list,
        Cond iniciales.
    Ndens : float 
        Densidad atómica, at/cm3.
    model: 'str'
        
    Ts : float
        periodo de muestreo.

    Returns
    -------
    Xm : array
        variables de estado.
    phi_comp : array
        flujo compensado.
    '''
    N = len(i)
    l = np.log(2)/225.6
    sigma = sigmas_oiea['V']
    Pg = []
    while Pg == []:
        try:
            Pg, Pb = list(ctes_SPND[model].values())
        except:
            model = str(input('Ingrese un modelo de VSPND válido (V-210):\n'))
    Fp = Pg[0]/(Pg[0]+Pb[0])
    Fq = 1 - Fp
    dFq = (1/(Pg[0]+Pb[0]))*np.sqrt((Pg[1]*Fq)**2 + (Pb[1]*Fp)**2) #misma que dFp
    Fq = [Fq, dFq] 
    
    a = l/Fp[0]
    A = -l*Pb[0]/Pg[0]
    zm = np.exp(-a*Ts)

    Xm = np.zeros(N)
    phi_comp = np.zeros(N)
    Xm[0], phi_comp[0] = ((A/a)*y0[0], y0[1])
    for j in range(N-1):
        Xm[j+1] = zm*Xm[j] + (A/a)*(1-zm)*i[j]
        phi_comp[j+1] = (i[j+1] + Xm[j+1])*1/(Pg[0]*sigma*Ndens)
    return (Xm, phi_comp)
