# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:58:18 2024

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
# import scipy.signal as sig
# import sympy as sym
import pandas as pd
import glob 
# from decimal import Decimal

# sym.init_printing()

# t, s = sym.symbols('t, s')

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
    # Tiempo = np.array(Data['Offset'])
    if spnd != 1947:
        try:
            Tiempo = np.array([LocalToMin(tt) for tt in Data['T local']])
        except:
            time_local = [s.split('_')[1] for s in np.array(Data['T local'], dtype = str)]
            Tiempo = np.array([LocalToMin(tt) for tt in time_local])
    else:
        Tiempo = np.round(np.array(Data['T local']))
    Corriente = np.array(Data['Corriente [pA]'])        
    return (Tiempo, Corriente)

def DIM_V(i, y0, Pg, Pb, Ndens, Ts):
    '''
    DIM para señales de vanadio

    Parameters
    ----------
    i : array
        corriente en A.
    y : lista de 2 elementos
        cond iniciales.
    Fp : float
        fraction del prompt.
    Ndens : float 
        densidad atómica, at/cm3.
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
    sigma = 4.9e-24 #cm-2

    a = l*(Pb+Pg)/Pg
    A = -l*Pb/Pg
    zm = np.exp(-a*Ts)

    Xm = np.zeros(N)
    phi_comp = np.zeros(N)
    Xm[0], phi_comp[0] = ((A/a)*y0[0], y0[1])
    for j in range(N-1):
        Xm[j+1] = zm*Xm[j] + (A/a)*(1-zm)*i[j]
        phi_comp[j+1] = (i[j+1] + Xm[j+1])*1/(Pg*sigma*Ndens)
    return (Xm, phi_comp)

def cte_min(Kb, Kb_err):
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

def may_err(may, may_min, otra_min, lm, lg):
    may_min, dmay_min = may_min
    otra_min, dotra_min = otra_min
    dmay_maymin = (2*may_min-lm-lg + may)/(otra_min - may_min)
    dmay_otra = -may/(otra_min - may_min)
    return np.sqrt((dmay_maymin*dmay_min)**2 + (dmay_otra*dotra_min)**2)

def cte_may(a, b, sigma):
    lm, lg = np.log(2)/np.array([260.4, 42.3])
    Sm, Sg = np.array([0.077, 0.923])*sigma    
    A = (a[0] - lg)*(a[0] - lm)/(b[0] - a[0])
    B = (b[0] - lg)*(b[0] - lm)/(a[0] - b[0])
    dA = may_err(A, a, b, lm, lg)
    dB = may_err(B, b, a, lm, lg)
    return ([A, dA], [B, dB])

def X_j(A, a, Xaj_1, iRhj_1, Ts):
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

def DIM_Rh(i, y0, Ndens, ctes, Ts):
    a, b, A, B, P103, sigma = ctes
    N = len(i)

    Xa = np.zeros((N, 2))
    Xb = np.zeros((N, 2))
    phi_comp = np.zeros((N, 2))
    
    Xa[0] = (A[0]/a[0])*y0[0]*np.array([1, np.sqrt((A[1]/A[0])**2 + (a[1]/a[0])**2)])
    Xb[0] = (B[0]/b[0])*y0[0]*np.array([1, np.sqrt((B[1]/B[0])**2 + (b[1]/b[0])**2)])
    phi_comp[0] = [y0[1], 0]
    
    for jj in range(1, N):
        Xa[jj] = X_j(A, a, Xa[jj-1], i[jj-1], Ts)
        Xb[jj] = X_j(B, b, Xb[jj-1], i[jj-1], Ts)
        phi_comp[jj][0] = (1/(P103[0]*Ndens*sigma))*(i[jj] + Xa[jj][0] + Xb[jj][0])
        phi_comp[jj][1] = np.sqrt((phi_comp[jj][0]*P103[1]/P103[0])**2 + (1/(P103[0]*Ndens*sigma)**2)*(Xa[jj][1]**2 + Xb[jj][1]**2))
    return ([Xa, Xb, phi_comp])    

#%% RODIO  - con el mismo VSPND
# A1: P103 = [3.20e-21, 0.16e-21], P104 = [2.112e-20, 0.015e-20]
# A0: P103 = [2.97e-21, 0.17e-21], P104 = [2.135e-20, 0.017e-20]
# DW9: P103 = [2.228e-21, 0.024e-21], P104 = [2.2644e-20, 0.0025e-20] #A0


serie = 'DW9'
nombre = 'A0'

Ts = 1 #s
Pg = {'A0': [2.97e-21, 0.17e-21], 'DW9': [2.97e-21, 0.17e-21]}
Pb = {'A0': [2.135e-20, 0.017e-20], 'DW9': [2.135e-20, 0.017e-20]}

P103 = Pg[nombre]
P104 = Pb[nombre]

Fp = P103[0]/(P103[0]+P104[0])
Fq = 1 - Fp

dFq = (1/(P103[0]+P104[0]))*np.sqrt((P103[1]*Fq)**2 + (P104[1]*Fp)**2) #misma que dFp

Fq = [Fq, dFq] 

sigma103 = 143.5e-24
largo = {'DW9': 2.055, 'A0': 1}
N103 = 12.423*(6.022e23/102.905)*largo[nombre]*np.pi*(0.1/2)**2

# M = cte_M(P103, P104, sigma103)
a, b = cte_min(*Fq)
A, B = cte_may(a, b, sigma103)

sens = {'A0': 2.11, 'DW9': 4.10e-21, 'V': 8.7e-21} 
tRh, iRh = np.transpose(np.loadtxt("Datos_cortados\{}\SPND{}_I_cut.txt".format(nombre, serie))[:1700]) #load_data(spnd, 'Rh', h)
i0 = np.mean(np.loadtxt("Datos_cortados\{}_I0\SPND{}_I0_cut.txt".format(nombre, serie)))

iRh = (iRh - i0)*1e-12
tRh -= 0

flujo_est = np.mean(iRh[-100:])/sens[nombre]

y0 = [np.mean(iRh[-100:])/4, 0]

Xam, Xbm, phi_compRh = DIM_Rh(iRh, y0, N103, [a, b, A, B, P103, sigma103], Ts)

#%% Figura y elección de t0
plt.figure(3)
plt.axhline(flujo_est, color='k', linewidth=3.0, label='$\phi_{calc} ≈ $'+str(round(flujo_est*1e-9, 2))+'E9 nv'+'{}'.format(nombre), zorder = 10)
# plt.errorbar(tRh*60, phi_compRh[:, 0], yerr=phi_compRh[:, 1], fmt='ro', label= '$\phi_c$')#'= $I_{Rh, sat}$/4')
plt.plot(tRh*60, phi_compRh[:, 0], '.', label= '$\phi_c$ {}'.format(nombre))#'= $I_{Rh, sat}$/4')
plt.plot(tRh*60, iRh/sens[nombre], '.', label= '$i_{Rh}$'+'{}/sens'.format(nombre))
plt.legend(loc = 'lower right', fontsize=11)
plt.grid(ls='--')
plt.xlabel('t [s]', fontsize=13)
plt.ylabel('$\phi$ [nv]', fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=13)
# x0 = plt.ginput(n=-1, timeout=-1)

plt.figure(4)
plt.plot(tRh[5:]*60, 100*phi_compRh[:, 1][5:]/phi_compRh[:, 0][5:], label = 'flujo comp. err')
plt.xlabel('t [s]')
plt.ylabel('Err $\phi_c$ relativo %')
plt.legend()

plt.figure(5)
plt.plot(tRh*60, 100*np.abs(Xam[:, 1]/phi_compRh[:, 0]*1/(P103[0]*N103*sigma103)), 'g.', label = '$\Delta$Xa')
plt.plot(tRh*60, 100*np.abs(Xbm[:, 1]/phi_compRh[:, 0]*1/(P103[0]*N103*sigma103)), 'r.', label = '$\Delta$Xb')
plt.axhline(100*P103[1]/P103[0], color='k', label = '$\Delta$P103')
plt.yscale('log')
plt.legend(fontsize=11)
plt.xlabel('t [s]', fontsize=13)
plt.ylabel('Err $\phi_c$ relativo %', fontsize=13)
plt.grid(ls='--')

#%%
nprom_phi = 200
Tc = np.arange(nprom_phi, 1460, step=20)
phi_mean = np.zeros(len(Tc))
phi_mean_err = np.zeros(len(Tc))

for jj, tc in enumerate(Tc):
    phi_mean[jj] = np.mean(phi_compRh[:, 0][tc-nprom_phi:tc])
    phi_mean_err[jj] = np.sqrt(np.std(phi_compRh[:, 0][tc-nprom_phi:tc], ddof=1)**2 
                               + np.sum(phi_compRh[:, 1][tc-nprom_phi:tc]**2)*(1/nprom_phi)**2)

plt.figure(5)
plt.plot(Tc, 100*phi_mean_err/phi_mean, label = 'flujo comp. err')
plt.xlabel('t [s]')
plt.ylabel('Err $\phi_c$ relativo %')
plt.yscale('log')
plt.grid(ls='--', which='both')
plt.legend()

plt.figure(6)
plt.errorbar(Tc, phi_mean, yerr= phi_mean_err, fmt= '.', label = 'flujo comp.')
plt.xlabel('t [s]')
plt.ylabel('$\phi_c$ [nv]', fontsize=13)
plt.grid(ls='--', which='both')
plt.legend()

#%% Promedios en función de tc
# nprom = 200
# Tc = np.arange(nprom, len(phi_compRh), step=10) #24.75 min
# Prom_Rh = np.zeros((len(Tc), 2))

# for jj, tc in enumerate(Tc):
#     Prom_Rh[jj] = np.array([np.mean(phi_compRh[tc-nprom:tc]), np.std(phi_compRh[tc-nprom:tc], ddof=1)])
#     # phiRhmean, phiRhstd = np.array([np.mean(fRh[tc-100:tc]), np.sqrt(np.std(fRh[tc-100:tc], ddof=1)**2 + np.sum(dfRh[tc-100:tc]**2)/len(dfRh[tc-100:tc])**2)])*(sigma_Rh*N0_Rh)

# plt.figure(1)
# plt.errorbar(Tc, Prom_Rh[:,0], yerr=Prom_Rh[:, 1], fmt='.', label='Flujo promedio Rh, {} adq/tc'.format(nprom), zorder=0)
# # plt.axhline(flujo_est, color='k', label='$\phi_{calc} ≈ $'+str(round(flujo_est*1e-9, 2))+'E9 nv', zorder=2)
# # plt.axvline(int(x0[0][0]*60), ymin=min(phi_comp), ymax=max(phi_comp), color='g', label='t0')
# plt.legend(loc='lower right', fontsize=11)
# plt.ylabel('$\phi$ (tc)', fontsize=13)
# plt.xlabel('$t_c$ [s]', fontsize=13)
# plt.grid(ls='--')
# plt.tick_params(axis='both', which='major', labelsize=13)

# plt.figure(2)
# plt.plot(Tc, 100*Prom_Rh[:, 1]/Prom_Rh[:,0], '.', label='Prom. {} adq/tc'.format(nprom))
# # plt.plot(Tc, 100*np.abs(1-Prom_Rh[:,0]/flujo_est), '.', label='$|1 - \phi_c / \phi_{calc}|$')
# plt.legend(fontsize=11)
# plt.yscale('log')
# plt.ylabel('Err relativo %', fontsize=13)
# plt.xlabel('$t_c$ [s]', fontsize=13)
# plt.grid(ls='--')
# plt.tick_params(axis='both', which='major', labelsize=13)

#%% VANADIO

Ts = 1 #s
Pg = 2.7e-21
Pb = 3.58e-20
V51 = 3.99*6.022e23/50.9415

tV, iV = np.transpose(np.loadtxt("Datos_cortados\{}\SPND{}_I_cut.txt".format('V', spnd))) #load_data(spnd, 'V', h)
tV -= tV[0]
iV = iV*1e-12

y0 = [np.mean(iV[:10]), flujo_est]

Xm, phi_comp = DIM_V(iV, y0, Pg, Pb, V51, Ts) 

#%% Figura y elección de t0
plt.figure(8)
plt.scatter(tV, phi_comp, label= 'flujo comp.')
plt.plot(tV, iV/sensV, 'r-', label= '$i_{Rh}/sens$')
# plt.plot(tV, Xm)
plt.axhline(flujo_est, xmin=tV[0], xmax= tV[-1], color='k', label='phi calc')
plt.legend()
plt.xlabel('t [min]')
plt.ylabel('$\phi_c$ [nv]')
# x0 = plt.ginput(n=-1, timeout=-1)

#%% Promedios en funcion de tc

nprom = 100
Prom_V = np.zeros((len(Tc), 2))

for jj, tc in enumerate(Tc):
    Prom_V[jj] = np.array([np.mean(phi_comp[tc-nprom:tc]), np.std(phi_comp[tc-nprom:tc], ddof=1)])
    # phiRhmean, phiRhstd = np.array([np.mean(fRh[tc-100:tc]), np.sqrt(np.std(fRh[tc-100:tc], ddof=1)**2 + np.sum(dfRh[tc-100:tc]**2)/len(dfRh[tc-100:tc])**2)])*(sigma_Rh*N0_Rh)

plt.figure(1)
plt.errorbar(Tc, Prom_V[:,0], yerr=Prom_V[:, 1], fmt='.', label='Flujo promedio {}, {} adq/tc'.format('V', nprom))
# plt.axhline(flujo_est, xmin=tV[0], xmax= tV[-1], color='k', label='phi calc')
# plt.axvline(int(x0[0][0]*60), ymin=min(phi_comp), ymax=max(phi_comp), color='g', label='t0')
plt.legend(loc='lower left')
plt.ylabel('$\phi$ (tc)')
plt.xlabel('Tc [s]')
plt.grid(ls='--')

plt.figure(2)
plt.plot(Tc, 100*Prom_V[:, 1]/Prom_V[:,0], '.', label='{} adq/tc'.format(nprom))
plt.plot(Tc, 100*np.abs(1-Prom_V[:,0]/flujo_est), '.', label='Diff comp-sat')
plt.legend()
# plt.yscale('log')
plt.ylabel('Err $\phi$ comp, %')
plt.xlabel('Tc [s]')
plt.grid(ls='--')

#%% Comparación VSPND y RhSPND

plt.figure(3)
plt.plot(Tc, 100*np.abs(1-Prom_V[:,0]/Prom_Rh[:, 0]), '.')
plt.yscale('log')
plt.ylabel('Diff $\phi$ comp, %')
plt.xlabel('Tc [s]')
plt.grid(ls='--')

#%% Intento hacer las cuentas de Laplace y Z 

# s51, N51, Pb, Pg, tv = sym.symbols('s51, N51, Pb, Pg, tv')

# Fp = Pg/(Pg+Pb)
# Fp_v = Fp.subs([(Pg, 3.487e-21), (Pb, 3.846e-20)])

# Fq = 1-Fp_v

# tv_v = tv.subs ([(tv, 325.47)]) #seg
# m = 1/(Fp_v*tv_v)

# M = -Fq*m

# Gdi = (1/(Pg*s51*N51))*(1 + M/(s+m))

# Gdi_v = Gdi.subs([(Pg, 3.487e-21), (s51, 4.9e-24), (N51, 7.1655785e22)])

# num, den = Gdi_v.as_numer_denom()
# num = np.fromiter(num.as_poly(s).all_coeffs(), dtype=float)
# den = np.fromiter(den.as_poly(s).all_coeffs(), dtype=float)

# F_s = sig.lti(num, den)
# dt = 1

# F_z_zoh = F_s.to_discrete(dt=dt, method='zoh')
# print(F_z_zoh)

# t = np.arange(0, 1440, 1)

# t, y_zoh = F_z_zoh.step(t=t)

# # plot the step responses
# t, y_s = F_s.step(T=t)
# plt.plot(t, y_s, color='k')
# plt.scatter(t, np.squeeze(y_zoh), color='r')

#%% Errores de las constantes

# def Xa_err(A, a, niter, iRh, Ts):
#     a, da = a
#     A, dA = A
#     an = np.append(np.array(y0[0]-iRh[0]), -np.diff(iRh))
#     za, dza = [np.exp(-a*Ts), -da*Ts*np.exp(-a*Ts)]
#     N = np.arange(1, niter)
#     dAa = (A/a)*np.sqrt((dA/A)**2 + (da/a)**2)

#     dXa_za = np.zeros(niter)
#     dXa_Aa = np.zeros(niter)
    
#     for jj in N:
#         dXa_Aa[jj] = iRh[jj-1] + np.sum(an[:jj]*za**(jj-np.arange(jj)))
#         dXa_za[jj] = np.sum((jj - np.arange(jj))*an[:jj]*za**(jj - 1 -np.arange(jj)))

#     term_za = dza*(A/a)*dXa_za
#     term_Aa = dAa*dXa_Aa
    
#     dXa = np.sqrt(term_Aa**2 + term_za**2)
#     dXa[0] = np.abs(dAa)*y0[0]
#     return term_za, term_Aa, dXa 

# def Xa_err_v2(A, a, niter, Xa, Ts):
#     a, da = a
#     A, dA = A
#     an = np.append(np.array(y0[0]-iRh[0]), -np.diff(iRh))
#     za = np.exp(-a*Ts)
#     N = np.arange(1, niter)
#     dXa_a = np.zeros(niter)
#     dXa_A = np.zeros(niter)
        
#     for jj in N: #n de las ecs. es np.arange(jj) (rdo que el arange llega hasta jj-1 = j, que es hasta donde va la suma)
#         dXa_a[jj] = -(1/a)*(Xa[jj] + A*Ts*np.sum((jj - np.arange(jj))*an[:jj]*za**(jj -np.arange(jj)))) 
#         dXa_A[jj] = Xa[jj]/A

#     term_a = da*dXa_a
#     term_A = dA*dXa_A
    
#     dXa = np.sqrt(term_a**2 + term_A**2)
#     dXa[0] = np.sqrt((dA/A)**2 + (da/a)**2)*Xa[0]
#     return term_a, term_A, dXa 

#%%
# spnd = 1950

# Ts = 1 #s
# lm, lg = np.log(2)/np.array([260.4, 42.3])
# sigma103 = 143.5e-24
# N103 = 12.423*(6.022e23/102.905)*np.pi*(0.1/2)**2

# P103, P104 = 0.304E-20, 2.129E-20 #A0
# dP103, dP104 = (0.022e-20, 0.022e-20) #A0

# Fp = P103/(P103+P104)
# Fq = P104/(P103+P104)

# dFq = (1/(P103+P104))*np.sqrt((dP103*Fq)**2 + (dP104*Fp)**2) #misma que dFp

# a, b = cte_min(P103, P104)
# A, B = cte_may(a, b, sigma103)

# tRh, iRh = np.transpose(np.loadtxt("Datos_cortados\{}\SPND{}_I_cut.txt".format('Rh', spnd))) #load_data(spnd, 'Rh', h)
# i0 = np.mean(np.loadtxt("Datos_cortados\{}_I0\SPND{}_I0_cut.txt".format('Rh', spnd)))
# i0_err = np.std(np.loadtxt("Datos_cortados\{}_I0\SPND{}_I0_cut.txt".format('Rh', spnd))*1e-12, ddof=1)

# iRh = (iRh - i0)*1e-12
# tRh -= tRh[0]

# alfa, alfa_prima = np.array([lg*(Fp + 0.923*Fq) + lm*Fp, -0.077*lg-lm])

# f = np.sqrt(alfa**2 - 4*Fp*lm*lg)

# a, b = (1/(2*Fp))*np.array([(alfa + f), (alfa - f)]) 
# a_prima, b_prima = np.array([1, -1])*lm*lg/(Fp*f) + (1/Fp + alfa_prima/f)*np.array([a, b]) #(1/(2*Fp))*(1/Fp + alfa_prima/f)*(alfa + f) + lm*lg/(Fp*f)

# da, db = np.abs(np.array([a_prima, b_prima])*dFq)

# dA = may_err(A, [a, da], [b, db], lm, lg)
# dB = may_err(B, [b, db], [a, da], lm, lg)

# y0 = [iRh[0], 0]
# Xam, Xbm, phi_compRh = DIM_Rh(iRh, y0, N103, [a, b, A, B, P103, sigma103], Ts)

# niter = len(iRh)
# N = np.arange(niter)
# # term_za, term_Aa, dXa = Xa_err_v2([A, dA], [a, da], niter, Xam, Ts)
# # term_zb, term_Bb, dXb = Xa_err_v2([B, dB], [b, db], niter, Xbm, Ts)
# term_za, term_Aa, dXa = Xa_err_v3([A, dA], [a, da], Xam, iRh, Ts, y0)
# term_zb, term_Bb, dXb = Xa_err_v3([B, dB], [b, db], Xbm, iRh, Ts, y0)

# phi_c_err = np.sqrt((phi_compRh*dP103/P103)**2 + ((dXa**2 + dXb**2)/(P103*N103*sigma103)**2))

#%%
# plt.figure(1)
# plt.plot(N, term_za, '.', label = 'term da')
# plt.plot(N, term_Aa, '.', label = 'term dA')
# plt.plot(N, dXa, '.', label = 'Err Xa')
# # plt.axhline(i0_err, label = 'Err Fondo')
# plt.yscale('log')
# plt.legend()
# plt.xlabel('N iter')

# plt.figure(2)
# plt.plot(N, term_zb, '.', label = 'term db')
# plt.plot(N, term_Bb, '.', label = 'term dB')
# # plt.axhline(i0_err, label = 'Err Fondo')
# plt.plot(N, dXb, '.', label = 'Err Xb')
# plt.yscale('log')
# plt.legend()
# plt.xlabel('N iter')

#%%


