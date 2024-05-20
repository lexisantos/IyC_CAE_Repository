# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:29:54 2024

@author: Alex
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import sympy as sym
# from numba import njit
from scipy.optimize import curve_fit
import os

sigmas_oiea = {'V': 4.90e-24, 'Rh': 143.5e-24}
spnd_emitter = {'A0': 'Rh', 'A1': 'Rh', 'V-210': 'V', 'DW9': 'Rh'}

#%% Ecuaciones físicas

def seccioneff_Maxw(s_0, T):
    """
    Corrige la sección eficaz por temperatura, usando una distribución Maxwelliana.

    Parameters
    ----------
    s_0 : float
        Sección eficaz térmica.
    T : float
        Temperatura en K.

    Returns
    -------
    float
        sección eficaz corregida..

    """
    T += 273.15
    T0 = round(0.0253/8.6173324e-5, 2)
    a = np.sqrt(np.pi*T0/(4*T))
    return s_0*a

class sens:
    """
    Funciones i/phi de los SPNDs conocidos
    """
    def V(t, P_g, P_b):
        t52 = 224.6
        return (P_g + P_b*(1-np.exp(-np.log(2)*t/t52)))
    def Rh(t, P103, P104): #, P104m):
        t104m, t104 = (260.4, 42.3)
        a = 0.077/(1-t104/t104m)
        return (P103 + P104*(1 + (a-1)*np.exp(-np.log(2)*t/t104) - a*np.exp(-np.log(2)*t/t104m)))# + P104m*(1-np.exp(-np.log(2)*t/t104m)))

#%% Herramientas 
def LocalToSeg(my_time):
    """
    Parameters
    ----------
    my_time : hora en formato hh:mm:ss.

    Returns
    -------
    t1 : devuelve el equivalente en segundos
    """
    factors = (3600, 60, 1)
    t1 = sum(i*j for i, j in zip(map(float, my_time.split(':')), factors))
    return t1

def redondeo(mean, err, cs, texto = False):
    """
    Devuelve el valor medio con la cantidad de cifras significativas definidas.
    """
    digits = -np.floor(np.log10(err)).astype(int)+cs-1
    if err<1:
        err_R = format(np.round(err, decimals = digits), f'.{digits}f')
        mean_R = str(np.round(mean, decimals = len(err_R)-2))
    else:
        err_R = format(np.round(err, decimals = digits), '.0f')
        ndot = 0
        mean_R = format(np.round(mean, decimals = cs-1-len(err_R)-ndot), '.0f')
    if texto == True:
        return ('('+mean_R+' ± '+err_R+')')
    else:
        return (float(mean_R), float(err_R))

def select_from_plot(data, labels, cant, fmt):
    """
    Devuelve los puntos seleccionados en una lista
    
    Parameters
    ----------
    data : list, float
        (x, y) de los datos.
    labels : list, str
        Labels del eje x e y (en ese orden).
    cant : int
        Cantidad de puntos esperados.
    fmt : str
        Formato de puntos en el gráfico.

    Returns
    -------
    xs : list
        Lista de puntos seleccionados, sólo si su longitud coincide con cant. Caso contrario, vuelve a pedir que selecciones los puntos.

    """
    while True:    
        xs = []
        labelx, labely = labels
        x, y = data
        plt.figure(1, figsize=(6,6))
        plt.plot(x, y, fmt)
        plt.xlabel(labelx)
        plt.ylabel(labely)
        plt.grid(ls = '--')
        xs.append(plt.ginput(n = -1, timeout=-1))
        if len(xs[0]) != cant:
            print('Seleccione la cantidad correcta de puntos')
            plt.clf()
        else:
            plt.pause(0.15)
            plt.close()
            break
    return xs

#%% Proceso de datos

def data_tanda(listaSPND, lista_sig, lista_tirr, lista_Ts, lista_versiones, lista_det, lista_pat):
    """
    Devuelve un DataFrame con los parámetros de las calibraciones a procesar.
    
    Parameters
    ----------
    listaSPND : list, str
        Nros de series de los SPNDs.
    lista_sig : list, str
        Indica si vemos el patrón o el detector para cada serie.
    lista_tirr : list, float
        Tiempos de irradiación, en segundos.
    lista_Ts : list, float
        Periodo de sampleo, en segundos.
    lista_versiones : list, str
        Con qué versión del programa fueron levantados, e.g. 'old' o 'new'.
    lista_det : list, str
        Lista de modelo de detectores. 
    lista_pat: list, str
        Lista de modelo de patrones.

    Returns
    -------
    data : obj, DataFrame
        DESCRIPTION.

    """
    head = ['Señal', 'Tirradiacion', 'Ts', 'Version', 'modelDet', 'modelPat']
    arr = np.transpose([lista_sig, lista_tirr, lista_Ts, lista_versiones, lista_det, lista_pat])
    data = pd.DataFrame(arr, index = listaSPND, columns = head)
    return data

class processdata:
    def __init__(self, noserie, signal, irradiation_time, sample_time, version, modeldet, modelpat):
        self.noserie = noserie
        self.signal = signal.capitalize()
        self.version = version.lower()
        self.tirr = int(irradiation_time)
        self.modelDet = modeldet
        self.modelPat = modelpat
        self.Ts = float(sample_time)
        
    def load_data(self, path: str=os.getcwd()):
        separador = {'old': ' ', 'new': '_'}
        nombret = {'old': 'T Local', 'new': 'T_Local'}
        nombrei = {'old': 'Corriente [pA]', 'new': f'Crudo{self.signal}'}
        nombretxt = {'old': spnd_emitter[getattr(self, 'model' + self.signal[:3])], 
                     'new': str(self.noserie[0])}
        try: 
            path = glob.glob(path+ r"/{}/{}*.txt".format(self.noserie, nombretxt[self.version]))[0]
            Data = pd.read_csv(path, sep='\t', encoding='unicode_escape')
            ifin = np.where(Data[nombret[self.version]] == 'FIN')[0]
            if ifin.size != 0:
                Data = Data[:ifin[0]]
            Tiempo = np.array([LocalToSeg(s.split(separador[self.version])[1]) 
                               for s in np.array(Data[nombret[self.version]], dtype = str)])
        except:
            return(print('No hay datos de esa serie o señal. Revisar versión ingresada o el formato de guardado en T Local.'))
        Corriente = np.array(Data[nombrei[self.version]])
        return (Tiempo, Corriente)
        
    def data_cut(self, path: str=os.getcwd()):
        T, I = self.load_data(path)
        text = {'pat': '(t inicio fondo, t final fondo, t inicio irradiación)', 
                'det': '(t inicio fondo, t final fondo)'}
        nro_pts = {'pat': 3, 'det': 2}
        newpath = r'{}/Datos_cortados/Pos_cuts'.format(path) 
        señal = self.signal.lower()[:3]
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        print("\r Seleccione, en el siguiente orden: {}".format(text[señal]))
        x = select_from_plot([T-T[0], I], ['Tiempo [s]', 'Corriente [pA]'], nro_pts[señal], '.-') 
        t_sel = [round(xx[0]/self.Ts) for xx in x[0]]
        x01, x02 = t_sel[:2]
        I0 = I[x01:x02]
        if señal == 'pat':
            i1 = t_sel[2]
            dt = np.diff(T)
            dt[dt==0]= 1 # Elimino errores por repetición de T Local
            i2 = i1 + self.tirr
            np.savetxt('{}/Pos_cuts_{}.txt'.format(newpath, self.noserie), [i1,i2])
        elif señal == 'det':
            try:
                xs = np.loadtxt("{}/Pos_cuts_{}.txt".format(newpath, self.noserie))
            except:
                return print('No se encuentran datos')
            j1, j2 = xs #indices de dónde fue cortado iRh
            j1, j2 = [int(j1), int(j2)]
            if self.version == 'old':
                tRh, iRh = self.load_data(path)
                i1, i2 = [np.where(T == tRh[j1])[0][0], np.where(T == tRh[j2])[0][0]]
                dt = T[0] - tRh[0]
                print(f'Diferencia temporal con patrón de {dt}, para {self.noserie}')
            elif self.version == 'new':
                i1, i2 = j1, j2
            else:
                return print('Versión inválida')
        else:
            return print('Señal inválida')
        return([T[i1:i2]-T[i1], I[i1:i2], I0], [T[1:]-T[i1], np.diff(I)])
    
    def data_cut_save(self, path: str=os.getcwd()):
        newpath = r'{}/Datos_cortados/Pos_cuts'.format(path) 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        it_cut, diff =  self.data_cut(path)
        t_cut, i_cut, i0_cut = it_cut
        try:
            emisor = spnd_emitter[getattr(self, 'model' + self.signal[:3])]
            newpath = path+f'/Datos_cortados/{self.noserie}'
            np.savetxt(f'{newpath}/{self.signal}_{emisor}_Icut_SPND{self.noserie}.txt', np.transpose([t_cut, i_cut]), header='Tiempo[s] Corriente[pA]')
            np.savetxt(f'{newpath}/{self.signal}_{emisor}_I0cut_SPND{self.noserie}.txt', np.transpose(i0_cut), header='Corriente[pA]')
            np.savetxt(f'{newpath}/{self.signal}_{emisor}_diff_SPND{self.noserie}.txt', np.transpose(diff))
            print(f"\r{self.signal} de la serie {self.noserie} fue guardado.")
        except:
            return print(f"{self.noserie} no pudo ser guardado")

#%% Análisis de datos

def param_iniciales(emisor, t, i, ctes, Ts: int=1):
    """
    Guess inicial para ajustes con modelos de SPNDs. Toma tres puntos equiespaciados en [50, -1].

    Parameters
    ----------
    emisor : str.
        DESCRIPTION.
    t : array
        Datos de tiempo, en segundos.
    i : array
        Datos de corriente (A).
    ctes : array or list
        Sec eficaz (cm-2), flujo neutrónico, nro de nucleidos padre.
    Ts : float
        Período de sampleo (default = 1).

    Returns
    -------
    array, float
        Devuelve un array con una estimación de las constantes P.

    """
    emisor = emisor.lower()
    sigma, phi, N0 = ctes
    ts = np.linspace(50, len(t), num=3, endpoint=True, dtype=int)*Ts
    if emisor == 'rh':
        i0 = np.zeros(2)
        M = np.zeros((2, 2)) #3 si P104m
        a, ti, tRh104, tRh104m, w104m = sym.symbols('a, ti, t104, t104m, w104m')
        A = sym.Array([1, 1+ (a-1)*sym.exp(-sym.log(2)*ti/tRh104)-a*sym.exp(-sym.log(2)*ti/tRh104m)])#, w104m*(1 - sym.exp(-sym.log(2)*ti/tRh104m))])
        for ii in range(len(ts)):
            t1 = t[ts[ii]] 
            t104m, t104 = (260.4, 42.3)
            A_v = A.subs([(ti, t1), (tRh104, t104), (tRh104m, t104m), (a, 0.077/(1-t104/t104m))]) #, (w104m, 0.077)
            M[ii] = np.asarray(A_v).astype(np.float64)
            i0[ii] = i[ts[ii]]
    elif emisor == 'v':
        i0 = np.zeros(2)
        M = np.zeros((2, 2))
        ti, t52 = sym.symbols('ti, t52')
        A = sym.Array([1, 1-sym.exp(-sym.log(2)*ti/t52)])
        for ii in range(len(ts)):
            t1 = t[ts[ii]]  
            tv = 224.6
            A_v = A.subs([(ti, t1), (t52, tv)])
            M[ii] = np.asarray(A_v).astype(np.float64)
            i0[ii] = i[ts[ii]]
    else:
        return print('Emisor inválido')
    M_inv = np.linalg.inv(M)
    return 1/(sigma*phi*N0)*np.dot(M_inv, i0)

def ajuste_model(t, i, i_err, emisor, params):
    """
    Devuelve el ajuste para una señal de corriente del SPND, con el flujo previamente calculado.

    Parameters
    ----------
    t : array, float
        Tiempo [s].
    i : array, float
        Corriente [pA].
    i_err : float
        Desv. estándar de la corriente de fondo [pA].
    emisor : str
        Emisor del SPND.
    params : list
        [sec eficaz (cm-2), flujo (nv, calculado desde algún patrón), N (nro de nucleos padre)].

    Returns
    -------
    tuple
        Modelo de sens (t, *popt), cte P's, R2, residuo

    """
    emisor = emisor.capitalize()
    try:
        S_model = getattr(sens, emisor)
    except:
        return print('Emisor inválido')
    p_initial = param_iniciales(emisor, t , i*1e-12, params)
    S = i*1e-12/np.prod(params)
    popt, pcov = curve_fit(S_model, t, S, p0=p_initial, sigma=np.full(len(i), i_err*1e-12)) #np.full(len(sens), i0_err*1e-12/(sigma*N0*flujos[ii]))
    residuals = S - S_model(t, *popt)
    perr = np.sqrt(np.diag(pcov))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((S-np.mean(S))**2)
    R2 = 1 - (ss_res / ss_tot)
    Ps = np.dtstack((popt, perr))
    return (S_model(t, *popt), Ps, R2, residuals)

#@njit
def massive_fits(t_cut, listaSPND, emisor, params, dts, path, Ts):
    """
      Hace ajustes a un grupo de datos de SPND con el mismo emisor, para i/(phi*sigma*N). 
      Es importante que el nro de serie esté en los nombres de los archivos. 

    Parameters
    ----------
    t_cut : list, int
        Intervalo de posiciones (de t) en donde hacer los ajustes.
    listaSPND : lista
        Números de serie de SPNDs en el grupo.
    emisor : str
        Nombre del elemento emisor (V, Rh).
    params : dict
        {spnd: [flujo (nv, calculado desde algún patrón), N (nro de nucleos padre)]}
    dts : TYPE
        DESCRIPTION.
    path : str, 2 entradas format()
        DESCRIPTION.
    Ts : list, int
        Lista de períodos de muestreo.

    Returns
    -------
    None.

    """
    emisor = emisor.capitalize()
    sigma = sigmas_oiea[emisor]
    t_init, t_fin = t_cut
    P_tot = np.zeros((len(listaSPND), 2))
    P_err = np.zeros((len(listaSPND), 2))
    R2 = np.zeros(len(listaSPND))
    Res = []
    try:
        sens_model = getattr(sens, emisor)
    except:
        return print('Emisor inválido')
    for ii, spnd in enumerate(listaSPND):
        t, i = np.transpose(np.loadtxt(path.format('I', spnd)))
        t, i = t[t<=t_fin], i[t<=t_fin]
        t, i = t[t>=t_init], i[t>=t_init]
        
        i0 = np.loadtxt(path.format('I0', spnd))
        i0, i0err = (np.mean(i0), np.std(i0, ddof=1))
        
        i -= i0
        t = t*60 -dts[spnd]
        p_initial = param_iniciales(emisor, t , i*1e-12, [sigma, *params[spnd]], Ts[spnd])
        S = i*1e-12/np.prod([sigma, *params[spnd]])
        popt, pcov = curve_fit(sens_model, t, S, p0=p_initial, sigma=np.full(len(i), i0err*1e-12)) #np.full(len(sens), i0_err*1e-12/(sigma*N0*flujos[ii]))
        Res.append(S - sens_model(t, *popt))
        perr = np.sqrt(np.diag(pcov))
        ss_res = np.sum(Res[-1]**2)
        ss_tot = np.sum((S-np.mean(S))**2)
        R2[ii] = 1 - (ss_res / ss_tot)
        P_tot[ii] = popt
        P_err[ii] = perr
    P_tot, P_err, R2, Res = [{a:b for a, b in zip(listaSPND, xx)} for xx in [P_tot, P_err, R2, Res]]
    return (P_tot, P_err, R2, Res)
