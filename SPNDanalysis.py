# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:29:54 2024

@author: Alex
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import sys
import sympy as sym
import scipy as sci
from scipy.optimize import curve_fit
import os
import re

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

def sens_model(emisor):
    if emisor == 'V':
        def sens_mod(t, P_g, P_b):
            t52 = 224.6
            return (P_g + P_b*(1-np.exp(-np.log(2)*t/t52)))
    elif emisor == 'Rh':
        def sens_mod(t, P103, P104): #, P104m):
            t104m, t104 = (260.4, 42.3)
            a = 0.077/(1-t104/t104m)
            return (P103 + P104*(1 + (a-1)*np.exp(-np.log(2)*t/t104) - a*np.exp(-np.log(2)*t/t104m)))# + P104m*(1-np.exp(-np.log(2)*t/t104m)))
    else: 
        return print('No hay modelos definidos para este emisor')
    return sens_mod

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
    sigma, phi, N0 = ctes
    ts = np.linspace(50, len(t), num=3, endpoint=True, dtype=int)*Ts
    if emisor == 'Rh':
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
    elif emisor == 'V':
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
    S_model = sens_model(emisor)
    p_initial = param_iniciales(emisor, t , i*1e-12, params)
    sens = i*1e-12/np.prod(params)
    popt, pcov = curve_fit(S_model, t, sens, p0=p_initial, sigma=np.full(len(i), i_err*1e-12)) #np.full(len(sens), i0_err*1e-12/(sigma*N0*flujos[ii]))
    residuals = sens - S_model(t, *popt)
    perr = np.sqrt(np.diag(pcov))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((sens-np.mean(sens))**2)
    R2 = 1 - (ss_res / ss_tot)
    Ps = np.dtstack((popt, perr))
    return (S_model(t, *popt), Ps, R2, residuals)

def massive_fits(t_cut, listaSPND, emisor, params, dts, path, Ts):
    """
      Hace ajustes a un grupo de datos de SPND con el mismo emisor, para i/(phi*sigma*N). 
      Es importante que el nro de serie esté en los nombres de los archivos. 

    Parameters
    ----------
    t_cut : TYPE
        DESCRIPTION.
    listaSPND : lista
        Números de serie de SPNDs en el grupo.
    emisor : list
        [nombre del emisor, sec eficaz (cm-2)].
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
    emisor, sigma = emisor
    t_init, t_fin = t_cut
    P_tot = np.zeros((len(listaSPND), 2))
    P_err = np.zeros((len(listaSPND), 2))
    R2 = np.zeros(len(listaSPND))
    Res = []
    S_model = sens_model(emisor)
    for ii, spnd in enumerate(listaSPND):
        t, i = np.transpose(np.loadtxt(path.format('I', spnd)))
        t, i = t[t<=t_fin], i[t<=t_fin]
        t, i = t[t>=t_init], i[t>=t_init]
        
        i0 = np.loadtxt(path.format('I0', spnd))
        i0, i0err = (np.mean(i0), np.std(i0, ddof=1))
        
        i -= i0
        t = t*60 -dts[spnd]
        p_initial = param_iniciales(emisor, t , i*1e-12, [sigma, *params[spnd]], Ts[spnd])
        sens = i*1e-12/np.prod([sigma, *params[spnd]])
        popt, pcov = curve_fit(S_model, t, sens, p0=p_initial, sigma=np.full(len(i), i0err*1e-12)) #np.full(len(sens), i0_err*1e-12/(sigma*N0*flujos[ii]))
        Res.append(sens - sens_model(t, *popt))
        perr = np.sqrt(np.diag(pcov))
        ss_res = np.sum(Res[-1]**2)
        ss_tot = np.sum((sens-np.mean(sens))**2)
        R2[ii] = 1 - (ss_res / ss_tot)
        P_tot[ii] = popt
        P_err[ii] = perr
    P_tot, P_err, R2, Res = [{a:b for a, b in zip(listaSPND, xx)} for xx in [P_tot, P_err, R2, Res]]
    return (P_tot, P_err, R2, Res)

def LocalToMin(my_time):
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

def load_data(series, signal, path: str=os.getcwd(), version: str='old'):
    """
    Carga de datos.
    Es importante que los datos estén nombrados con el formato:{series}\{signal o descripcion}... .txt

    Parameters
    ----------
    series : str,
        Nombre del SPND. Ejemplos: V1934, DW9, A0
    signal : str, 
        Para version 'old', indicar qué señal se quiere ver de está medición. Ejemplos: Comp, V o Rh.
        Para la nueva versión, indicar las primeras palabras del nombre del archivo.
    path : str, 
        Ubicación absoluta de las carpetas con los datos.
    version : str, 
        Dice qué versión del software se usó para levantar la señal ('old', 'new').
        Default to 'old'.

    Returns
    -------
        Arrays de Tiempo absoluto (s), Corriente (pA).

    """
    version = version.lower()
    separador = {'old': ' ', 'new': '_'}
    nombret = {'old': 'T Local', 'new': 'T_Local'}
    try: 
        path = glob.glob(path+ r"/{}/{}*.txt".format(series, signal))[0]
        Data = pd.read_csv(path, sep='\t', encoding='unicode_escape')
        ifin = np.where(Data[nombret[version]] == 'FIN')[0]
        if ifin.size != 0:
            Data = Data[:ifin[0]]
        Tiempo = np.array([LocalToMin(s.split(separador[version])[1]) for s in np.array(Data[nombret[version]], dtype = str)])
    except:
        return(print('No hay datos de esa serie o señal. Revisar versión ingresada o el formato de guardado en T Local.'))
    if version == 'old':
        Corriente = np.array(Data['Corriente [pA]'])
        return (Tiempo, Corriente)
    elif version == 'new':
        Corriente_det = np.array(Data['CrudoDet'], dtype = np.float64())     
        Corriente_pat = np.array(Data['CrudoPatron'], dtype = np.float64())
        return (Tiempo, Corriente_det, Corriente_pat)
    else:
        return(print('Versión ingresada inválida.'))

def select_from_plot(data, labels, cant, fmt):
    """
    Devuelve los puntos seleccionados en una lista
    
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.
    cant : TYPE
        DESCRIPTION.
    fmt : TYPE
        DESCRIPTION.

    Returns
    -------
    xs : TYPE
        DESCRIPTION.

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
    
def data_cut(T, I, series, t_irr, Ts: float=1, version: str='old', abs_path: str=os.getcwd(), signal: str='patron', emisorpat: str='Rh'):
    """
    Corte de corriente a mano, a partir del gráfico.
    
    Parameters
    ----------
    T : ndarray,
        Array de datos crudos de tiempo.
    I : ndarray,
        Array de datos crudos de corriente.
    series: str,
        Serie del SPND, o nombre que se le da 
    t_ irr : float, seg
        Duración de irradiación
    Ts : float, seg
        Periodo de sampleo  
    abs_path:
        Ubicación absoluta de las carpetas con los datos.
    signal : str, 
        Indicar si es patrón o si es detector a comparar. Default to patrón.
    emisorpat : str, 
        Material emisor del SPND patrón. Default to Rh.

    Returns
    -------
    lista de arrays: 
        [t iniciado de 0, corriente, corriente de fuga].
    lista de: 
        t_diff, I_diff: variación de corriente por paso temporal [pA/seg] con el array de tiempo corrido
        
    """
    text = {'pat': '(t inicio fondo, t final fondo, t inicio irradiación)', 'det': '(t inicio fondo, t final fondo)'}
    nro_pts = {'pat': 3, 'det': 2}
    newpath = r'{}/Datos_cortados/Pos_cuts'.format(abs_path) 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    version = version.lower()
    signal = signal.lower()[:3]
    sys.stdout.write("\r Seleccione, en el siguiente orden: {}".format(text[signal]))
    x = select_from_plot([T, I], ['Tiempo [s]', 'Corriente [pA]'], nro_pts[signal], '.-')  
    sys.stdout.write('\x1b[2K') #deletes the last write
    t_sel = [round(xx[0]/Ts) for xx in x[0]]
    x01, x02 = t_sel[:2]
    I0 = I[x01:x02]
    if signal == 'pat':
        i1 = t_sel[2]
        dt = np.diff(T)
        dt[dt==0]= 1 # Elimino errores por repetición de T Local
        i2 = i1 + t_irr
        np.savetxt('{}/Pos_cuts_{}.txt'.format(newpath, series), [i1,i2])
    elif signal == 'det':
        try:
            xs = np.loadtxt("{}/Pos_cuts_{}.txt".format(newpath, series))
        except:
            return('No se encuentran datos')
        j1, j2 = xs #indices de dónde fue cortado iRh
        j1, j2 = [int(j1), int(j2)]
        if version == 'old':
            tRh, iRh = load_data(series, emisorpat, abs_path, version)
            i1, i2 = [np.where(T == tRh[j1])[0][0], np.where(T == tRh[j2])[0][0]]
            dt = T[0] - tRh[0]
            print(f'Diferencia temporal con patrón de {dt}, para {series}')
        elif version == 'new':
            i1, i2 = j1, j2
        else:
            print('Versión inválida')
    else:
        return print('Señal inválida')
    return([T[i1:i2]-T[i1], I[i1:i2], I0], [T[1:]-T[i1], np.diff(I)])

def data_cut_save(lista_cat, lista_emisores, listaSPND, lista_versiones, lista_Ts, lista_tirr, path_abs: str=os.getcwd()):
    """
    Sirve para un grupo de datos (misma tanda, o con nro de serie continuado, por ej.).
    Guarda la corriente en el intervalo de irradiación y la corriente de fuga, ambos en función del tiempo.
    No tiene en cuenta la diferencia!! (ver data_cut, no usa el criterio todavía)

    Parameters
    ----------
    lista_cat : list, str
        'Detector' o 'Patron', para cada SPND.
    lista_emisores : list, str
        Material emisor del SPND. Signal o señal en otras funciones.
    listaSPND : list, str
        Nros de series de los SPNDs.
    lista_versiones : list, str
        Con qué versión del programa fueron levantados.
    lista_Ts : list, float, s
        Periodo de sampleo, en segundos.
    lista_tirr : list, float, s
        Tiempos de irradiación, en segundos.
    path_abs : str
        Directorio en donde se encuentran las carpetas con los datos de la misma tanda.

    Returns
    -------
       None. Guarda los datos cortados para cada SPND, creando la carpeta si hace falta.

    """
    newpath = path_abs+'/Datos_cortados'
    emisores = list(set(lista_emisores))
    for emisor in emisores:
        for subpath in ['', '_I0', 'Diff']:
            newpath = r'{}/{}{}'.format(newpath, emisor, subpath) 
            if not os.path.exists(newpath):
                os.makedirs(newpath)
    for jj, spndno in enumerate(listaSPND):
        if lista_versiones[jj] == 'old':
            T, I = load_data(spndno, lista_emisores[jj], path_abs)
            print("\rProcesando {}{}".format(lista_emisores[jj], spndno))
            datoscortados, diff =  data_cut(T, I, spndno, lista_tirr[jj], lista_Ts[jj], 
                                            path_abs, lista_cat[jj], 
                                            lista_emisores[jj], lista_cat[jj])
            T_cut, I_cut, I0_cut = datoscortados
            try:
                np.savetxt('{}/{}/SPND{}_I_cut.txt'.format(newpath, lista_emisores[jj], spndno), np.transpose([T_cut, I_cut]), header='Tiempo[s] Corriente[pA]')
                np.savetxt('{}/{}_I0/SPND{}_I0_cut.txt'.format(newpath, lista_emisores[jj], spndno), np.transpose(I0_cut), header='Corriente[pA]')
                np.savetxt('{}/Diff/SPND{}_{}diff.txt'.format(newpath, spndno, lista_emisores[jj]), np.transpose(diff))
                print("\r{} de {} de la serie {} fue guardado.".format(lista_cat[jj], lista_emisores[jj], spndno))
                sys.stdout.write('\x1b[2K') 
            except:
                print("{} de {} de la serie {} fue guardado.".format(lista_cat[jj], lista_emisores[jj], spndno))
        elif lista_versiones[jj] == 'new':
            T, I, I_patron = load_data(spndno, lista_emisores[jj], path_abs, version='new')
            datoscortados, diff =  data_cut(T, I, spndno, lista_tirr[jj], lista_Ts[jj], 
                                            path_abs, lista_cat[jj], 
                                            lista_emisores[jj], lista_cat[jj])
            return (print())
        else:
            return (print('Versión inválida'))
