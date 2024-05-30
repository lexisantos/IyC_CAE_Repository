#Recordar correr 'pip install xlrd', cambiar la direccion de 
# pip install pandas
# pip install plotly
# pip install ipywidgets
# # jupyter labextension install jupyterlab-plotly

import numpy as np
import pandas as pd
from datetime import datetime
import re
import urllib.request

day_year = 365.24219878
s_day = 86400 

tabla_RA3 = pd.read_excel('D:/Documentos RA-3/Copia de Listado de fuentes v13.xls',
                          sheet_name='Fuentes', index_col='Fuente')

Mr = {'198Au': 44.01, }

Livechart = "https://nds.iaea.org/relnsd/v1/data?"

def redondeo(mean, err, cs, texto = False):
    """
    Devuelve al valor medio con la misma cant. de decimales que el error (con 2 c.s.).
    """
    digits = -np.floor(np.log10(err)).astype(int)+cs-1
    if err<1:
        err_R = format(np.round(err, decimals = digits), f'.{digits}f')
        mean_R = str(np.round(mean, decimals = len(err_R)-2))
    else:
        err_R = format(np.round(err, decimals = digits), '.0f')
        mean_R = format(np.round(mean, decimals = cs-1-len(err_R)), '.0f')
    if texto == True:
        return (mean_R, '±',err_R)
    else:
        return (float(mean_R), float(err_R))
       
def ajuste_pol(grado, xdata, ydata, y_err=None):
    try:
        weight = 1/y_err
    except:
        weight = y_err
    coef, pcov = np.polyfit(xdata, ydata, deg = grado, w= weight, cov=True)
    yfit = np.polyval(coef, xdata)
    residuals = ydata - yfit
    perr = np.sqrt(np.diag(pcov))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yfit-np.mean(yfit))**2)
    R2 = 1 - (ss_res / ss_tot)
    try: 
        chi2 = (1/(len(residuals)-1))*np.sum(residuals**2)
    except:
        chi2 = 0
    return coef, perr, R2, chi2, residuals

def lc_pd_dataframe(url):
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
    return pd.read_csv(urllib.request.urlopen(req))

def select_data(measured, data_table, tolerance, idx=0): 
    """
    Es como un VLOOKUP

    Parameters
    ----------
    measured : TYPE
        DESCRIPTION.
    data_table : TYPE
        DESCRIPTION.
    tolerance : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    rows, cols = len(measured), len(data_table.T)
    data_sel = np.zeros((rows, cols-1))
    for jj, en in enumerate(measured):
        for ee in data_table:
            x = en/ee[idx]
            if 1-tolerance<x<1+tolerance:
                data_sel[jj] = np.delete(ee, idx)
    return data_sel

def dif_rel(x1, x2):
    return (2*(x1 - x2)/(x1 + x2))*100

def Incognita(Fuente, Energias, cps_peaks, err_cps, poly_params, dt,
              tolerancia: float=0.01):
    data_iaea = loadfromIAEA(Fuente, 'decay')
    data_doc = tabla_RA3.loc[Fuente][['Act       [Bq]', 'σ Act']].values.astype(float)
    data_doc[1] = data_doc[1]*data_doc[0]
    data_cal = data_doc*np.exp(np.log(2)*dt/data_iaea.get(['half_life_sec']).to_numpy().mean())
    eiu = data_iaea.get(['energy', 'intensity', 'unc_i']).to_numpy()
    peaks = select_data(Energias, eiu, tolerancia)/100
    peaks[np.isnan(peaks)] = 0
    effs = np.exp(np.polyval(poly_params, np.log(Energias)))
    Act_calc = cps_peaks/(peaks[:, 0]*effs)
    Err_calc = Act_calc*np.sqrt((err_cps/cps_peaks)**2 + (peaks[:, 1]/peaks[:, 0])**2)
    diff = dif_rel(Act_calc, data_cal[0])
    return np.array([Act_calc, Err_calc]).T, data_cal, diff

class loadfromIAEA:
    def __init__(self, Fuente, state, radiation_type: str='g', only_stable: bool=True):
        nuclei = re.split('-|_| ', Fuente)[0]
        path = {'decay': f"fields=decay_rads&nuclides={nuclei}&rad_types={radiation_type}",
                'estable': f"fields=ground_states&nuclides={nuclei}"}
        df = lc_pd_dataframe(Livechart + path[state.lower()])
        self.estado = state
        self.nucleo = nuclei
        if state.lower() == 'decay' and only_stable == True:
            self.data = df.query("p_energy==0") 
        else:
            self.data = df
    def get(self, cols):
        df = self.data[cols]
        df = df[df[cols].notna()]
        return df

class NAA_calib:
    def __init__(self, Fuente, FechaCalib):
        self.fuente = Fuente
        self.datafromIAEA = loadfromIAEA(Fuente, 'decay').data
        self.datafromRA3 = tabla_RA3.loc[Fuente]
        self.fecha_cal = datetime(*FechaCalib)
        self.fecha_doc = self.datafromRA3['Fecha']
        self.dt_caldoc = float((self.fecha_doc - self.fecha_cal).days)*s_day
        intensity, unc_int, energy, hl = self.datafromIAEA.get(['intensity', 'unc_i', 'energy', 'half_life_sec']).to_numpy().T
        self.halflife_s = hl.mean()
        self.act_doc = self.datafromRA3[['Act       [Bq]', 'σ Act']].values.astype(float)
        self.act_doc[1] = self.act_doc[1]*self.act_doc[0]
        self.act_cal = self.act_doc*np.exp(np.log(2)*self.dt_caldoc/self.halflife_s)
        self.int_energy = np.transpose([energy, intensity, unc_int])
        #np.array([np.mean(hl), np.mean(self.datafromRA3.get('unc_hls'))])
        # self.dt_caldoc = dt_caldoc
    def cal_eff(self, NetCounts, Energias, grado_pol, tlive, 
                NetCounts_err= None, criterio: float=0, tolerancia: float=0.01):
        i_sel = self.int_energy[:, 1]>criterio*100
        I_E_IAEA = np.array(self.int_energy[i_sel])
        data_sel = select_data(Energias, I_E_IAEA, tolerancia)/100  
        eff = NetCounts/(tlive*self.act_cal[0]*data_sel[:, 0])
        try:
            eff_err = eff*np.sqrt((NetCounts_err/NetCounts)**2 + (data_sel[:, 1]/data_sel[:, 0])**2 
                                  + (self.act_cal[1]/self.act_cal[0])**2)
                                  #np.exp(np.log(2)*self.dt_caldoc/self.halflife_s[0])*np.log(2))
        except:
            eff_err = None
        coef, perr, R2, chi2, res = ajuste_pol(grado_pol, np.log(Energias), np.log(eff), y_err=eff_err/eff)
        self.eff_params = {'an': coef, 'an_err': perr, 'R-squared': R2, 'chi-square': chi2,
                           'residuals': res, 'grado_pol': grado_pol}
        return eff, eff_err, coef, perr, R2, chi2, data_sel, res
