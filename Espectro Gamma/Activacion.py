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

year = 365.24219878
s_day = 86400 

tabla_RA3 = pd.read_excel('D:/Documentos RA-3/Copia de Listado de fuentes v13.xls',
                          sheet_name='Fuentes', index_col='Fuente')

tabla_iaea = {'Abundance': {'Cu': 0.6915, 'Au': 1}, 'Sec_eff': {'Cu': 4.50e-24, 'Au': 98.65e-24},
              't1/2': {'Cu': 45722, 'Au': 232770, 'Eu': 13.517}, 'Mr': {'Cu': 63.55, 'Au': 196.967}}
              #abundancia %, seccion eff (n,g) [cm^-2], t1/2 del isótopo [s], Mr masa molar [g/mol]

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
    
def poly(grado, an, x):
    exps = np.arange(grado+1)
    return np.array([np.dot(an[::-1], xx**exps) for xx in x])
    
def ajuste_pol(grado, xdata, ydata, y_err=None):
    try:
        weight = 1/y_err
    except:
        weight = y_err
    coef, pcov = np.polyfit(xdata, ydata, deg = grado, w= weight, cov=True)
    yfit = poly(grado, coef, xdata)
    residuals = ydata - yfit
    perr = np.sqrt(np.diag(pcov))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yfit-np.mean(yfit))**2)
    R2 = 1 - (ss_res / ss_tot)
    try: 
        chi2 = (1/(len(residuals)-2))*np.sum((ydata*residuals*weight)**2)
    except:
        chi2 = 0
    return coef, perr, R2, chi2, residuals

def lc_pd_dataframe(url):
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
    return pd.read_csv(urllib.request.urlopen(req))

class loadfromIAEA:
    def __init__(self, nuclei, state, radiation_type: str='g', only_stable: bool=True):
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
        self.emisor = re.split('-|_| ', Fuente)[0]
        self.datafromIAEA = loadfromIAEA(self.emisor, 'decay').data
        self.datafromRA3 = tabla_RA3.loc[Fuente]
        self.fecha_cal = datetime(*FechaCalib)
        self.fecha_doc = self.datafromRA3['Fecha']
        dt_caldoc = float((self.fecha_doc - self.fecha_cal).days)*s_day
        intensity, unc_int, energy, hl = self.datafromIAEA.get(['intensity', 'unc_i', 'energy', 'half_life_sec']).values.T
        self.halflife_s = np.mean(hl)
        self.act_doc = self.datafromRA3[['Act       [Bq]', 'σ Act']].values.astype(float)
        self.act_cal = self.act_doc*np.exp(np.log(2)*dt_caldoc/self.halflife_s)
        self.act_cal[1] = self.act_cal[1]*self.act_cal[0]
        self.int_energy = np.transpose([energy, intensity])
        self.int_err = unc_int #np.array([np.mean(hl), np.mean(self.datafromRA3.get('unc_hls'))])
        # self.dt_caldoc = dt_caldoc
    def cal_eff(self, NetCounts, Energias, grado_pol, tlive, NetCounts_err= None, criterio: float=0):
        i_sel = self.int_energy[:, 1]>criterio
        I_E_IAEA = np.array(self.int_energy[i_sel])
        I_err_sel = np.array(self.int_err[i_sel])
        energy_sel = np.zeros(len(Energias))
        intensity_sel = np.zeros(len(Energias))
        int_err_sel = np.zeros(len(Energias))
        for jj, en in enumerate(Energias):
            for n, ee in enumerate(I_E_IAEA):
                x = en/ee[0]
                if 0.99<x<1.01:
                    intensity_sel[jj] = ee[1]/100
                    int_err_sel[jj] = I_err_sel[n]/100
                    energy_sel[jj] = ee[0]
        eff = NetCounts/(tlive*self.act_cal[0]*intensity_sel)
        try:
            eff_err = eff*np.sqrt((NetCounts_err/NetCounts)**2 + (int_err_sel*self.act_cal[0])**2 + (self.act_cal[1]*intensity_sel)**2)
                                  #np.exp(np.log(2)*self.dt_caldoc/self.halflife_s[0])*np.log(2))
        except:
            eff_err = None
        coef, perr, R2, chi2, res = ajuste_pol(grado_pol, np.log(Energias), np.log(eff), y_err=eff_err)
        return eff, eff_err, coef, perr, R2, chi2, np.array([energy_sel, intensity_sel]).T, res
 
