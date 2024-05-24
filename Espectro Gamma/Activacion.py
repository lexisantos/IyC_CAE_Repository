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
import matplotlib.pyplot as plt

year = 365.24219878
s_day = 86400 

tabla_RA3 = pd.read_excel('D:/Documentos RA-3/Copia de Listado de fuentes v13.xls',
                          sheet_name='Fuentes', index_col='Fuente')

tabla_iaea = {'Abundance': {'Cu': 0.6915, 'Au': 1}, 'Sec_eff': {'Cu': 4.50e-24, 'Au': 98.65e-24},
              't1/2': {'Cu': 45722, 'Au': 232770, 'Eu': 13.517}, 'Mr': {'Cu': 63.55, 'Au': 196.967}}
              #abundancia %, seccion eff (n,g) [cm^-2], t1/2 del isótopo [s], Mr masa molar [g/mol]

Livechart = "https://nds.iaea.org/relnsd/v1/data?"

def poly(grado, an, x):
    exps = np.arange(grado)
    return np.sum(an[::-1]*x**exps)
    
def ajuste_pol(grado, xdata, ydata, y_err=None):
    weight = None if y_err ==None else 1/y_err
    coef, pcov = np.polyfit(xdata, ydata, deg = grado, w= weight, cov=True)
    yfit = poly(grado, coef, xdata)
    residuals = ydata - yfit
    perr = np.sqrt(np.diag(pcov))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yfit-np.mean(yfit))**2)
    R2 = 1 - (ss_res / ss_tot)
    chi2 = 0 if y_err == None else (1/(len(residuals)-2))*np.sum((ydata*residuals/y_err)**2)
    return yfit, coef, perr, R2, chi2

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
        df = df[df.notna()]
        return [df[col] for col in cols]

class NAA_calib:
    def __init__(self, Fuente, FechaCalib):
        self.fuente = Fuente
        self.emisor = re.split('-|_| ', Fuente)[0]
        self.datafromIAEA = loadfromIAEA(self.emisor, 'decay')
        self.datafromRA3 = tabla_RA3.loc[Fuente]
        self.fecha_cal = datetime(*FechaCalib)
        self.fecha_doc = self.datafromRA3['Fecha']
        dt_caldoc = float((self.fecha_doc - self.fecha_cal).days)
        intensity, unc_int, energy, hl = self.datafromIAEA.get(['intensity', 'unc_i', 'energy', 'half_life_sec'])
        hl_d = np.mean(hl)/s_day
        self.act_doc = self.datafromRA3[['Act       [Bq]', 'σ Act']].values.astype(float)
        self.act_cal = self.act_doc*np.exp(np.log(2)*dt_caldoc/hl_d)
        self.act_cal[1] = self.act_cal[1]*self.act_cal[0]
        self.int_energy = np.transpose([energy, intensity])
        self.int_err = unc_int
        self.halflife_s = np.mean(hl)
    def eficiencia(self, NetCounts, Energias, grado_pol, criterio: float=0):
        i_sel = self.int_energy[:, 1]>criterio
        I_E_IAEA = np.array(self.int_energy[i_sel])
        intensity_sel = np.zeros(len(Energias))
        for jj, en in enumerate(Energias):
            for ee in I_E_IAEA:
                x = en/ee[0]
                if 0.99<x<1.01:
                    intensity_sel[jj] = ee[1] 
        eff = NetCounts/(self.halflife_s*self.act_cal[0]*intensity_sel)
        return eff

class NAA_flujo:
    def __init__(self, composition, irradiation_time, postirr_time, live_time):
        self.td = postirr_time
        self.ti = irradiation_time
        self.tlive = live_time
        self.comp = composition #{'El1': %, 'El2': %}
    
    def DDA(self, real_time):
        halflife = np.array([tabla_iaea['t1/2'][mat] for mat in list(self.comp.keys())])
        f = np.log(2)*(real_time/halflife)
        corr = f/(1-np.exp(-f))
        return corr

    def N_padres(self, masa):
        m_parcial = np.array([self.comp[mat] for mat in list(self.comp.keys())])
        Mr = np.array([tabla_iaea['Mr'][mat] for mat in list(self.comp.keys())])
        N = 6.022e23*m_parcial/Mr
        return N
     

def calc_act(mat, Espectros, comp, param, nmed):
    '''
    

    Parameters
    ----------
    mat : str
        Material del que se obtuvo un espectro. Por ej., W17.
    Espectros : dict
        Picos dependiendo el isótopo. Informa la energía del pico y su intensidad en %.
    comp : list, str
        Nombres de los picos a ver de Espectros.
    eff_ord : list, float
        Ordenada al origen de la calib de eff, con su error.
    eff_pend : list, float
        Pendiente de la calib de eff, con su error.

    Returns
    -------
    Acts_m : dict
        Actividad media, entre todas las mediciones hechas, por cada isótopo, con su error por prop. de errores.

    '''
    datos = pd.read_csv('Cuentas Alambres W\datos_{}.txt'.format(mat), sep='\t')
    datos['Medicion'] = datos['Medicion'].astype(str)
    datos = datos.set_index(['Medicion'])
    Acts = {}
    treals = {}
    for i, el in enumerate(comp):
        datos_med = datos.loc[el]
        Acts[el] = np.zeros((nmed, 2))
        treals[el] = np.zeros(nmed)
        for j in range(nmed):
            try:
                E, Net, Net_err, tlive, treal = datos_med[datos_med.keys()[[0, 3, 4, 5, 6]]].iloc[j] #selecciono los datos 
            except:
                E, Net, Net_err, tlive, treal = datos_med[datos_med.keys()[[0, 3, 4, 5, 6]]]
            eff, eff_err = eficiencia(E, *param[:, 0])*np.array([1, np.sqrt(np.sum((param[:, 1]*np.log(E)**np.arange(len(param[:, 0])))**2))])
            Acts[el][j] = np.array([1, np.sqrt((Net_err/Net)**2 + (eff_err/eff)**2)])*Act_E(Net, eff, tlive, Espectros[el][1])
            treals[el][j] = treal
        # Acts_m[el] = np.array([np.mean(Acts[el][:,0]), np.sqrt(np.std(Acts[el][:,0], ddof=1)**2 + (1/N**2)*np.sum(Acts[el][:,1]**2))])
    return Acts, treals


 #np.polyfit
