#Recordar correr 'pip install xlrd', cambiar la direccion de 
# $ pip install pandas
# $ pip install plotly
# $ pip install ipywidgets
# $ jupyter labextension install jupyterlab-plotly

import numpy as np
import pandas as pd
from datetime import datetime
import re

year = 365.24219878
tabla_RA3 = pd.read_excel('D:/Documentos RA-3/Copia de Listado de fuentes v13.xls',
                          sheet_name='Fuentes', index_col='Fuente')

tabla_iaea = {'Abundance': {'Cu': 0.6915, 'Au': 1}, 'Sec_eff': {'Cu': 4.50e-24, 'Au': 98.65e-24},
              't1/2': {'Cu': 45722, 'Au': 232770, 'Eu': 13.517}, 'Mr': {'Cu': 63.55, 'Au': 196.967},
              'Yield': {'Eu152': 3}}
              #abundancia %, seccion eff (n,g) [cm^-2], t1/2 del isótopo [s], Mr masa molar [g/mol]

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

class NAA_calib:
    def __init__(self, Fuente, FechaCalib):
        self.fuente = Fuente
        self.emisor = re.split('-|_| ', Fuente)[0]
        self.alldata = tabla_RA3.loc[Fuente]
        self.fecha_cal = datetime(*FechaCalib)
        self.fecha_doc = self.alldata['Fecha']
        dt_caldoc = float((self.fecha_cal - self.fecha_doc).days)
        self.act_doc = self.alldata[['Act       [Bq]', 'σ Act']].values.astype(float)
        self.act_cal = self.act_doc*np.exp(-np.log(2)*dt_caldoc/tabla_iaea['t1/2'][Fuente.split('_')[0]])
        
    def eficiencia(self, NetCounts, Energias, grado):
        ratios = 
        eff = NetCounts/(tabla_iaea['t1/2'][self.emisor]*self.act_cal*ratios)
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
