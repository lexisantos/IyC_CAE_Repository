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

def lc_pd_dataframe(url):
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
    return pd.read_csv(urllib.request.urlopen(req))

df = lc_pd_dataframe(Livechart + "fields=decay_rads&nuclides=152eu&rad_types=g")

for col in list(df.columns)[:4]:
    df = df[pd.to_numeric(df[col],errors='coerce').notna()]

df.intensity = df['intensity'].astype(float)

plt.scatter(df['energy'][df["intensity"]>2], df['intensity'][df["intensity"]>2]) # plot in log scale
plt.xlabel('Energy [keV]')
plt.ylabel('Intensity %')

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
