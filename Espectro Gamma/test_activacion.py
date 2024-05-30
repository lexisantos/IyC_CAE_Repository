import matplotlib.pyplot as plt
from Codigos_py.Repositorio import Activacion as metact

path_patron = 'D:/Calibracion SPND/Calibracion A0 27may24/Eu152_Net_Energy_ROI_27may24.txt'
path_incog = 'D:/Calibracion SPND/Calibracion A0 27may24/Na22_Incog_Net_Energy_ROI_27may24.txt' #Cs: 'D:/Calibracion SPND/Calibracion A0 27may24/Cs137_Incog_Net_Energy_ROI_27may24.txt'
Fuente_pat = 'Eu152_76044A-440'
Fuente_incog = 'Na22_76048-440'#'Cs137_76071-440'

tlive_pat = 83789.06
tlive_incog = 6287.82 #497.96 Cs
tlive_fondo = 86400

fecha_cal = [2024, 5, 27]

#%% Llamo tablas de la IAEA 

E_GV, Net_GV, Net_GV_err, Fondo, Fondo_err = metact.np.loadtxt(path_patron, delimiter='\t', skiprows=1).T
Net_GV, NetGV_err = (Net_GV-Fondo, metact.np.sqrt(Net_GV_err**2 + Fondo_err**2))

datos_cal = metact.NAA_calib(Fuente_pat, fecha_cal)

#%% Cal de eficiencia

grado_pol = 3
eff_data, err_eff, coef, coef_err, R2, chi2, data_sel, residuals = datos_cal.cal_eff(Net_GV, E_GV, grado_pol, tlive_pat, criterio = 0.025, NetCounts_err = Net_GV_err)

for jj in range(len(coef)):
    print(f'a{grado_pol-jj} = '+ ' '.join([*metact.redondeo(coef[jj], coef_err[jj], 2, texto=True)])+ f' keV^-{grado_pol-jj}')
print('χr2 = {}'.format(chi2))
print('R2 = {}'.format(R2))
print('Válido en E = ({}-{}) keV'. format(E_GV.min(), E_GV.max()))

E_arr = metact.np.linspace(E_GV.min(), E_GV.max(), num=1000)
logE_arr  = metact.np.log(E_arr)
eff_fit = metact.np.polyval(coef, logE_arr)

plt.figure(1, figsize = (5,5))
plt.errorbar(metact.np.log(E_GV), metact.np.log(eff_data), yerr= err_eff/eff_data, fmt='.')
# plt.errorbar(E_GV, eff_data, yerr= err_eff, fmt='.')
plt.plot(logE_arr, eff_fit, label = 'fit, grado {}'.format(grado_pol))
plt.grid(True, ls = '--')
plt.ylabel('$ln$ Eff')
plt.xlabel('$ln$ E')
plt.tight_layout()
plt.legend()

# plt.figure(2, figsize = (5,5))
# plt.hist(residuals, label = 'hist., grado {}'.format(grado_pol))
# plt.grid(True, ls = '--')
# plt.xlabel('Residuals')
# plt.ylabel('Densidad de prob.')
# plt.tight_layout()
# plt.legend()

#%% Cal de Act y E de pico incógnita

E_x, Net_x, Net_x_err, Fondo, Fondo_err = metact.np.loadtxt(path_incog, delimiter='\t', skiprows=1, ndmin=2).T

Net_cps, Err_cps = (Net_x/tlive_incog - Fondo/tlive_fondo, metact.np.sqrt((Net_x_err/tlive_incog)**2 + (Fondo_err/tlive_fondo)**2))

Act_calc, Act_tab, dif = metact.Incognita(Fuente_incog, E_x, Net_cps, Err_cps, coef, datos_cal.dt_caldoc)
