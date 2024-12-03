#%%
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import pandas as pd
import chardet
import re
from glob import glob
from scipy.interpolate import interp1d
from uncertainties import ufloat, unumpy
from datetime import datetime,timedelta
import matplotlib as mpl
from uncertainties import ufloat, unumpy
from scipy.interpolate import CubicSpline,PchipInterpolator
#%%
#%% LECTOR RESULTADOS
def lector_resultados(path): 
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']
        
    # Leer las primeras 6 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                match = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                if match:
                    key = match.group(1)[2:]
                    value = float(match.group(2))
                    meta[key] = value
                else:
                    # Capturar los casos con nombres de archivo en las últimas dos líneas
                    match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                    if match_files:
                        key = match_files.group(1)[2:]  # Obtener el nombre de la clave sin '# '
                        value = match_files.group(2)     # Obtener el nombre del archivo
                        meta[key] = value
                    
    # Leer los datos del archivo
    data = pd.read_table(path, header=18,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)
        
    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)
   
    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)
    
    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,  Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N

#LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}
    
    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})  
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m
    
    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata

def calcular_hc(H, m):
    # Encuentra los índices donde m cruza el eje x (cambio de signo)
    # el error es la diferencia entre valor negativo y positivo
    cruces = np.where(np.diff(np.sign(m)) != 0)[0]

    hc_valores = []
    for i in cruces:
        # Interpolación lineal para encontrar el cruce exacto
        h1, h2 = H[i], H[i + 1]
        m1, m2 = m[i], m[i + 1]
        h_c = h1 - m1 * (h2 - h1) / (m2 - m1)
        hc_valores.append(h_c)
    
    # Obtén valores positivos y negativos
    hc_positivos = [h for h in hc_valores if h > 0]
    hc_negativos = [h for h in hc_valores if h < 0]

    # Calcula el promedio absoluto de los positivos y negativos
    if hc_positivos and hc_negativos:
        hc_promedio = (np.mean(hc_positivos) + abs(np.mean(hc_negativos))) / 2
        hc_err=    hc_valores[0]+hc_valores[1] 
    else:
        hc_promedio = None  # Si no hay suficientes cruces
    print('Hc = ', hc_promedio,hc_valores)
    return hc_promedio, hc_err


#%%

ciclos_100 = glob(os.path.join('resultados_100', '*ciclo_promedio*'),recursive=True)
ciclos_100.sort()
labels_100 = [os.path.split(s)[-1].split('bobN5')[-1].split('_')[0] for s in ciclos_100]

ciclos_97 = glob(os.path.join('resultados_97', '*ciclo_promedio*'),recursive=True)
ciclos_97.sort()
labels_97 = [os.path.split(s)[-1].split('bobN5')[-1].split('_')[0] for s in ciclos_97]

_,_,_,H100_1,M_100_1,meta_100_1=lector_ciclos(ciclos_100[0]) 
_,_,_,H100_2,M_100_2,meta_100_2=lector_ciclos(ciclos_100[1]) 
_,_,_,H100_3,M_100_3,meta_100_3=lector_ciclos(ciclos_100[2]) 
_,_,_,H100_4,M_100_4,meta_100_4=lector_ciclos(ciclos_100[3])

_,_,_,H97_1,M_97_1,meta_97_1=lector_ciclos(ciclos_97[0]) 
_,_,_,H97_2,M_97_2,meta_97_2=lector_ciclos(ciclos_97[1]) 
_,_,_,H97_3,M_97_3,meta_97_3=lector_ciclos(ciclos_97[2]) 
_,_,_,H97_4,M_97_4,meta_97_4=lector_ciclos(ciclos_97[3])
 
#%% divido por concentracion
concentracion_100 = np.array([meta_100_1['Concentracion_g/m^3'],
                             meta_100_2['Concentracion_g/m^3'],
                             meta_100_3['Concentracion_g/m^3']
                             ,meta_100_4['Concentracion_g/m^3']])/1000 #g/L == kg/m³ 

concentracion_97 = np.array([meta_97_1['Concentracion_g/m^3'],
                             meta_97_2['Concentracion_g/m^3'],
                             meta_97_3['Concentracion_g/m^3']
                             ,meta_97_4['Concentracion_g/m^3']])/1000 #g/L == kg/m³ 

m_100_1 = M_100_1/concentracion_100[0]
m_100_2 = M_100_2/concentracion_100[1]
m_100_3 = M_100_3/concentracion_100[2]
m_100_4 = M_100_4/concentracion_100[3]
print('Coercitivo 100')
Hc_100_1_mean,Hc_100_1_err=calcular_hc(H100_1,m_100_1)
Hc_100_2_mean,Hc_100_2_err=calcular_hc(H100_2,m_100_2)
Hc_100_3_mean,Hc_100_3_err=calcular_hc(H100_3,m_100_3)
Hc_100_4_mean,Hc_100_4_err=calcular_hc(H100_4,m_100_4)

Hc_100_1=ufloat(Hc_100_1_mean,Hc_100_1_err)
Hc_100_2=ufloat(Hc_100_2_mean,Hc_100_2_err)
Hc_100_3=ufloat(Hc_100_3_mean,Hc_100_3_err)
Hc_100_4=ufloat(Hc_100_4_mean,Hc_100_4_err)

m_97_1 = M_97_1/concentracion_97[0]
m_97_2 = M_97_2/concentracion_97[1]
m_97_3 = M_97_3/concentracion_97[2]
m_97_4 = M_97_4/concentracion_97[3]
print('Coercitivo 97')
Hc_97_1_mean,Hc_97_1_err=calcular_hc(H97_1,m_97_1)
Hc_97_2_mean,Hc_97_2_err=calcular_hc(H97_2,m_97_2)
Hc_97_3_mean,Hc_97_3_err=calcular_hc(H97_3,m_97_3)
Hc_97_4_mean,Hc_97_4_err=calcular_hc(H97_4,m_97_4)

Hc_97_1=ufloat(Hc_97_1_mean,Hc_97_1_err)
Hc_97_2=ufloat(Hc_97_2_mean,Hc_97_2_err)
Hc_97_3=ufloat(Hc_97_3_mean,Hc_97_3_err)
Hc_97_4=ufloat(Hc_97_4_mean,Hc_97_4_err)
#%% ploteo ciclos 

fig, (ax1,ax2)=plt.subplots(ncols=2,figsize=(12,5),constrained_layout=True,sharey=True,sharex=True)

ax1.plot(H100_1,m_100_1, label=labels_100[0]+ f' - {concentracion_100[0]:.2f} g/L')
ax1.plot(H100_3,m_100_3, label=labels_100[2]+ f' - {concentracion_100[2]:.2f} g/L')
ax1.plot(H100_4,m_100_4, label=labels_100[3]+ f' - {concentracion_100[3]:.2f} g/L')
ax1.plot(H100_2,m_100_2, label=labels_100[1]+ f' - {concentracion_100[1]:.2f} g/L')
ax1.set_title('100',fontsize=14)

axin1 = ax1.inset_axes([0.00, -1.0, 0.95,0.85],)
axin1.plot(H100_1,m_100_1,'.-', label=labels_100[0]+ f' - {Hc_100_1:.0f} A/m')
axin1.plot(H100_3,m_100_3,'.-', label=labels_100[2]+ f' - {Hc_100_3:.0f} A/m')
axin1.plot(H100_4,m_100_4,'.-', label=labels_100[3]+ f' - {Hc_100_4:.0f} A/m')
axin1.plot(H100_2,m_100_2,'.-', label=labels_100[1]+ f' - {Hc_100_2:.0f} A/m')
axin1.set_xlim(-80e2,80e2)           
axin1.set_ylim(-20,20)           

ax2.set_title('97',fontsize=14)
ax2.plot(H97_1,m_97_1, label=labels_97[0]+ f' - {concentracion_97[0]:.2f} g/L')
ax2.plot(H97_3,m_97_3, label=labels_97[2]+ f' - {concentracion_97[2]:.2f} g/L')
ax2.plot(H97_4,m_97_4, label=labels_97[3]+ f' - {concentracion_97[3]:.2f} g/L')
ax2.plot(H97_2,m_97_2, label=labels_97[1]+ f' - {concentracion_97[1]:.2f} g/L')

axin2 = ax2.inset_axes([0.00, -1.0, 0.95,0.85])
axin2.plot(H97_1,m_97_1,'.-',label=labels_97[0]+ f' - {Hc_97_1:.0f} A/m')
axin2.plot(H97_3,m_97_3,'.-',label=labels_97[2]+ f' - {Hc_97_3:.0f} A/m')
axin2.plot(H97_4,m_97_4,'.-',label=labels_97[3]+ f' - {Hc_97_4:.0f} A/m')
axin2.plot(H97_2,m_97_2,'.-',label=labels_97[1]+ f' - {Hc_97_2:.0f} A/m')
axin2.set_xlim(-80e2,80e2)           
axin2.set_ylim(-20,20)
axin1.axhline(0, color='black',linewidth=0.8,zorder=-2)  # Horizontal line (y=0)
axin1.axvline(0, color='black',linewidth=0.8,zorder=-2)  # Vertical line (x=0)    
axin2.axhline(0, color='black',linewidth=0.8,zorder=-2)  # Horizontal line (y=0)
axin2.axvline(0, color='black',linewidth=0.8,zorder=-2)  # Vertical line (x=0)    

ax1.indicate_inset_zoom(axin1, edgecolor="black")
ax2.indicate_inset_zoom(axin2, edgecolor="black")
for ai in [axin1,axin2]:
    ai.grid()
    ai.legend(title='Coercitivo',ncol=2,loc='upper center', bbox_to_anchor=(0.5, -0.1))

for a in [ax1,ax2]:
    a.legend(ncol=1)
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.set_ylabel('M/[NPM] (Am/kg)')
    
plt.savefig('comparativa_ciclos_100_97.png',dpi=400,facecolor='w')
# %%
