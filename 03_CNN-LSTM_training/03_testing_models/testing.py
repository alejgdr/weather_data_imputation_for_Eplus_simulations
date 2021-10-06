import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Model,Sequential
# from tensorflow.keras.optimizers import SGD,Adam,Adamax
# from tensorflow.keras.losses import MAPE,MSE
# from tensorflow.keras import losses
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras.metrics import MAPE,MAE,MeanAbsolutePercentageError,MeanSquaredError,RootMeanSquaredError
from datetime import datetime
from pickle import load, dump


def importa(archivo,nombres):
 esoru=pd.read_csv(archivo,names=nombres,skiprows=1)
 esoru.tiempo=pd.to_datetime(esoru.tiempo,format='%Y-%m-%d %H:%M:%S')
 esoru.set_index('tiempo',inplace=True)
 return(esoru)

def seasonal_exporta(archivo,predi,istep,in_size,rango,season_size,nombres,sol_data_correction=False,save=False,archivo_nombre='imputados_corregidos.csv'): #Sustituye datos de entrada por datos predecidos
     #nombres=['tiempo','Direct','Global','Difusa','Temperatura','Humedad','Viento','Presion','WDir_Avg','Rain_Tot']
     esoru=pd.read_csv(archivo,names=nombres,skiprows=1)
     #esoru.tiempo=pd.to_datetime(esoru.tiempo,format='%Y-%m-%d %H:%M:%S')
     #esoru.set_index('tiempo',inplace=True)
     esoru.Global.iloc[istep+season_size:istep+season_size+rango]=predi.copy() #agregar nueva columna 
     esoru.tiempo=pd.to_datetime(esoru.tiempo,format='%Y-%m-%d %H:%M:%S')
     esoru.set_index('tiempo',inplace=True)
     if (sol_data_correction==True):
        esoru=nightzero(esoru,archivo_nombre,save)
     return(esoru)

def seasonal_pre_process(in_size,out_size,esoru,scalerx,scalery,inputs,outputs,training_step,season_size): #when scaler is already fitted
     set_size=in_size+out_size
     esona=esoru.interpolate(method='polynomial',order=1)
     fecha1='2019-01-01'
     fecha2='2019-01-31'
     outna=esona[outputs]#[fecha1:fecha2]
     inpna=esona[inputs]#[fecha1:fecha2]
     print(inpna)
     train_val_ratio=1#.9 #Qué porcentaje serán los datos de entrenamiento y validación 
     train_ratio=1#.8 #Qué porcentaje serán los datos de solo entrenamiento 
     
     arresoru=np.array(outna)
     arresoruin=np.array(inpna)
     pre_array=[]
     pre_arrax=[]
     arresoruin=scalerx.transform(arresoruin)
     arresoru=scalery.transform(arresoru)
     for set_step in range (0,len(arresoruin)-season_size-out_size,training_step):
         x1=arresoruin[set_step:set_step+in_size]
         pre_arrax.append(x1)
         y=arresoru[set_step+season_size:set_step+season_size+out_size]
         pre_array.append(y)
     y_array,x_array=np.stack(pre_array),np.stack(pre_arrax)
     return (x_array,y_array)

def Multioneshot(esoru,forward_steps,out_size,in_size,istep,model,inputs,outputs,training_step,season_size,scalerx,scalery):
    x_array,y_array=seasonal_pre_process(in_size,out_size,esoru,scalerx,scalery,inputs,outputs,training_step,season_size)
#     pre_process(in_size,out_size,inputs,esoru,scaler,scaler2)
    output=[]
    target=[]
    for step in range (istep,istep+forward_steps,out_size):
        pry=model.predict(x_array[step].reshape(1,in_size,6))
        pry=scalery.inverse_transform(pry)
        tary=y_array[step]
        tary=scalery.inverse_transform(tary)
        output.append(pry)
        target.append(tary)
    predi=np.asarray(output,dtype='object').reshape(forward_steps)
    target=np.asarray(target,dtype='object').reshape(forward_steps)
#     output=np.asarray(output,dtype='object').reshape(forward_steps,1)
#     target=np.asarray(target,dtype='object').reshape(forward_steps,1)
#     output=output.reshape(forward_steps)
    return(predi,target) 

def alturaTMX(N,tiest):
    #(día juliano, tiempo estándar)
    lat=18.8397315*np.pi/180
    logloc=99.2364961
    logest=90
    #N=int(N)
    delta=23.45*np.pi/180*np.sin(((2*np.pi)/365)*(284+N))
    B=(N-1)*((2*np.pi)/365) 
    Et=229.2*(.000075+(.001868*np.cos(B))-(.032077*np.sin(B))-(.014615*np.cos(2*B))-(.04089*np.sin(2*B)))
    cenit=[]
    tsol=tiest+(4*(logest-logloc))+Et
    omega=.25*(tsol-720)
    theta=np.rad2deg(np.arccos((np.cos(lat)*np.cos(np.radians(omega))*np.cos(delta))+(np.sin(delta)*np.sin(lat))))
    altura=90-theta
    return(altura)

def nightzero(df,archivo_nombre,save=True):
    df['diajuliano']=df.index.dayofyear
    df['minutodia']=(df.index.hour*60)+df.index.minute
    df['alturasolar']=alturaTMX(df.diajuliano,df.minutodia)
    df.loc[df.alturasolar<0,'Global']=0
    df.loc[df.alturasolar<0,'Direct']=0
    df.loc[df.alturasolar<0,'Difusa']=0 
    dfcorr_noche=df[['Direct','Global','Difusa','Temperatura','Humedad','Presion']]
    if save==True:
        dfcorr_noche.to_csv('../../01_Documentos/02_preprocessed/'+archivo_nombre)
    return(dfcorr_noche)

def dfmetricas(impesoru,impesoru_target,model_name):
    dfrad=impesoru_target.copy()
    dfrad['prediccion']=impesoru.Global.copy().astype(float)
    dfrad['minutodia']=(dfrad.index.hour*60)+dfrad.index.minute
    dfrad['me']=(dfrad.prediccion-dfrad.Global).astype(float)
    dfrad['mae']=np.abs(dfrad.Global-dfrad.prediccion).astype(float)
    dfmingroup=dfrad.groupby(['minutodia',pd.Grouper(freq='1H')]).mean()
    dfmindia=dfmingroup.groupby(pd.Grouper(level='minutodia',axis=0)).mean()

    dfmindia.loc[dfmindia.alturasolar<0,'mae']=np.nan
    meandiay=dfmindia.mae.mean()
    
    dfsamp2=dfrad.resample('D').sum()
    dfsamp=dfrad.resample('D').mean()
    dfsamp['dif_energia_wh']=dfsamp2['me']/6
    dfsamp['dif_energia_wh_mae']=dfsamp2['mae']/6
    dfsamp['energia_wh/m2']=dfsamp2['Global']/6
    dfsamp['porcentaje_mae']=(dfsamp['dif_energia_wh_mae']/dfsamp['energia_wh/m2'])*100
    dfsamp['energia_predicha_wh/m2']=dfsamp2['prediccion']/6

    tablita=['model','dif_energia_wh','porcentaje_energia','porcentaje_mae','mae_de_día_promedio']
    
    dfsamp3=dfsamp.resample('Y').mean()
    dfsamp3['mae_de_día_promedio']=meandiay
    dfsamp3['dif_energia_wh']=-dfsamp3['energia_wh/m2']+dfsamp3['energia_predicha_wh/m2']
    dfsamp3['porcentaje_energia']=dfsamp3['dif_energia_wh']/dfsamp3['energia_wh/m2']*100
    dfsamp3['model']=model_name
    return(dfsamp3[tablita],dfsamp,dfrad)

def begin_table(infodf,cols_gen,path,nombre_archivo): #creates a new empty archive to store a dataframe, just use it once
    df=infodf
    df.to_csv(path+nombre_archivo)
    return(df)

def actualizar_bitacora(infodf,cols_gen,path,nombre_archivo): #adds a new row on a predetermined dataframe 
    df=pd.read_csv(path+nombre_archivo)
    #infodf=pd.DataFrame(data=info,columns=cols_gen)
    newdf=pd.concat([df,infodf])
    newdf=newdf.set_index('model')
    newdf.to_csv(path+nombre_archivo)
    return(pd.read_csv((path+nombre_archivo)))

def metricsamples(path,models,in_size,val_data_archivo):
    nombres1=['tiempo','Direct','Global','Difusa','Temperatura','Humedad','Presion','alturasolar','azimuth']
    esoru=importa(val_data_archivo,nombres1)
    out_size= 144 #model4
    #in_size, out_size =72,18 #model3
    dias_rango=360#350#=time_hor
    istep=0#49400-288#6400 
    scaler3=load(open('../../01_Documentos/03_Models/y_scalerv04.pkl','rb'))
    scaler4=load(open('../../01_Documentos/03_Models/x_scalerv04.pkl','rb'))
    fecha1='01/01/2019'#'01/10/2019'
    fecha2='31/12/2019'#'31/12/2019'
    rango=out_size*int((dias_rango*144)/out_size)
    inputs=['Global','Direct','Temperatura','Humedad','azimuth','alturasolar']
    model5=load_model(path+models)
    predi4,targ4=Multioneshot(fecha1,fecha2,esoru,rango,out_size,in_size,istep,model5,inputs,scaler2=scaler3,scaler=scaler4,prep=1)
    nombres=['tiempo','Direct','Global','Difusa','Temperatura','Humedad','Presion','alturasolar','azimuth']
    impesoru=exporta(val_data_archivo,predi4,istep,in_size,rango,nombres,sol_data_correction=True)
    impesoru_target=importa(val_data_archivo,nombres)
    yearly,daily,hourly=dfmetricas(impesoru,impesoru_target,models)
    return (yearly,daily,hourly)