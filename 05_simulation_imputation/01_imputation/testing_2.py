import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,LSTM,Input,SimpleRNN,GRU,Conv1D,MaxPool1D,Conv2D,MaxPool2D,Dropout
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import SGD,Adam,Adamax
from tensorflow.keras.losses import MAPE,MSE
from tensorflow.keras import losses
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MAPE,MAE,MeanAbsolutePercentageError,MeanSquaredError,RootMeanSquaredError
from datetime import datetime
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from pickle import load, dump


def importa(archivo,nombres):
 esoru=pd.read_csv(archivo,names=nombres,skiprows=1)
 esoru.tiempo=pd.to_datetime(esoru.tiempo,format='%Y-%m-%d %H:%M:%S')
 esoru.set_index('tiempo',inplace=True)
 return(esoru)

def importa2(archivo):
 nombres=['tiempo','Direct','Global','Difusa','Temperatura','Humedad','Presion']
 esoru=pd.read_csv(archivo,names=nombres,skiprows=1)
 esoru.tiempo=pd.to_datetime(esoru.tiempo,format='%Y-%m-%d %H:%M:%S')
 esoru.set_index('tiempo',inplace=True)
 return(esoru)

def exporta(archivo,predi,istep,rango,nombres,path_exported_file='exported_data.csv',save=False): #Sustituye datos de entrada por datos predecidos
     #nombres=['tiempo','Direct','Global','Difusa','Temperatura','Humedad','Viento','Presion','WDir_Avg','Rain_Tot']
     esoru=pd.read_csv(archivo)
     #esoru.tiempo=pd.to_datetime(esoru.tiempo,format='%Y-%m-%d %H:%M:%S')
     #esoru.set_index('tiempo',inplace=True)
     esoru.Ig.iloc[istep:istep+rango]=predi[:rango] #agregar nueva columna 
     esoru.time=pd.to_datetime(esoru.time,format='%Y-%m-%d %H:%M:%S')
     esoru.set_index('time',inplace=True)
     if (save==True):
        esoru.to_csv(path_exported_file)
     return(esoru)

def pre_process(in_size,out_size,inputs,output,esoru,scaler,scaler2):
     #global scaler,scaler2,x_train,y_train,x_val,y_val,x_test,y_test #quitar las variables globales
     set_size=in_size+out_size
     esona=esoru.interpolate(method='polynomial',order=1)
     outna=esona[output]#.loc[fecha1:fecha2]                    #Ver si puedo escalar desde esona
     inpna=esona[inputs]#.loc[fecha1:fecha2]
     #train_val_ratio=.9 #Qué porcentaje serán los datos de entrenamiento y validación 
     #train_ratio=.8 #Qué porcentaje serán los datos de solo entrenamiento 
     
     arresoru=np.array(outna)
     arresoruin=np.array(inpna)
     #scaler.fit(arresoruin[0:int(len(arresoruin))])#*train_ratio)])
     #scaler2.fit(arresoru[0:int(len(arresoru))])#*train_ratio)])
     pre_array=[]
     pre_arrax=[]
     arresoruin=scaler.transform(arresoruin)
     arresoru=scaler2.transform(arresoru)
     for set_step in range (len(arresoruin)-set_size):
         x1=arresoruin[set_step:set_step+set_size-out_size]
         pre_arrax.append(x1)
         y=arresoru[set_step+set_size-out_size:set_step+set_size]
         pre_array.append(y)
     y_array,x_array=np.array(pre_array),np.array(pre_arrax)
     return (x_array,y_array)

def time_prep(esona,listahoras,listadias,lista_meses):
    meses=pd.get_dummies(esona.index.month)
    dias=pd.get_dummies(esona.index.day)
    horas=pd.get_dummies(esona.index.hour)
    lista_meses=['Jan','Feb','Mar','Apr','May','Jun','Jul','Ago','Sept','Oct','Nov','Dic']
    meses.columns=lista_meses
    horas.columns=listahoras
    dias.columns=listadias
    dfdumt=pd.concat([meses,dias,horas],axis=1)#,ignore_index=True)
    dfdumt=dfdumt.set_index(esona.index)
    df_time=pd.concat([esona,dfdumt],axis=1)
    return(df_time)

def Multioneshot(esoru,rango,out_size,in_size,istep,model,inputs,output,scaler,scaler2):
    x_array,y_array=pre_process(in_size,out_size,inputs,output,esoru,scaler,scaler2)
    prediction=[]
    y=[]
    for x in range(istep,istep+rango,out_size):#4371
        #target=np.concatenate((y_array[x:x+1]),axis=0)
        target=y_array[x:x+1].reshape(out_size,1)
        target=scaler2.inverse_transform(target)
        #pr=model3.predict(x_array[x:x+out_size])[:, :, np.newaxis]
        pr=model.predict(x_array[x:x+1])#[:, :, np.newaxis]
        pr=scaler2.inverse_transform(pr)
        #predict=np.concatenate((np.nan,pr[0]),axis=0)
        prediction.append(pr[0])
        y.append(target)
    predi=np.array(prediction)
    targ=np.array(y)
    
    fig, ax=plt.subplots(figsize=(10,5))
    ax.plot(predi.reshape(out_size),'g-',label='predicción')
    ax.plot(targ.reshape(out_size),'k-',label='target')
    ax.legend()
    ax.set_ylabel('Global radiation [W/m^2]')
    ax.set_xlabel('Time [10min]')
    predi, targ= predi.reshape(out_size),targ.reshape(out_size)
    return(predi,targ) 

def metrics(predi,targ,rango):
    print('Para '+ str(rango)+ ' pasos evaluados')
    mse=np.mean((predi.reshape(rango)-targ.reshape(rango))**2)
    rmse=np.sqrt(mse)
    mae=np.mean(np.abs(predi.reshape(rango)-targ.reshape(rango)))
    print('mse:',mse)
    print('rmse:',rmse)
    print('mae:',mae)
    return(mse,rmse,mae)

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
def declinacionTMX(N):
    #(día juliano, tiempo estándar)
    lat=18.8397315*np.pi/180
    logloc=99.2364961
    logest=90
    #N=int(N)
    delta=23.45*np.pi/180*np.sin(((2*np.pi)/365)*(284+N))
    #B=(N-1)*((2*np.pi)/365) 
    #Et=229.2*(.000075+(.001868*np.cos(B))-(.032077*np.sin(B))-(.014615*np.cos(2*B))-(.04089*np.sin(2*B)))
    #cenit=[]
    #tsol=tiest+(4*(logest-logloc))+Et
    #omega=.25*(tsol-720)
    #theta=np.rad2deg(np.arccos((np.cos(lat)*np.cos(np.radians(omega))*np.cos(delta))+(np.sin(delta)*np.sin(lat))))
    #altura=90-theta
    return(delta)

def nightzero_timeprep2(df,archivo_nombre,save=True): #No corrige datos, solo agrega a df altura y declinación solar
    df['diajuliano']=df.index.dayofyear
    df['minutodia']=(df.index.hour*60)+df.index.minute
    df['alturasolar']=alturaTMX(df.diajuliano,df.minutodia)
    df['declinacion']=declinacionTMX(df.diajuliano)
    #tmxnoche.loc[tmxnoche.alturasolar<0,'prediccion']=0
    #df.loc[df.alturasolar<0,'Global']=0
    #df.loc[df.alturasolar<0,'Direct']=0
    #df.loc[df.alturasolar<0,'Difusa']=0 #La difusa también?
    dfcorr_noche=df[['Direct','Global','Difusa','Temperatura','Humedad','Presion','alturasolar','declinacion']]
    if save==True:
        dfcorr_noche.to_csv('../../01_Documentos/02_preprocessed/'+archivo_nombre)
    return(dfcorr_noche)

def nightzero(df,archivo_nombre,save=True):
    df['diajuliano']=df.index.dayofyear
    df['minutodia']=(df.index.hour*60)+df.index.minute
    df['alturasolar']=alturaTMX(df.diajuliano,df.minutodia)
    #tmxnoche.loc[tmxnoche.alturasolar<0,'prediccion']=0
    df.loc[df.alturasolar<0,'Global']=0
    df.loc[df.alturasolar<0,'Direct']=0
    df.loc[df.alturasolar<0,'Difusa']=0 #La difusa también?
    dfcorr_noche=df[['Direct','Global','Difusa','Temperatura','Humedad','Presion']]
    if save==True:
        dfcorr_noche.to_csv('../../01_Documentos/02_preprocessed/'+archivo_nombre)
    return(dfcorr_noche)