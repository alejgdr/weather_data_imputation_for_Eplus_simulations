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
from pickle import dump,load
import joblib
import sys

def importa(archivo,nombres): #Imports a time series, with columns=nombres
        esoru=pd.read_csv(archivo,names=nombres,skiprows=1)
        esoru.tiempo=pd.to_datetime(esoru.tiempo,format='%Y-%m-%d %H:%M:%S')
        esoru.set_index('tiempo',inplace=True)
        return (esoru)

def seasonal_pre_process(in_size,out_size,esoru,scaler,scaler2,inputs,outputs,training_step,season_size): #scaler fit is here #fits the scaler to the training data , scales  the data, and arrange it in arrays to feed the model
     set_size=in_size+out_size
     esona=esoru.interpolate(method='polynomial',order=1)
     outna=esona[output]#[0:int(144*30.98)]
     inpna=esona[inputs]#[0:int(144*30.98)]
     train_val_ratio=1#.9 #Qué porcentaje serán los datos de entrenamiento y validación 
     train_ratio=1#.8 #Qué porcentaje serán los datos de solo entrenamiento 
    
     arresoru=np.array(outna)
     arresoruin=np.array(inpna)
     scaler.fit(arresoruin[0:int(len(arresoruin)*train_ratio)])
     scaler2.fit(arresoru[0:int(len(arresoru)*train_ratio)])
     dump(scaler, open('03_scalers/x_scalerv08.pkl','wb'))
     dump(scaler2,open('03_scalers/y_scalerv08.pkl','wb'))
     pre_array=[]
     pre_arrax=[]
     arresoruin=scaler.transform(arresoruin)
     arresoru=scaler2.transform(arresoru)
     for set_step in range (0,len(arresoruin)-season_size-out_size,training_step):
         x1=arresoruin[set_step:set_step+in_size]
         pre_arrax.append(x1)
         y=arresoru[set_step+season_size:set_step+season_size+out_size]
         pre_array.append(y)
     x_array=np.stack(pre_arrax)
     y_array=np.stack(pre_array)
     return (scaler,scaler2,x_array,y_array)

def seasonal_pre_process2(in_size,out_size,esoru,scaler,scaler2,inputs,outputs,training_step,season_size): #when scaler is already fitted
#Scales the validation data and arrange it to feed the model
     set_size=in_size+out_size
     esona=esoru.interpolate(method='polynomial',order=1)
     outna=esona[outputs]
     inpna=esona[inputs]
     print(inpna)
     arresoru=np.array(outna)
     arresoruin=np.array(inpna)
     pre_array=[]
     pre_arrax=[]
     arresoruin=scaler.transform(arresoruin)
     arresoru=scaler2.transform(arresoru)
     for set_step in range (0,len(arresoruin)-season_size-out_size,training_step):
         x1=arresoruin[set_step:set_step+in_size]
         pre_arrax.append(x1)
         y=arresoru[set_step+season_size:set_step+season_size+out_size]
         pre_array.append(y)
     y_array,x_array=np.stack(pre_array),np.stack(pre_arrax)
     return (x_array,y_array)
    
def train_ann(x_train,y_train,x_val,y_val,epochs,in_size,out_size,bs):
    #global hist,run_epocs,training_time,initial_time,end_time
    initial_time=datetime.today()
    model=Sequential([
     Conv1D(filters=64,kernel_size=1,input_shape=[None,x_train.shape[2]],activation='relu'),
     MaxPool1D(pool_size=1,strides=1),
#      Conv1D(filters=128,kernel_size=1,activation='relu'),
#      MaxPool1D(pool_size=1,strides=1),
#      Conv1D(filters=256,kernel_size=1,activation='relu'),
#      MaxPool1D(pool_size=1,strides=1),
#      Conv1D(filters=128,kernel_size=1,activation='relu'),
#      MaxPool1D(pool_size=1,strides=1),
#      Conv1D(filters=64,kernel_size=1,activation='relu'),    
#      MaxPool1D(pool_size=1,strides=1),
        
#      LSTM(64,return_sequences=True),
#      LSTM(128,return_sequences=True),
#      LSTM(256,return_sequences=True),
#      LSTM(128,return_sequences=True),
     LSTM(64,return_sequences=False),
     Dropout(.1),
#      Dense(out_size,activation='sigmoid'),
#      Dense(out_size,activation='sigmoid'), 
     Dense(out_size,activation='sigmoid'), 
     ])
    print('in_size:',in_size)
    print('out_size:',out_size)
    model.compile(optimizer=opt,loss=MSE)
    hist=model.fit(x=x_train,y=y_train,batch_size=bs,validation_data=(x_val,y_val),epochs=epochs,callbacks=[cb])
    model.save('01_models/'+name_model+'.h5')
    print('Acabó y se guardó el entrenamiento... Guardando imagen de historial de entrenamiento' )
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.savefig('02_training_figures/'+name_model+'.png')
    run_epocs=len(hist.history['loss'])
    model.summary()
    end_time=datetime.today()
    training_time=end_time-initial_time
    return (model,hist,run_epocs,training_time,initial_time,end_time)
#*******************************************************************************************************
archivo='../../01_weather_data/02_cleaned_data/TMY_night_solar_angles.csv'
nombres1=['tiempo','Direct','Global','Difusa','Temperatura','Humedad','Presion','alturasolar','azimuth']
tmydf=importa(archivo,nombres1)
esoru=importa('../../01_weather_data/02_cleaned_data/esoru_night_angles.csv',nombres1)
#*******************************************************************************************************   
in_size=6
out_size=6
training_step=1
season_size=144
bs=int(sys.argv[1])
lr=1e-5
#****************************************************************************************************
#listminhora,listadias,lista_meses=lista_maker()
output=['Global']
inputs=['Global','Direct','Temperatura','Humedad','azimuth','alturasolar']
opt=Adam(lr)
cb=EarlyStopping(patience=7,restore_best_weights=True)
scaler=MinMaxScaler()
scaler2=MinMaxScaler()
initial_epochs=1000
name_model='train_seasonal_layers_1_1_1_out_s_'+str(out_size)+'in_s_'+str(in_size)+'_lr_'+str(lr)+'_'+(output[0])+'_BS'+str(bs)
#*******************************************************************************************************
print('batch_size:',bs)
print(name_model+'.h5'+ '\n')
print('Gotten inputs starting program')   
scaler,scaler2,x_train,y_train=seasonal_pre_process(in_size,out_size,tmydf,scaler,scaler2,inputs,output,training_step,season_size)
x_val,y_val=seasonal_pre_process2(in_size,out_size,esoru,scaler,scaler2,inputs,output,training_step,season_size)
print('preprocessing ended... starting training')
model,hist,run_epocs,training_time,initial_time,end_time=train_ann(x_train,y_train,x_val,y_val,initial_epochs,in_size,out_size,bs)
print('tiempo de entrenamiento: ',training_time)
print('épocas corridas: ',run_epocs)
print (name_model,training_time,run_epocs)
val_error=model.evaluate(x_val,y_val)
train_error=model.evaluate(x_train,y_train)
print('val_error:',val_error)
print('train_error:',train_error)