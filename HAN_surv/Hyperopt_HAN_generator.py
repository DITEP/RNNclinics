

### directories
out_file = 'HAN_choffe_256_generator'
data_dir="data/"
results_dir="results/"
scripts_dir="."

print("welcome to", out_file)

#libraries

import numpy as np
import pandas as pd
import pickle
import keras
import os
from nltk.tokenize import sent_tokenize
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
import copy
from gensim.models import Word2Vec
from hyperopt import STATUS_OK, tpe, hp, Trials, fmin
import logging
from sklearn.preprocessing import StandardScaler
from math import sqrt

os.chdir(scripts_dir)
from han_model import HAN
from han_model import AttentionLayer


###############################
# setting parameters
###############################

params = {
    'MAX_WORDS_PER_SENT': 30,
    'MAX_SENT': 30,
    'MAX_WORDS': 3,
    'MAX_SEQ_LEN':512,
    'embedding_dim': 128,
    'word_encoding_dim': 256,
    'sentence_encoding_dim': 256,
    'MAX_EVALS': 10,  # number of models to evaluate with hyperopt
    'l1':0,
    'l2':0,
    'dropout':0.2,
    'Nepochs' : 5,
    'lr':0.001,
    #'batch_size' : 100,
    'Nbatches': 256,  #Number of smaller batches for pretraining
}


#####################
# load data
#####################
print("load data")

df = pd.read_csv(data_dir+"train_choffe.csv")
#texts = df["0"]
dates = df["Unnamed: 0"].values
dates = dates.reshape(len(dates),1)
scaler=StandardScaler()
scaler=scaler.fit(dates)
normalized=scaler.transform(dates)
normalized.reshape(len(normalized))
dates=normalized
dates=dates.reshape(len(dates))
#texts=texts.values

X=np.load(data_dir+"X_choffe.npy")

embedding_matrix=np.load(data_dir+"embedding_matrix.npy")

#########################
#split train and test
#########################
print("split")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,dates, test_size=0.2, random_state=42)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.1,random_state=2)

#####################
#create data generator
#####################
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, labels, batch_size=32, n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.X = X
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        return self.X[indexes], self.labels[indexes]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


#####################
#create model
#####################
 
i=0
def create_model(params):
    global i
    i += 1
    l2 = params['l2']
    l1= params['l1']
    dropout = params['dropout']
    lr = params['lr']
    
    han_model = HAN(params['MAX_WORDS_PER_SENT'], params['MAX_SENT'], 1, embedding_matrix, params['word_encoding_dim'], params['sentence_encoding_dim'], l1, l2, dropout)
    han_model.summary()
    optimizer = optimizers.Adam( lr )
    han_model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer=optimizer)
    
    my_callbacks = [
      tf.keras.callbacks.ModelCheckpoint(filepath=results_dir+out_file+'model.{epoch:02d}-{val_loss:.2f}.h5')
    ]

    train_generator = DataGenerator(X_train,y_train,batch_size=params['Nbatches'])
    val_generator = DataGenerator(X_val,y_val,batch_size=params['Nbatches'])
    
    han_model.fit_generator(generator=train_generator,
                    validation_data=val_generator,
                    use_multiprocessing=True,
                    epochs=params["Nepochs"],
                    workers=6,
                    callbacks=my_callbacks)
      
    
    scores1 = han_model.evaluate(X_test, y_test, verbose=0)
    print (scores1)
    y_pred_num = han_model.predict(X_test)  
    d={'loss':[scores1[0]], 'acc':[scores1[1]]}
    df_scores=pd.DataFrame(data=d)
    print(df_scores)
    df_params=pd.DataFrame.from_dict([params])
    #print(df_params)
    df_new=df_params.join(df_scores)
    #print(df_new)
    df_results=pd.read_csv(results_dir+out_file+'results_all2.csv')
    df_results=df_results.append(df_new)
    df_results.to_csv(results_dir+out_file+'results_all2.csv', index=False)
    han_model.save(results_dir+out_file+str(i)+'.hd5')
    return {'loss': scores1[0], 'acc': scores1[1], 'params': params, 'status': STATUS_OK}



space = {
    'l1': hp.qloguniform('l1', np.log(0.00001), np.log(0.1), 0.0001),
    'l2': hp.qloguniform('l2', np.log(0.00001), np.log(0.1), 0.0001),
    'dropout': hp.quniform('dropout', 0, 0.2, 0.5),
    'lr': hp.qloguniform('lr', np.log(0.00001),  np.log(0.001), 0.00001)
}

print("train hyperopt")

# Trials object to track progress
bayes_trials = Trials()
try:
    with open(results_dir+out_file+'results_all2.csv',"r") as f:
        df_results=f.read()
except FileNotFoundError:
    df_results = pd.DataFrame(columns=('loss','acc','l1','l2','dropout','lr'))
    df_results.to_csv(results_dir+out_file+'results_all2.csv', index=False)
    
best = fmin(fn=create_model(params), space=space, algo=tpe.suggest, max_evals=params['MAX_EVALS'], trials=bayes_trials)
print(best)

