from random import random
import tensorflow as tf 
import numpy as np
from Dataset import Dataset
import heapq
import math
import matplotlib.pyplot as plt 
import sklearn.model_selection

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

class NeuMF(object):
    def __init__(self, num_users, num_items,model_layers, reg_mf, reg_mlp, k):
        self.num_users = num_users
        self.num_items = num_items
        self.k = k 
        self.model_layers = model_layers
        self.reg_mf = reg_mf
        self.reg_mlp = reg_mlp
    
    def get_model(self):
        #define input is a 1-d array
        
        user_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='user_input')
        item_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='item_input')

        #defile the way to initialize the keras layer
        embedding_init = tf.keras.initializers.RandomNormal(stddev=0.01)
        mf_embedding_user = tf.keras.layers.Embedding(
            self.num_users, 
            self.k,
            embeddings_initializer=embedding_init,
            embeddings_regularizer=tf.keras.regularizers.l2(self.reg_mf),
            input_length=1,
            name='mf_embedding_user'
            )
        mf_embedding_item = tf.keras.layers.Embedding(
            self.num_items, 
            self.k,
            embeddings_initializer=embedding_init,
            embeddings_regularizer=tf.keras.regularizers.l2(self.reg_mf),
            input_length=1,
            name='mf_embedding_item'
            )
        mlp_embedding_user = tf.keras.layers.Embedding(
            self.num_users, 
            self.model_layers[0]//2,
            embeddings_initializer=embedding_init,
            embeddings_regularizer=tf.keras.regularizers.l2(self.reg_mlp[0]),
            input_length=1,
            name='mlp_embedding_user'
            )
        mlp_embedding_item = tf.keras.layers.Embedding(
            self.num_items, 
            self.model_layers[0]//2,
            embeddings_initializer=embedding_init,
            embeddings_regularizer=tf.keras.regularizers.l2(self.reg_mlp[0]),
            input_length=1,
            name='mlp_embedding_item'
            )

        #GMF latent
        mf_user_latent = tf.keras.layers.Flatten()(mf_embedding_user(user_input))
        mf_item_latent = tf.keras.layers.Flatten()(mf_embedding_item(item_input))
        gmf_layer = tf.keras.layers.multiply([mf_user_latent, mf_item_latent])

        #MLP latent
        mlp_user_latent = tf.keras.layers.Flatten()(mlp_embedding_user(user_input))
        mlp_item_latent = tf.keras.layers.Flatten()(mlp_embedding_item(item_input))
        mlp_layer = tf.keras.layers.concatenate([mlp_user_latent, mlp_item_latent])

        num_layers = len(self.model_layers)
        for i in range(1, num_layers):
            new_layer = tf.keras.layers.Dense(
                self.model_layers[i],
                kernel_regularizer=tf.keras.regularizers.l2(self.reg_mlp[i]),
                activation='relu',
                name = 'layer%d' %i
            )
            mlp_layer = new_layer(mlp_layer)
            #new_layer = tf.keras.layers.Dropout(0.4)
            #mlp_layer = new_layer(mlp_layer)    
            #new_layer = tf.keras.layers.BatchNormalization()
            #mlp_layer = new_layer(mlp_layer)
        
        NeuMF_layer = tf.keras.layers.concatenate([gmf_layer, mlp_layer])
        prediction = tf.keras.layers.Dense(
            1,
            kernel_initializer='lecun_uniform',
            activation='sigmoid',
            name='prediction'
        )(NeuMF_layer)

        model = tf.keras.models.Model(inputs=[user_input, item_input], outputs = prediction)
        return model

    def init_from_pretrain_model(self, model, gmf_model, mlp_model, model_layers):
        pass 

    def evalueate_model(self, model, test_data, test_negatives):
        hits = []
        ndcgs = []
        for i in range(len(test_data)):
            hr, ndcg = self.evaluate_one_sample(model, i, test_data, test_negatives)
            hits.append(hr)
            ndcgs.append(ndcg)
        return hits, ndcgs 

    def evaluate_one_sample(self, model, idx, test_data, test_negatives):
        ratings = test_data[idx]
        items = test_negatives[idx]
        user = ratings[0]
        item = ratings[1]
        user_item_predict = {}
        items.append(item)
        users = np.full(len(items), user, dtype='int32')
        predictions = model.predict(
            [users, np.array(items)],
            batch_size = 128,
            verbose = 0
        )
        for j in range(len(items)):
            itm = items[j]
            user_item_predict[itm] = predictions[j]
        items.pop()
        ranklist = heapq.nlargest(10, user_item_predict, key=user_item_predict.get)
        hr = self.get_hit_ratio(ranklist, item)
        ndcg = self.get_ndcg(ranklist, item)
        return hr, ndcg

    def get_hit_ratio(self, ranklist, item):
        for itm in ranklist:
            if itm == item:
                return 1
        return 0
    
    def get_ndcg(self, ranklist, item):
        for j in range(len(ranklist)):
            itm = ranklist[j]
            if itm == item:
                return math.log(2) / math.log(j + 2)
        return 0

if __name__ == '__main__':
    print('loading data from Dataset...')
    dataset = Dataset()
    num_users = dataset.num_users   
    num_items = dataset.num_items   
    X_train = dataset.X_train
    Y_train = dataset.Y_train 
    X_test= dataset.X_test
    Y_test = dataset.Y_test
    X_val = dataset.X_val
    Y_val = dataset.Y_val
    test_data = dataset.test_positives
    test_negatives = dataset.test_negatives
    print('build model...')
    model_layers = [64, 32, 16, 8]
    reg_mf = 0
    reg_mlp = [0, 0, 0, 0]
    k = 8
    learning_rate = 0.0005
    ncf = NeuMF(num_users, num_items, model_layers, reg_mf, reg_mlp, k)
    model = ncf.get_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())

    '''
    hits, ndcgs = ncf.evalueate_model(model, test_data, test_negatives)
    hr = np.array(hits).mean()
    ndcg = np.array(ndcgs).mean()
    '''

    print('start training...')
    batch_size = 512
    best_hr = 0
    best_ndcg= 0
    #history = LossHistory()
    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=60,
        validation_data=(X_val, Y_val),
        shuffle=True
    )
    '''
    hits, ndcgs = ncf.evalueate_model(model, test_data, test_negatives)
    hr = np.array(hits).mean()
    ndcg = np.array(ndcgs).mean()
    print('%d-th epoch: hr = %.3f, ndcg = %.3f' % (i, hr, ndcg))
    '''
    iterations=range(len(history.history['loss']))
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(iterations, history.history['loss'], label='Training loss')
    plt.plot(iterations, history.history['val_loss'], label='Validation loss')
    plt.title('Traing and Validation loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(iterations, history.history['acc'], label='Training acc')
    plt.plot(iterations, history.history['val_acc'], label='Validation acc')
    plt.title('Traing and Validation acc')
    plt.legend()
    plt.show()

    '''
    
    '''