from os import path, makedirs

from sklearn.linear_model import LinearRegression, Ridge
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint

# # Create Models

# In the following three cells we are creating three ML models

# In[1]:


def create_LR(df_train_X, df_train_Y):
    """
    In this cell we are creating and training a linear regression model
    df_train_X: contains all available features excluding the features related to 'task_channel' we aim to predict (train)
    df_train_Y: contains all available features related to 'task_channel' only for the train


    return: trained linear regression model
    """
    lr_model = LinearRegression()
    lr_model.fit(df_train_X.values, df_train_Y.values)
    return lr_model


# In[2]:


def create_Ridge(df_train_X, df_train_Y):
    """
    In this cell we are creating and training a ridge regression model


    df_train_X: contains all available features excluding the features related to 'task_channel' we aim to predict (train)
    df_train_Y: contains all available features related to 'task_channel' only for the train

    return: trained ridge regression model
    """
    ridge_model = Ridge()
    ridge_model.fit(X=df_train_X.values, y=df_train_Y.values)
    return ridge_model


# In[3]:


def create_model_dnn(task_channel, df_train_X, df_train_Y, test_plate):
    """
    In this cell we are creating and training a multi layer perceptron (we refer to it as deep neural network, DNN) model

    task_channel: the current channel that we aim to predict
    df_train_X: contains all available features excluding the features related to 'task_channel' we aim to predict (train)
    df_train_Y: contains all available features related to 'task_channel' only for the train
    test_plate: the ID of a given plate. This information assist us while printing the results.

    return: trained dnn model
    """
    folder = 'dnn_models'
    makedirs(folder, exist_ok=True)

    # Stracture of the network#
    inputs = Input(shape=(df_train_X.shape[1],))
    dense1 = Dense(512, activation='relu')(inputs)
    dense2 = Dense(256, activation='relu')(dense1)
    dense3 = Dense(128, activation='relu')(dense2)
    dense4 = Dense(100, activation='relu')(dense3)
    dense5 = Dense(50, activation='relu')(dense4)
    dense6 = Dense(25, activation='relu')(dense5)
    dense7 = Dense(10, activation='relu')(dense6)
    predictions = Dense(df_train_Y.shape[1], activation='sigmoid')(dense7)

    # model compilation
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam', loss='mse')

    # model training
    test_plate_number = test_plate[:5]
    filepath = path.join(folder, f'{test_plate_number}_{task_channel}.h5')
    my_callbacks = [
        ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,
                        mode='auto', period=1)]
    model.fit(df_train_X, df_train_Y, epochs=5, batch_size=1024 * 8, verbose=0, shuffle=True, validation_split=0.2,
              callbacks=my_callbacks)
    return model
