#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[3]:


tf.test.is_gpu_available(
  cuda_only=False, min_cuda_compute_capability=None
)


# In[4]:


import numpy as np
import pandas as pd
import tensorflow as tf
import os


# In[5]:


train_x = pd.read_csv(r"C:\Users\AYO IGE\Documents\Datasets\HAR\HAPT\HAPT Data Set\Train\X_train.txt", 
                     sep="\s+", header=None)


# In[6]:


list_of_features = list()
with open(r"C:\Users\AYO IGE\Documents\Datasets\HAR\HAPT\HAPT Data Set\features.txt") as filename:
    list_of_features = [line.strip() for line in filename.readlines()]
print("Total number of features: {}".format(len(list_of_features)))

train_x.columns = [list_of_features]


# In[7]:


train_y = pd.read_csv(r"C:\Users\AYO IGE\Documents\Datasets\HAR\HAPT\HAPT Data Set\Train\y_train.txt", 
                     sep="\s+", header=None, names=["ActivityLabel"], squeeze = True)


# In[8]:


train_ylabel_Mapping = train_y.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',                       4:'SITTING', 5:'STANDING',6:'LAYING', 7:'STAND_TO_SIT', 8:'SIT_TO_STAND',9:'SIT_TO_LIE',
                       10:'LIE_TO_SIT', 11:'STAND_TO_LIE', 12:'LIE_TO_STAND'})


# In[9]:


train_x.head()


# In[10]:


train_x = train_x.drop(['fBodyGyroJerkMag-Skewness-1'], axis=1)


# In[11]:


train_x.head()


# In[12]:


# Loading the test dataset 'test_x'
test_x = pd.read_csv(r"C:\Users\AYO IGE\Documents\Datasets\HAR\HAPT\HAPT Data Set\Test/X_test.txt", 
                    sep="\s+", header=None)


# In[13]:


test_x.columns = [list_of_features]


# In[14]:


test_x.head()


# In[15]:


test_x = test_x.drop(['fBodyGyroJerkMag-Skewness-1'], axis=1)


# In[16]:


test_x.head()


# In[17]:


test_y = pd.read_csv(r"C:\Users\AYO IGE\Documents\Datasets\HAR\HAPT\HAPT Data Set\Test/y_test.txt", 
                     sep="\s+", header=None, names=["ActivityLabel"], squeeze = True)


# In[18]:


test_ylabel_Mapping = test_y.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',                       4:'SITTING', 5:'STANDING',6:'LAYING', 7:'STAND_TO_SIT', 8:'SIT_TO_STAND',9:'SIT_TO_LIE',
                       10:'LIE_TO_SIT', 11:'STAND_TO_LIE', 12:'LIE_TO_STAND'})


# In[19]:


# We look for any duplicate/null values
print('No of duplicates in train: {}'.format(sum(train_x.duplicated())))
print('No of duplicates in test : {}'.format(sum(test_x.duplicated())))
print('We have {} NaN/Null values in train'.format(train_x.isnull().values.sum()))
print('We have {} NaN/Null values in test'.format(test_x.isnull().values.sum()))


# In[20]:


train_x.shape


# In[21]:


train_x.head()


# In[22]:


train_y.head()


# In[23]:


from sklearn.preprocessing import LabelEncoder


# In[24]:


label_encoder = LabelEncoder()
label_encoder.fit(train_y)

train_y = label_encoder.transform(train_y)
test_y = label_encoder.transform(test_y)


# In[25]:


from sklearn.preprocessing import StandardScaler


# In[26]:


scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)


# In[27]:


train_x = np.array(train_x)
test_x = np.array(test_x)
train_y = np.array(train_y)
test_y = np.array(test_y)


# In[28]:


from tensorflow.keras.utils import to_categorical
train_y = to_categorical(train_y, 12)
test_y = to_categorical(test_y, 12)
train_y.shape, test_y.shape


# In[29]:


train_x = np.expand_dims(train_x, axis=-1)
test_x = np.expand_dims(test_x, axis=-1)


# In[30]:


train_x.shape


# In[31]:


from keras.layers import GlobalAveragePooling1D, Reshape


# In[32]:


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, Activation, MaxPool1D, Flatten, Dropout, UpSampling1D, Reshape, GlobalAveragePooling2D, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard


# In[33]:


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# In[34]:


def SqueezeAndExcitation(inputs, ratio=8):
    b, c = (560, 1)
    x = GlobalAveragePooling1D()(inputs)
    x = Dense(c//ratio, activation="relu", use_bias=False)(inputs)
    x = Dense(c, activation="sigmoid", use_bias=False)(x)
    x = Multiply()([inputs, x])
    return x


# In[35]:


def build_autoencoder(shape, dim=128):
    inputs = Input(shape)

    x = inputs
    num_filters = [128, 64, 32]
    kernel_size = [9, 7, 5]
    for i in range(len(num_filters)):
        nf = num_filters[i]
        ks = kernel_size[i]

        x = Conv1D(nf, ks, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool1D((2))(x)
    
    b, f, n = x.shape
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Reshape((512,1), input_shape=(512,))(x)
    x = SqueezeAndExcitation(x)
    x = Flatten()(x)
    latent = Dense(128, activation="linear", name="LATENT")(x)
    #x = Dense(512, activation="relu")(latent)
    x = Dense(f*n, activation="relu")(latent)
    x = Reshape((f, n))(x)
    
    
    num_filters = [32, 64, 128]
    kernel_size = [5, 7, 9]
    
    for i in range(len(num_filters)):
        nf = num_filters[i]
        ks = kernel_size[i]
        
        x = UpSampling1D((2))(x)
        x = Conv1D(nf, ks, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    
    x = Conv1D(shape[1], 1, padding="same")(x)
    
    model = Model(inputs, x)
    return model


# In[36]:


""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Create a folder to save files """
create_dir("files")

""" Hyperparameters """
batch_size = 32
num_epochs = 100
input_shape = (560, 1)
num_classes = 12
latent_dim = 128
lr = 1e-4

model_path = "files/model_autoencoder.h5"
csv_path = "files/log_autoencoder.csv"

""" Dataset """
print(f"Train: {train_x.shape}/{train_y.shape} - Test: {test_x.shape}/{test_y.shape}")

""" Adding noise to training dataset """
mu, sigma = 0, 0.1  
noise = np.random.normal(mu, sigma, train_x.shape)

train_x1 = train_x + noise

""" Model & Training """
autoencoder = build_autoencoder(input_shape, dim=latent_dim)
autoencoder.summary()
adam = tf.keras.optimizers.Adam(lr)
autoencoder.compile(loss='mse', metrics=['accuracy'], optimizer=adam)
callbacks = [
    ModelCheckpoint(model_path, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
]
autoencoder.fit(train_x1, train_x,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks = callbacks
    )


# In[37]:


def build_classifier(autoencoder, num_classes=12):
    inputs = autoencoder.input
    outputs = autoencoder.get_layer("LATENT").output
    x = Dense(num_classes, activation="softmax")(outputs)
    
    model = Model(inputs, x)
    return model


# In[38]:


""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Create a folder to save files """
create_dir("files")

""" Hyperparameters """
batch_size = 128
num_epochs = 100
input_shape = (560, 1)
num_classes = 12
latent_dim = 128
lr = 1e-4

model_path = "files/model_classifier.h5"
csv_path = "files/log_classifier.csv"

""" Model & Training """
classifier = build_classifier(autoencoder, num_classes=num_classes)
classifier.summary()
adam = tf.keras.optimizers.Adam(lr)
classifier.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
callbacks = [
    ModelCheckpoint(model_path, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
]
histo = classifier.fit(train_x, train_y,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(test_x, test_y),
    callbacks = callbacks
    )


# In[39]:


get_ipython().system('pip install mlxtend')


# In[40]:


import matplotlib.pyplot as plt


# In[41]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix


# In[42]:


""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Model: Weight file """
model_path = "files/model_classifier.h5"
model = load_model(model_path)

""" Evalution """
y_pred = np.argmax(model.predict(test_x), axis=-1)
y_true = np.argmax(test_y, axis=1)

acc = accuracy_score(y_true, y_pred, normalize=True)
print(f"Accuracy: {acc}")

mat = confusion_matrix(y_true, y_pred)
print(mat)


# In[43]:


from sklearn.metrics import accuracy_score, classification_report


# In[44]:


print(classification_report(y_true,y_pred))


# In[45]:


def plot_learningCurve(history, epochs):
  #accuracy
  epoch_range = range(1, epochs+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc ='upper left')
  plt.show()
#validaion loss
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc ='upper left')
  plt.show()


# In[46]:


plot_learningCurve(histo, 22)


# In[ ]:


from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score


# In[ ]:


mat = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(conf_mat = mat, show_normed=True,  figsize=(5,5),  )


# In[ ]:


from sklearn.metrics import f1_score


# In[ ]:


f1_score(y_true, y_pred, average = "weighted")


# In[ ]:




