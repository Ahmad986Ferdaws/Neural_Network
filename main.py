import numpy as np
import pandas as pd
from matplotlib omport pyplot as plt

data= pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df.head()

data = np.array(df)
image_train = np.array(data[:, 1:])
image_label = np.array(data[:,:1])

image_train = np.reshape(image_train, (-1,28,28,1))

image_train = image_train /255
image_label = tf.keras.utils.to_categorical(image_label , 10)

model = tf.keras.Sequential([
  tf.keras.layers.Input( shape=( 28,28,1)),
  
