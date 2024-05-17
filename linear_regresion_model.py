import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Ułatwienie odczytu wyjść z NumPy
np.set_printoptions(precision=3, suppress=True)
  
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

# Pobieranie danych
raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
print(dataset.tail())
print('Kształt danych to:', dataset.shape)

# Usuwanie brakujących wierszy
dataset = dataset.dropna()

# Przekształcenie kolumny 'Origin'
dataset['Origin'] = dataset['Origin'].map({1: 'Poland', 2: 'Usa', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

print(dataset.head())

# Podział na zbiory treningowe i testowe
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Przygotowanie funkcji i etykiet
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

print("train_features.dtypes:", train_features.dtypes)
# Konwersja wszystkich danych na float32
train_features = train_features.astype(np.float32)
test_features = test_features.astype(np.float32)
train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

# Normalizacja
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

# Normalizacja przykładu
first = np.array(train_features[:1])
with np.printoptions(precision=2, suppress=True):
    print('First example:', first)
    print('Normalized:', normalizer(first).numpy())

# Model liniowy dla Horsepower
horsepower = np.array(train_features['Horsepower'])
horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

horsepower_model = tf.keras.Sequential([horsepower_normalizer, layers.Dense(units=1)])
horsepower_model.summary()

horsepower_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')

history = horsepower_model.fit(
    horsepower, train_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2
)

# Funkcja do wykresu straty
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(history)

# Ocena modelu
test_results = {}
test_results['horsepower_model'] = horsepower_model.evaluate(np.array(test_features['Horsepower']), test_labels, verbose=0)

x = tf.linspace(0.0, 300, 300)
y = horsepower_model.predict(x)

# Wykres predykcji dla Horsepower
def plot_horsepower(x, y):
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()
    plt.show()

plot_horsepower(x, y)

# Model regresji wielowymiarowej
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')

history = linear_model.fit(
    train_features, train_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2
)

plot_loss(history)
test_results['linear_model'] = linear_model.evaluate(test_features, test_labels, verbose=0)

# Model głębokiego uczenia z wieloma wejściami
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
    return model

dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
dnn_horsepower_model.summary()

history = dnn_horsepower_model.fit(
    horsepower, train_labels,
    validation_split=0.2,
    verbose=0,
    epochs=100
)

plot_loss(history)

x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)
plot_horsepower(x, y)

test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(np.array(test_features['Horsepower']), test_labels, verbose=0)

# Model DNN z wieloma wejściami
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0,
    epochs=100
)

plot_loss(history)
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

# Badanie wydajności modeli
print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)

# prognozy 
test_predictions = dnn_model.predict(test_features).flatten()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

#rozkład błędów prognozy

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')

#zapisywanie wyników
dnn_model.save('dnn_model')



