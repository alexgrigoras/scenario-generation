import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.backend import clear_session
from tensorflow.config.experimental import enable_op_determinism
from sklearn.preprocessing import MinMaxScaler

SEED = 42
set_random_seed(SEED)
enable_op_determinism()


class SimpleTransformerForecasting:
    def __init__(self, size_layer, embedded_size, output_size, num_heads=8, dropout_rate=0.1):
        print("Initializing transformer model")
        self.scaler = MinMaxScaler()
        self.model = self.build_model(size_layer, embedded_size, output_size, num_heads, dropout_rate)

    def build_model(self, size_layer, embedded_size, output_size, num_heads, dropout_rate):
        inputs = tf.keras.Input(shape=(1, output_size))     # Single time step with multiple features
        x = Dense(embedded_size, activation="relu")(inputs)
        x = Dropout(dropout_rate)(x)
        x = MultiHeadAttention(num_heads=num_heads, key_dim=size_layer)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + inputs)    # Residual connection
        outputs = Dense(output_size)(x[:, -1])              # Predict next step

        model = Model(inputs, outputs)
        return model

    def train(self, df_train, batch_size, epochs, learning_rate):
        clear_session()
        self.model.compile(optimizer=Adam(learning_rate), loss=Huber())

        df_train_scaled = self.scaler.fit_transform(df_train.reshape(-1, 1))

        early_stopping = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=1e-5)

        X_train = np.expand_dims(df_train_scaled[:-1], axis=1)  # Ensure correct shape
        y_train = df_train_scaled[1:]                           # Target values

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, reduce_lr], verbose=1)

        print("Training completed")

        
    def forecast(self, df_test, forecast_length):
        forecasted_values = []
        last_input = self.scaler.transform(df_test[-1:].reshape(-1, 1)).reshape(1, 1, -1)  # Reshape correctly

        for _ in range(forecast_length):
            pred = self.model.predict(last_input, verbose=0)
            forecasted_values.append(pred[0])
            last_input = pred.reshape(1, 1, -1)     # Update input for next prediction

        return self.scaler.inverse_transform(forecasted_values).ravel()