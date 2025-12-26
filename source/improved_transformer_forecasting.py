import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Input
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import set_random_seed
from tensorflow.config.experimental import enable_op_determinism
from tensorflow.keras.utils import register_keras_serializable

SEED = 42
set_random_seed(SEED)
enable_op_determinism()


class ImprovedTransformerForecasting:
    def __init__(self, sequence_length=None, size_layer=64, embedded_size=64, output_size=1, num_heads=8, dropout_rate=0.1):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.output_size = output_size
        self.config = {
            "size_layer": size_layer,
            "embedded_size": embedded_size,
            "num_heads": num_heads,
            "dropout_rate": dropout_rate
        }

    def transformer_block(self, x, size_layer, num_heads, dropout_rate):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=size_layer)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        ffn_output = Dense(size_layer, activation='relu')(x)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        return LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    def build_model(self, size_layer, embedded_size, output_size, num_heads, dropout_rate):
        inputs = tf.keras.Input(shape=(None, output_size))
        x = Dense(embedded_size, activation='relu')(inputs)
        x = PositionalEncoding(d_model=embedded_size)(x)

        for _ in range(2):  # stacked Transformer blocks
            x = self.transformer_block(x, size_layer, num_heads, dropout_rate)

        x = Dense(embedded_size, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

        x = x[:, -1, :]     # use last time step's representation
        outputs = Dense(output_size)(x)

        return tf.keras.Model(inputs, outputs)

    def train(self, df_train, batch_size, epochs, learning_rate):
        clear_session()

        df_train = df_train.reshape(-1, 1)

        if self.sequence_length is None or self.sequence_length > len(df_train) - 1:
            self.sequence_length = len(df_train) - 1

        df_scaled = self.scaler.fit_transform(df_train)

        X_train, y_train = [], []
        for i in range(self.sequence_length, len(df_scaled)):
            X_train.append(df_scaled[i - self.sequence_length:i])
            y_train.append(df_scaled[i])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        self.model = self.build_model(**self.config, output_size=self.output_size)
        self.model.compile(optimizer=Adam(learning_rate), loss=Huber())

        early_stopping = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=1e-5)

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    callbacks=[early_stopping, reduce_lr], verbose=1)
    
    def forecast(self, df_test, forecast_length):
        if df_test.ndim == 1:
            df_test = df_test.reshape(-1, 1)

        forecasted_values = []

        window = self.sequence_length or df_test.shape[0]
        input_chunk = df_test[-window:]

        if input_chunk.shape[0] < window:
            pad = np.repeat(input_chunk[[0]], window - input_chunk.shape[0], axis=0)
            input_chunk = np.vstack([pad, input_chunk])

        last_input = self.scaler.transform(input_chunk)

        for _ in range(forecast_length):
            input_seq = last_input[-window:].reshape(1, window, -1)
            pred = self.model.predict(input_seq, verbose=0)
            forecasted_values.append(pred[0])
            last_input = np.vstack([last_input, pred])

        return self.scaler.inverse_transform(forecasted_values).ravel()

@register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000, **kwargs):
        super().__init__(**kwargs)
        position = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        angle_rads = position * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)
        self.d_model = d_model

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model
        })
        return config