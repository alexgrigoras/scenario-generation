import numpy as np
import pandas as pd
from tqdm import tqdm
from math import sqrt
from datetime import timedelta
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 
tf.compat.v1.random.set_random_seed(1234)
tf.compat.v1.disable_eager_execution()

class TransformerForecasting:
    def __init__(self):
        print("Init transformer")

    def calculate_accuracy(self, real, predict):
        real = np.array(real) + 1
        predict = np.array(predict) + 1
        percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
        return percentage * 100

    def anchor(self, signal, weight):
        buffer = []
        last = signal[0]
        for i in signal:
            smoothed_val = last * weight + (1 - weight) * i
            buffer.append(smoothed_val)
            last = smoothed_val
        return buffer

    def forecast(self, df_train, scaler, num_layers, size_layer, batch_size, epoch, dropout_rate, forecast_length, learning_rate):
        tf.compat.v1.reset_default_graph()
        modelnn = self.Attention(size_layer, size_layer, learning_rate, df_train.shape[1], df_train.shape[1], dropout_rate=dropout_rate)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        date_ori = pd.to_datetime(df_train.iloc[:, 0]).tolist()
        len_export = forecast_length

        pbar = tqdm(range(epoch), desc = 'train loop')
        for i in pbar:
            total_loss, total_acc = [], []
            for k in range(0, df_train.shape[0] - 1, batch_size):
                index = min(k + batch_size, df_train.shape[0] - 1)
                batch_x = np.expand_dims(
                    df_train.iloc[k : index, :].values, axis = 0
                )
                batch_y = df_train.iloc[k + 1 : index + 1, :].values
                logits, _, loss = sess.run(
                    [modelnn.logits, modelnn.optimizer, modelnn.cost],
                    feed_dict = {
                        modelnn.X: batch_x,
                        modelnn.Y: batch_y
                    },
                ) 
                total_loss.append(loss)
                total_acc.append(self.calculate_accuracy(batch_y[:, 0], logits[:, 0]))
            pbar.set_postfix(cost = np.mean(total_loss), acc = np.mean(total_acc))

        output_predict = np.zeros((df_train.shape[0] + forecast_length, df_train.shape[1]))
        output_predict[0] = df_train.iloc[0]
        upper_b = (df_train.shape[0] // batch_size) * batch_size

        for k in range(0, (df_train.shape[0] // batch_size) * batch_size, batch_size):
            out_logits = sess.run(
                modelnn.logits,
                feed_dict = {
                    modelnn.X: np.expand_dims(
                        df_train.iloc[k : k + batch_size], axis = 0
                    )
                },
            )
            output_predict[k + 1 : k + batch_size + 1] = out_logits

        if upper_b != df_train.shape[0]:
            out_logits = sess.run(
                modelnn.logits,
                feed_dict = {
                    modelnn.X: np.expand_dims(df_train.iloc[upper_b:], axis = 0)
                },
            )
            output_predict[upper_b + 1 : df_train.shape[0] + 1] = out_logits
            forecast_length -= 1
            date_ori.append(date_ori[-1] + timedelta(days = 1))
        
        for i in range(forecast_length):
            o = output_predict[-forecast_length - batch_size + i:-forecast_length + i]
            out_logits = sess.run(
                modelnn.logits,
                feed_dict = {
                    modelnn.X: np.expand_dims(o, axis = 0)
                },
            )
            output_predict[-forecast_length + i] = out_logits[-1]
            date_ori.append(date_ori[-1] + timedelta(days = 1))
        
        output_predict = scaler.inverse_transform(output_predict)
        deep_future = self.anchor(output_predict[:, 0], 0.3)
        
        return deep_future[-len_export:]

    class Attention:
        def __init__(self, size_layer, embedded_size, learning_rate, size, output_size,
                    num_blocks = 2,
                    num_heads = 8,
                    min_freq = 50,
                    dropout_rate = 0.9):
            self.X = tf.compat.v1.placeholder(tf.float32, (None, None, size))
            self.Y = tf.compat.v1.placeholder(tf.float32, (None, output_size))
            self.dropout_rate = dropout_rate
            
            encoder_embedded = tf.layers.dense(self.X, embedded_size)
            encoder_embedded = tf.nn.dropout(encoder_embedded, rate = 1-self.dropout_rate)
            x_mean = tf.reduce_mean(self.X, axis = 2)
            en_masks = tf.sign(x_mean)
            encoder_embedded += self.sinusoidal_position_encoding(self.X, en_masks, embedded_size)
            
            for i in range(num_blocks):
                with tf.variable_scope('encoder_self_attn_%d'%i,reuse=tf.AUTO_REUSE):
                    encoder_embedded = self.multihead_attn(queries = encoder_embedded,
                                                keys = encoder_embedded,
                                                q_masks = en_masks,
                                                k_masks = en_masks,
                                                future_binding = False,
                                                num_units = size_layer,
                                                num_heads = num_heads)

                with tf.variable_scope('encoder_feedforward_%d'%i,reuse=tf.AUTO_REUSE):
                    encoder_embedded = self.pointwise_feedforward(encoder_embedded,
                                                        embedded_size,
                                                        activation = tf.nn.relu)
                    
            self.logits = tf.layers.dense(encoder_embedded[-1], output_size)
            self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
                self.cost
            )

        def sinusoidal_position_encoding(self, inputs, mask, repr_dim):
            T = tf.shape(inputs)[1]
            pos = tf.reshape(tf.range(0.0, tf.to_float(T), dtype=tf.float32), [-1, 1])
            i = np.arange(0, repr_dim, 2, np.float32)
            denom = np.reshape(np.power(10000.0, i / repr_dim), [1, -1])
            enc = tf.expand_dims(tf.concat([tf.sin(pos / denom), tf.cos(pos / denom)], 1), 0)
            return tf.tile(enc, [tf.shape(inputs)[0], 1, 1]) * tf.expand_dims(tf.to_float(mask), -1)

        def multihead_attn(self, queries, keys, q_masks, k_masks, future_binding, num_units, num_heads):
            T_q = tf.shape(queries)[1]                                      
            T_k = tf.shape(keys)[1]                  

            Q = tf.layers.dense(queries, num_units, name='Q')                              
            K_V = tf.layers.dense(keys, 2*num_units, name='K_V')    
            K, V = tf.split(K_V, 2, -1)        

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)                         
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)                    
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)                      

            align = tf.matmul(Q_, tf.transpose(K_, [0,2,1]))                      
            align = align / np.sqrt(K_.get_shape().as_list()[-1])                 

            paddings = tf.fill(tf.shape(align), float('-inf'))                   

            key_masks = k_masks                                                 
            key_masks = tf.tile(key_masks, [num_heads, 1])                       
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, T_q, 1])            
            align = tf.where(tf.equal(key_masks, 0), paddings, align)       

            if future_binding:
                lower_tri = tf.ones([T_q, T_k])                                          
                lower_tri = tf.linalg.LinearOperatorLowerTriangular(lower_tri).to_dense()  
                masks = tf.tile(tf.expand_dims(lower_tri,0), [tf.shape(align)[0], 1, 1]) 
                align = tf.where(tf.equal(masks, 0), paddings, align)                      
            
            align = tf.nn.softmax(align)                                            
            query_masks = tf.to_float(q_masks)                                             
            query_masks = tf.tile(query_masks, [num_heads, 1])                             
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, T_k])            
            align *= query_masks
            
            outputs = tf.matmul(align, V_)                                                 
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)             
            outputs += queries                                                             
            outputs = self.layer_norm(outputs)                                                 
            return outputs

        def pointwise_feedforward(self, inputs, hidden_units, activation=None):
            outputs = tf.layers.dense(inputs, 4*hidden_units, activation=activation)
            outputs = tf.layers.dense(outputs, hidden_units, activation=None)
            outputs += inputs
            outputs = self.layer_norm(outputs)
            return outputs

        def layer_norm(self, inputs, epsilon=1e-8):
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))

            params_shape = inputs.get_shape()[-1:]
            gamma = tf.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
            beta = tf.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())
            
            outputs = gamma * normalized + beta
            return outputs