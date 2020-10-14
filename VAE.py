# -*- coding: utf-8 -*-
"""
Patient-based breathing models using the Variational Autoencoder
COPYRIGHT: TU Delft, Netherlands. 2020.
LICENSE: GNU AGPL-3.0 
"""

# Import dependencies
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('seaborn')

from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Lambda, Input, Dense, BatchNormalization, Flatten, Dropout
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Reshape, MaxPooling1D, UpSampling1D, ReLU

class VariationalAutoencoder():
    
    def __init__(self, PATIENT, LATENT_DIM, BETA0, X):
        
        # Supress buggy warnings during training
        tf.get_logger().setLevel('ERROR')
        
        # Model parameters
        self.latent_dim = LATENT_DIM
        self.patient = PATIENT
        self.KL_beta0 = BETA0
        self.nSamples = 1000
        
        # Get data dimensions
        self.original_dim = X.shape[1] * X.shape[2]
        self.input_shape = X.shape[1:]
        self.KL_beta = BETA0*self.original_dim/self.latent_dim
        
        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.encoder.summary()
        self.decoder = self.build_decoder()
        self.decoder.summary()
        
        sgnl = Input(shape=self.input_shape)
        reconstructed_sgnl = self.decoder(self.encoder(sgnl)[2])
        self.variational_autoencoder = Model(sgnl, reconstructed_sgnl)
        
        # Reconstruction loss -p(x|z)
        reconstruction_loss =  tf.math.reduce_sum(K.square(sgnl - reconstructed_sgnl), [1,2])

        # KL divergence q(z|x)||p(z)
        kl_loss = 1 + self.encoder(sgnl)[1] - K.square(self.encoder(sgnl)[0]) - K.exp(self.encoder(sgnl)[1])
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        kl_loss *= self.KL_beta
            
        # Set up global loss
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.variational_autoencoder.add_loss(vae_loss) 
        self.variational_autoencoder.compile(optimizer=Adam(1e-4))
              
    def build_encoder(self):
        
        # Input layer - data point
        x = Input(shape=self.input_shape)

        # Convolution + pooling layers
        h = Conv1D(128, 4, strides=1, kernel_constraint=max_norm(3))(x)
        h = ReLU()(h)
        h = MaxPooling1D(pool_size=4, strides=1)(h)
        h = BatchNormalization()(h)
        h = Dropout(0.1)(h)
        
        h = Conv1D(256, 4, strides=1, kernel_constraint=max_norm(3))(h)
        h = ReLU()(h)
        h = MaxPooling1D(pool_size=4, strides=1)(h)
        h = BatchNormalization()(h)
        h = Dropout(0.1)(h)

        h = Conv1D(256, 4, strides=1, kernel_constraint=max_norm(3))(h)
        h = ReLU()(h)
        h = MaxPooling1D(pool_size=4, strides=1)(h)
        h = BatchNormalization()(h)
        h = Dropout(0.1)(h)

        h = Conv1D(64, 4, strides=1, kernel_constraint=max_norm(3))(h)
        h = ReLU()(h)
        h = BatchNormalization()(h)
        h = Flatten()(h)

        # Fully-connected layers
        h = Dense(256)(h)
        h = ReLU()(h)
        h = BatchNormalization()(h)

        h = Dense(128)(h)
        h = ReLU()(h)
        h = BatchNormalization()(h)
        
        # Mean and (log) variance of q(z|x)
        z_mean = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)
        z = Lambda(self.sample_z, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        
        return Model(x, [z_mean, z_log_var, z])

    def build_decoder(self):

        # Input layer - latent variables
        z = Input(shape=self.latent_dim)

        # Fully-connected layers
        h = Dense(128)(z)
        h = ReLU()(h)
        h = BatchNormalization()(h)

        h = Dense(256)(h)
        h = ReLU()(h)
        h = BatchNormalization()(h)
        h = Reshape((4,64))(h)

        # Deconvolutional layers
        h = Conv1DTranspose(128, 4, dilation_rate=2, kernel_constraint=max_norm(3))(h)
        h = ReLU()(h)
        h = BatchNormalization()(h)
        h = Dropout(0.3)(h)

        h = Conv1DTranspose(256, 4, dilation_rate=2, kernel_constraint=max_norm(3))(h)
        h = ReLU()(h)
        h = BatchNormalization()(h)
        h = Dropout(0.3)(h)

        h = Conv1DTranspose(256, 4, dilation_rate=1, kernel_constraint=max_norm(3))(h)
        h = ReLU()(h)
        h = BatchNormalization()(h)
        h = Dropout(0.3)(h)

        h = Conv1DTranspose(128, 4, dilation_rate=1, kernel_constraint=max_norm(3))(h)
        h = ReLU()(h)
        h = BatchNormalization()(h)
        h = Dropout(0.1)(h)

        # Output - reconstructed input
        x_ = Conv1DTranspose(6, 4)(h)
        
        return Model(z, x_)

    def sample_z(self, args):
        """
        Reparametrization trick
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


    def train(self, X, batch_size=256, epochs=10000, cv=3, verbose=0):
        """
        Train VAE different times, and save best weights.
        INPUT:
        X..................dataset
        batch_size.........number of data points per training step
        epochs.............number of data iterations
        cv.................number of models trained
        """
        # Save initial weights
        self.variational_autoencoder.save_weights(f"Conv1DAE/P{self.patient}/initial_weights.h5")
        
        for i in range(cv):

            # Random split of data for each trained model
            X_fit, X_test, _, _ = train_test_split(X, X, test_size=0.1, random_state=(1+i)*33)

            # Fit scaler to training set
            self.scaler = StandardScaler()
            X_fit = self.scaler.fit_transform(X_fit.reshape(-1, X_fit.shape[-1])).reshape(X_fit.shape)
            X_test = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            
            # Random split train and validation
            X_train, X_validation, _, _ = train_test_split(X_fit, X_fit, test_size=0.111, random_state=(1+i)*33)

            # Callbacks: EarlyStopping
            es = EarlyStopping(monitor='val_loss', mode='min', patience=75, verbose=1)
            mc = ModelCheckpoint(f"Conv1DAE/P{self.patient}/VAE-LS{self.latent_dim}-b{self.KL_beta0}.h5",
                                 monitor='val_loss', mode='min', save_best_only=True)
            
            # Train the autoencoder
            self.variational_autoencoder.load_weights(f"Conv1DAE/P{self.patient}/initial_weights.h5")
            history = self.variational_autoencoder.fit(X_train, epochs=epochs, batch_size=batch_size,
                                                       callbacks = [es, mc], validation_data=(X_validation, None), verbose=verbose)
            
            # Load best weights before EarlyStopping
            self.variational_autoencoder.load_weights(f"Conv1DAE/P{self.patient}/VAE-LS{self.latent_dim}-b{self.KL_beta0}.h5")
            
            # Calculate final losses
            # Evidence Lower Bound ELBO
            elbo_train = self.variational_autoencoder.evaluate(X_train,batch_size=batch_size,verbose=0)
            elbo_val = self.variational_autoencoder.evaluate(X_validation,batch_size=batch_size,verbose=0)
            elbo_test = self.variational_autoencoder.evaluate(X_test,batch_size=batch_size,verbose=0)

            # Reconstruction error (equivalent to mse)
            mse_train = tf.math.reduce_sum(K.square(X_train - self.variational_autoencoder.predict(X_train)))/X_train.shape[0]
            mse_val = tf.math.reduce_sum(K.square(X_validation - self.variational_autoencoder.predict(X_validation)))/X_validation.shape[0]
            mse_test = tf.math.reduce_sum(K.square(X_test - self.variational_autoencoder.predict(X_test)))/X_test.shape[0]
            print(f"CV [{i+1}/{cv}] [ELBO:{elbo_train:.3f}. ELBO val:{elbo_val:.3f}. ELBO test:{elbo_test:.3f}.")
            print(f"MSE:{mse_train:.3f}. MSE val:{mse_val:.3f}. MSE test:{mse_test:.3f}]")

            # Save model if the validation loss is improved
            if i == 0:
              best_loss = elbo_val
              self.encoder.save_weights(f"Conv1DAE/P{self.patient}/saves/VAEenc-LS{self.latent_dim}-b{self.KL_beta0}.h5")
              self.decoder.save_weights(f"Conv1DAE/P{self.patient}/saves/VAEdec-LS{self.latent_dim}-b{self.KL_beta0}.h5")
              self.save_model(X, X_fit, X_test, history)

            elif best_loss>elbo_val:
              best_loss = elbo_val
              self.encoder.save_weights(f"Conv1DAE/P{self.patient}/saves/VAEenc-LS{self.latent_dim}-b{self.KL_beta0}.h5")
              self.decoder.save_weights(f"Conv1DAE/P{self.patient}/saves/VAEdec-LS{self.latent_dim}-b{self.KL_beta0}.h5")
              self.save_model(X, X_fit, X_test, history)
              print("Loss improved. Weights saved.")   

    def save_model(self, X, X_train, X_test, history):
        """
        Save reconstructions, history and plot first 2 dimensions of latent space
        """   
        # Save reconstructions and history
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        x_rec = self.variational_autoencoder.predict(X_scaled)
        x_rec = self.scaler.inverse_transform(x_rec.reshape(-1, x_rec.shape[-1])).reshape(x_rec.shape)
        savemat(f"Conv1DAE/P{self.patient}/recs/VAErec-LS{self.latent_dim}-b{self.KL_beta0}.mat",{'x_rec':x_rec})
        savemat(f"Conv1DAE/P{self.patient}/recs/VAETH-LS{self.latent_dim}-b{self.KL_beta0}.mat",history.history)

        # Plot latent space
        x_enc = self.encoder.predict(X_train)[2]
        x_enc_test = self.encoder.predict(X_test)[2]
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(x_enc[:,0], x_enc[:,1], label='Train data')
        plt.scatter(x_enc_test[:,0], x_enc_test[:,1], label='Test data')
        plt.title(f"P{self.patient}-LS{self.latent_dim}")
        plt.xlabel('z1')
        plt.ylabel('z2')
        plt.legend(loc='upper right')
        plt.show()
        plt.savefig('latent.png',dpi=300,bbox_inches='tight')

        # Plot history for loss
        fig2 = plt.figure(figsize=(6, 6))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        plt.savefig('curves.png',dpi=300,bbox_inches='tight')

# Load data
# Data must be a tensor [samples, timesteps, channels]
data = loadmat(f"Conv1DAE/data/data-P{self.patient}.mat")
X = np.array(data["X"])

# Initialize VAE
vae = VariationalAutoencoder(14, 5, 0.02, X)

# Train VAE
vae.train(X, cv=1, verbose=1)
