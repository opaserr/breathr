# -*- coding: utf-8 -*-
"""
Patient-based breathing models using the Adversarial Autoencoder
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Lambda, Input, Dense, BatchNormalization, Flatten, Dropout, ReLU
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Reshape, MaxPooling1D, UpSampling1D, LeakyReLU, GaussianNoise

# Define AAE class
class AdversarialAutoencoder():
    
    def __init__(self, PATIENT, LATENT_DIM, X):
        
        # Eager execution slows down training
        tf.get_logger().setLevel('ERROR')
        tf.compat.v1.disable_eager_execution()
        
        # Model parameters
        self.latent_dim = LATENT_DIM
        self.patient = PATIENT
        self.ae_weight = 3
        self.nSamples = 1000
        self.scaler = StandardScaler()
        self.original_dim = X.shape[1] * X.shape[2]
        self.input_shape = X.shape[1:]
        
        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.encoder.summary()
        self.decoder = self.build_decoder()
        self.decoder.summary()

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(2e-4), metrics=['accuracy'])

        sgnl = Input(shape=self.input_shape)
        encoded_repr = self.encoder(sgnl)
        reconstructed_sgnl = self.decoder(encoded_repr)

        # The discriminator is not trained in reconstruction phase
        self.discriminator.trainable = False
        validity = self.discriminator(encoded_repr)

        self.adversarial_autoencoder = Model(sgnl, [reconstructed_sgnl, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[self.ae_weight,1], optimizer=Adam(1e-4))
          
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
        h = Dropout(0.1)(h)
        h = Flatten()(h)

        # Fully-connected layers
        h = Dense(256)(h)
        h = ReLU()(h)
        h = BatchNormalization()(h)

        h = Dense(128)(h)
        h = ReLU()(h)
        h = BatchNormalization()(h)
        
        z = Dense(self.latent_dim)(h)
        
        return Model(x, z)

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

    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(64, input_dim=self.latent_dim))
        model.add(LeakyReLU(0.1))
        model.add(Dense(128))
        model.add(LeakyReLU(0.1))
        model.add(Dense(128))
        model.add(LeakyReLU(0.1))
        model.add(Dense(64))
        model.add(LeakyReLU(0.1))
        model.add(Dense(1, activation="sigmoid"))
        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

    def train(self, X, batch_size=256):
        """
        Train AAE and save best weights.
        INPUT:
        X..................dataset [samples, timesteps, channels]
        batch_size.........number of data points per training step
        """
        # Split data into training and test set
        # Random split of data for each trained model
        X_fit, X_test, _, _ = train_test_split(X, X, test_size=0.1, random_state=33)

        # Fit scaler to training set
        self.scaler = StandardScaler()
        X_fit = self.scaler.fit_transform(X_fit.reshape(-1, X_fit.shape[-1])).reshape(X_fit.shape)
        X_test = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Random split train and validation
        X_train, X_validation, _, _ = train_test_split(X_fit, X_fit, test_size=0.111, random_state=33)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        grounds = np.concatenate((valid, fake))
        batch_number = int(np.fix(len(X_train)/batch_size))
        valid_val = np.ones((len(X_validation), 1))
        dis_loss = np.zeros((batch_number, 1))
        dis_acc = np.zeros((batch_number, 1))
        gen_loss = np.zeros((batch_number, 1))
        g_loss_train = np.zeros((batch_number, 1))

        # Auxiliary variables
        epoch = 0
        counter = 0
        min_error = 1000
        history = {'loss':[],'val_loss':[]}

        # EarlyStopping using counter
        while counter <=100:

            # Shuffle data at the beginning of each epoch
            np.random.shuffle(X_train)
            
            for i_batch in range(batch_number):
            
                # Select a random batch of images
                X_batch = X_train[i_batch*batch_size:(i_batch+1)*batch_size]
                latent_fake = self.encoder.predict(X_batch)
                latent_real = np.random.normal(size=(batch_size, self.latent_dim))
                latents = np.concatenate((latent_real,latent_fake))

                # Train the discriminator
                d_loss = self.discriminator.train_on_batch(latents, grounds)
                dis_loss[i_batch] = d_loss[0]
                dis_acc[i_batch] = d_loss[1]

                # Train the generator
                g_loss = self.adversarial_autoencoder.train_on_batch(X_batch, [X_batch, valid])
                g_loss_train[i_batch] = g_loss[1]
                gen_loss[i_batch] = g_loss[2]
            
            # Get training and validation error
            g_loss_validation = self.adversarial_autoencoder.evaluate(X_validation,
                                                                      [X_validation, valid_val],
                                                                      batch_size=batch_size, verbose=0)
            history['loss'].append(np.mean(g_loss_train))
            history['val_loss'].append(g_loss_validation[1])

            # Print the progress
            epoch += 1
            print (f'{epoch} [D loss: {np.mean(dis_loss):.3f}, acc: {100*np.mean(dis_acc):.3f}%]'
                   f'[G loss: {np.mean(gen_loss):.3f}][mse: {(self.original_dim * np.mean(g_loss_train)):.3f}]'
                   f'[val mse: {(self.original_dim * g_loss_validation[1]):.3f}]')
            
            # Early stopping criterion
            if g_loss_validation[1] > 1.02 * min_error:
                counter+= 1
            elif g_loss_validation[1] > min_error:
                self.encoder.save_weights(f"Conv1DAE/P{self.patient}/saves/AAEenc-LS{self.latent_dim}.h5")
                self.decoder.save_weights(f"Conv1DAE/P{self.patient}/saves/AAEdec-LS{self.latent_dim}.h5")
                self.discriminator.save_weights(f"Conv1DAE/P{self.patient}/saves/AAEdis-LS{self.latent_dim}.h5")
                counter = 0
            else:
                min_error = g_loss_validation[1]
                self.encoder.save_weights(f"Conv1DAE/P{self.patient}/saves/AAEenc-LS{self.latent_dim}.h5")
                self.decoder.save_weights(f"Conv1DAE/P{self.patient}/saves/AAEdec-LS{self.latent_dim}.h5")
                self.discriminator.save_weights(f"Conv1DAE/P{self.patient}/saves/AAEdis-LS{self.latent_dim}.h5")
                counter = 0

        self.encoder.load_weights(f"Conv1DAE/P{self.patient}/saves/AAEenc-LS{self.latent_dim}.h5")
        self.decoder.load_weights(f"Conv1DAE/P{self.patient}/saves/AAEdec-LS{self.latent_dim}.h5")       
        self.save_model(X, X_train, X_test, history)

        # Calculate final losses
        mse_train = self.adversarial_autoencoder.evaluate(X_train,[X_train, np.ones((len(X_train), 1))],batch_size=batch_size, verbose=0)[1]
        mse_val = self.adversarial_autoencoder.evaluate(X_validation,[X_validation, np.ones((len(X_validation), 1))],batch_size=batch_size, verbose=0)[1]
        mse_test = self.adversarial_autoencoder.evaluate(X_test,[X_test, np.ones((len(X_test), 1))],batch_size=batch_size, verbose=0)[1]
        print(f"[MSE:{self.original_dim*mse_train:.3f}. MSE val:{self.original_dim*mse_val:.3f}. MSE test:{self.original_dim*mse_test:.3f}]")

    def save_model(self, X, X_train, X_test, history):
        """
        Save reconstructions, history and plot first 2 dimensions of latent space
        """   
        # Save reconstructions and history
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        x_rec = self.decoder.predict(self.encoder.predict(X_scaled))
        x_rec = self.scaler.inverse_transform(x_rec.reshape(-1, x_rec.shape[-1])).reshape(x_rec.shape)
        savemat(f"Conv1DAE/P{self.patient}/recs/AAErec-LS{self.latent_dim}.mat",{'x_rec':x_rec})
        savemat(f"Conv1DAE/P{self.patient}/recs/AAETH-LS{self.latent_dim}.mat",history)

        # Plot latent space
        x_enc = self.encoder.predict(X_train)
        x_enc_test = self.encoder.predict(X_test)
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(x_enc[:,0], x_enc[:,1], label='Train data')
        plt.scatter(x_enc_test[:,0], x_enc_test[:,1], label='Test data')
        plt.title(f"P{self.patient}-LS{self.latent_dim}")
        plt.xlabel('z1')
        plt.ylabel('z2')
        plt.legend(loc='upper right')
        plt.savefig('latent.png',dpi=300,bbox_inches='tight')

        # Plot history for loss
        fig2 = plt.figure(figsize=(6, 6))
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('curves.png',dpi=300,bbox_inches='tight')

# Load data
# Data must be a tensor [samples, timesteps, channels]
data = loadmat(f"Conv1DAE/data/data-P{self.patient}.mat")
X = np.array(data["X"])

# Initialize AAE
aae = AdversarialAutoencoder(11, 10, X)

# Train AAE
aae.train(X)

