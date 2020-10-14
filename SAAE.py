# -*- coding: utf-8 -*-
"""
Modeling breathing signals from a population of patients using a 
modified semi-supervised Adversarial Autoencoder algorithm.
COPYRIGHT: TU Delft, Netherlands. 2020.
LICENSE: GNU AGPL-3.0 
"""

# Import libraries and dependencies
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
mpl.style.use('seaborn')

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import add, concatenate, Reshape
from tensorflow.keras.layers import Lambda, Input, Dense, BatchNormalization, Flatten, Dropout, ReLU
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D, LeakyReLU

# Define semi-supervised AAE class
class SemiSupervisedAdversarialAutoencoder():

    def __init__(self, LATENT_DIM, N_CLASSES, TS):
        
        # Eager execution slows down training
        tf.compat.v1.disable_eager_execution()
        
        # Initialize input data scaler
        self.scaler = StandardScaler()
        
        # Get data dimensions
        self.latent_dim = LATENT_DIM
        self.n_classes = N_CLASSES
        self.original_dim = TS * 6                  
        self.input_shape = (TS, 6)
        self.ae_weight = 4

        optimizer_dis = Adam(2e-4, 0.9)
        optimizer_ae = Adam(1e-4, 0.7)

        # Build and compile the discriminators
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                     optimizer=optimizer_dis, metrics=['accuracy'])

        # Build the encoder / decoder
        self.encoder,self.generator = self.build_encoder()
        self.decoder = self.build_decoder()
        sgnl = Input(shape=self.input_shape)
        [encoded_repr, encoded_label] = self.encoder(sgnl)
        reconstructed_sgnl = self.decoder([encoded_repr, encoded_label])

        # Compile the generator
        self.generator.compile(loss=['sparse_categorical_crossentropy'],
                               optimizer=optimizer_ae,
                               loss_weights=[10])

        # The discriminator is not trained in 2nd phase
        self.discriminator.trainable = False
        validity = self.discriminator([encoded_repr, encoded_label])

        self.adversarial_autoencoder = Model(sgnl, [reconstructed_sgnl, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[self.ae_weight, 1], 
            optimizer=optimizer_ae)
        

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
        y = Dense(self.n_classes, activation='softmax')(h)
        
        return Model(x, [z, y]), Model(x, y)

    def build_decoder(self):
        
        # Add latent vectors y and z
        z = Input(shape=(self.latent_dim,))
        y = Input(shape=(self.n_classes,))
        y1 = Dense(self.latent_dim, trainable=False, 
                   weights=[np.concatenate((np.identity(self.n_classes)*self.n_classes,
                            np.zeros((self.n_classes,self.latent_dim-self.n_classes))),axis=1)],
                   use_bias=False)(y)     
        merge = add([y1, z])

        # Fully-connected layers
        h = Dense(128)(merge)
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
        
        return Model([z,y], x_)

    def build_discriminator(self):

        z = Input(shape=(self.latent_dim,))
        y = Input(shape=(self.n_classes,))
        encoded_repr = concatenate([y, z])
        model = Sequential()
        model.add(Dense(64, input_dim=self.latent_dim+self.n_classes))
        model.add(LeakyReLU(0.1))
        model.add(Dense(128))
        model.add(LeakyReLU(0.1))
        model.add(Dense(128))
        model.add(LeakyReLU(0.1))
        model.add(Dense(64))
        model.add(LeakyReLU(0.1))
        model.add(Dense(1, activation="sigmoid"))
        validity_z = model(encoded_repr)

        return Model([z,y], validity_z)
    
    def sample_y(self, batch_size):
        """
        Samples from a uniform categorical distribution
        """
        y=np.zeros((batch_size, self.n_classes), dtype=np.float32)
        labels = np.random.randint(0, self.n_classes, batch_size)
        for b in range(batch_size):
            y[b, labels[b]] = 1
            
        return y

    def train(self, X, uX, uy, batch_size=256):
        
        """
        Train AAE and save best weights.
        INPUT:
        X.......unlabeled data samples
        uX......few labeled data samples
        uy.........labels for the labeled samples
        batch_size.....number of data points per training step
        """
        # Split data into training and test sets
        X_fit, X_test, _, _ = train_test_split(X, X, test_size=0.1, random_state=33)
        X_fit = self.scaler.fit_transform(X_fit.reshape(-1, X_fit.shape[-1])).reshape(X_fit.shape)
        X_test = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        uX = self.scaler.transform(uX.reshape(-1, uX.shape[-1])).reshape(uX.shape)

        # Random split train and validation
        X_train, X_validation, _, _ = train_test_split(X_fit, X_fit, test_size=0.111, random_state=33)

        # Adversarial ground truths
        history = {'loss':[],'val_loss':[]}
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        grounds = np.concatenate((valid, fake))
        valid_val = np.ones((len(X_validation), 1))
        batch_number = int(np.fix(len(X_train)/batch_size))
        dis_loss = np.zeros((batch_number, 1))
        dis_acc = np.zeros((batch_number, 1))
        gen_loss = np.zeros((batch_number, 1))
        g_loss_train = np.zeros((batch_number, 1))

        epoch = 0
        counter = 0
        min_error = 1000

        while counter <=30:

            # Shuffle data at the beginning of each epoch
            np.random.shuffle(X_train)
            
            for i_batch in range(batch_number):
            
                # Select a random batch of images
                X_batch = X_train[i_batch*batch_size:(i_batch+1)*batch_size]
                latent_fake = self.encoder.predict(X_batch)[0]
                labels_fake = self.encoder.predict(X_batch)[1]
                latent_real = np.random.normal(size=(batch_size, self.latent_dim))
                labels_real = self.sample_y(batch_size)
                latents = np.concatenate((latent_real,latent_fake))
                labels = np.concatenate((labels_real,labels_fake))

                # Train the Z discriminator
                d_loss = self.discriminator.train_on_batch([latents,labels], grounds)
                dis_loss[i_batch] = d_loss[0]
                dis_acc[i_batch] = d_loss[1]

                # Train the Y generator to match the few available labels
                selected_labels = np.random.choice(np.arange(len(uy)), batch_size)
                self.generator.train_on_batch(uX[selected_labels],[uy[selected_labels]])

                # Train the generator
                g_loss = self.adversarial_autoencoder.train_on_batch(X_batch, [X_batch, valid])
                g_loss_train[i_batch] = g_loss[1]
                gen_loss[i_batch] = g_loss[2]
                
            # Get training and validation error
            g_loss_validation = self.adversarial_autoencoder.evaluate(X_validation, [X_validation, valid_val],
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
                self.encoder.save_weights(f"ConvSAAE/saves/AAEenc-LS{self.latent_dim}.h5")
                self.decoder.save_weights(f"ConvSAAE/saves/AAEdec-LS{self.latent_dim}.h5")
                self.discriminator.save_weights(f"ConvSAAE/saves/AAEdis-LS{self.latent_dim}.h5")
                counter = 0
            else:
                min_error = g_loss_validation[1]
                self.encoder.save_weights(f"ConvSAAE/saves/AAEenc-LS{self.latent_dim}.h5")
                self.decoder.save_weights(f"ConvSAAE/saves/AAEdec-LS{self.latent_dim}.h5")
                self.discriminator.save_weights(f"drive/ConvSAAE/saves/AAEdis-LS{self.latent_dim}.h5")
                counter = 0

        self.encoder.load_weights(f"ConvSAAE/saves/AAEenc-LS{self.latent_dim}.h5")
        self.decoder.load_weights(f"ConvSAAE/saves/AAEdec-LS{self.latent_dim}.h5")       
        self.save_model(X, history)

        # Calculate final losses
        mse_train = self.adversarial_autoencoder.evaluate(X_train,[X_train, np.ones((len(X_train), 1))],batch_size=batch_size, verbose=0)[1]
        mse_val = self.adversarial_autoencoder.evaluate(X_validation,[X_validation, np.ones((len(X_validation), 1))],batch_size=batch_size, verbose=0)[1]
        mse_test = self.adversarial_autoencoder.evaluate(X_test,[X_test, np.ones((len(X_test), 1))],batch_size=batch_size, verbose=0)[1]
        print(f"[MSE:{self.original_dim*mse_train:.3f}. MSE val:{self.original_dim*mse_val:.3f}. MSE test:{self.original_dim*mse_test:.3f}]") 

    def save_model(self, X, history):
        """
        Generate random samples and plot encodings.
        INPUT:
        X_scaled...........scaled input data to encode [nsamples x dim]
        """   
        # Save reconstructions and history
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        x_rec = self.decoder.predict(self.encoder.predict(X_scaled))
        x_rec = self.scaler.inverse_transform(x_rec.reshape(-1, x_rec.shape[-1])).reshape(x_rec.shape)
        savemat(f"ConvSAAE/recs/AAErec-LS{self.latent_dim}.mat",{'x_rec':x_rec})
        savemat(f"ConvSAAE/recs/AAETH-LS{self.latent_dim}.mat",history)

        # Plot history for loss
        fig2 = plt.figure(figsize=(6, 6))
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('curves.png',dpi=300,bbox_inches='tight')


# Function selecting a random subset of signals per class
def sample_labels(X,y,nu):
  n_labels = len(np.unique(y))
  uX = np.empty(shape=(1,X.shape[1],X.shape[2]))
  uy = np.empty(shape=(1,))
  for i in range(len(np.unique(y))):
    ytemp = y[y==i]
    Xtemp = X[y==i]
    indeces = np.floor(np.random.uniform(len(ytemp),size=(nu,))).astype(int)
    uy = np.append(uy,ytemp[indeces],axis=0)
    uX = np.append(uX,Xtemp[indeces],axis=0)
  return uX[1:],uy[1:]

# Load data
data = loadmat(f"ConvSAAE/data/25GUH100.mat")
X = np.array(data["X"])
y = np.array(data["y"]) - 1
y = np.reshape(y,-1)
uX,uy = sample_labels(X,y,600)
print(X.shape, y.shape,uX.shape,uy.shape)

# Initialize semi-supervised AAE
aae = SemiSupervisedAdversarialAutoencoder(15,len(np.unique(y)),25)

# Train semi-supervised AAE
aae.train(X,uX,uy)

