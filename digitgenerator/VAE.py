import tensorflow as tf
import numpy as np
import os

class Vae(object):               
    
    def __init__(self,batchsize=50, epochsnum=10, latent_dim=8):       

        #parameters
        self.batch_size = batchsize
        self.epochs=epochsnum
        self.n_z=latent_dim
        
        #check if the parameters file is found, if not train the model
        if os.path.exists('tf_vae_files/vae-0.meta'):
            
            print('found pretrained variables which can be used')
            
        else:
            print("no trained parameters available, train the model with Digitgenerator().train_vae()")               
    
        
    @staticmethod
    def make_onehot(labels):
        
        train_len=len(labels)
        labels1h = np.zeros((train_len, 10))
        labels1h[np.arange(train_len), labels] = 1

        return labels1h
    

    # leaky reLu unit to improve the performance in the bottleneck
    @staticmethod
    def lrelu(x, leak=0.3, name="lrelu"):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)

    #convolution layer
    @staticmethod
    def conv2d(x, feats_in, feats_out, name):
        with tf.variable_scope(name):
            w = tf.get_variable("w",[5,5,feats_in, feats_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b = tf.get_variable("b",[feats_out], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(x, w, strides=[1,2,2,1], padding="SAME") + b
            return conv

    #deconvolution layer
    @staticmethod
    def conv_transpose(x, outputShape, name,reusing,training):
        with tf.variable_scope(name):

            w = tf.get_variable("w",[5,5, outputShape[-1], x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=0.02),trainable=training)
            b = tf.get_variable("b",[outputShape[-1]], initializer=tf.constant_initializer(0.0),trainable=training)
            convt = tf.nn.conv2d_transpose(x, w, output_shape=outputShape, strides=[1,2,2,1])
            return convt
        
    #sampling of the latent variable with re-parametrization technique
    @staticmethod
    def sample_latent_var(mu, logsigma_sq):
        
        epsilon = tf.random_normal(tf.shape(mu), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
        std_z = tf.exp(0.5 * logsigma_sq) #because logsigma_sq=log(std^2)
        z = mu + tf.multiply(std_z, epsilon) #now z is N(mu,sigma) where mu and sigma will be learned
        
        return z

    # encoder architecture
    def encode(self,input_data,lbls):
        with tf.variable_scope("encoder"):      
            
            #convolutional layersbls
            l1 = self.lrelu(self.conv2d(input_data, 1, 16, "l_1")) # 28x28x1 -> 14x14x16
            l1 = tf.contrib.layers.flatten(l1)
            
            #add the labels of the digits since we run a conditional variational autoencoder and wish to generate specific digits
            l1=tf.concat([lbls,l1], axis=1)
            
            #l2 is dense
            l2=tf.contrib.layers.fully_connected(l1,512,activation_fn=tf.nn.relu,
                                        weights_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.04),scope='encoder_l2')
            
            #make sigma and mu out
            mu=tf.contrib.layers.fully_connected(l2,self.n_z,activation_fn=None,
                                         weights_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.04),scope='encoder_mu')

            logsigma_sq=tf.contrib.layers.fully_connected(l2,self.n_z,activation_fn=None,
                                         weights_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.04),scope='encoder_sigma')

            return mu, logsigma_sq

    # decoder architecture
    def decode(self,z,input_len,lbls,reusing=None,training=True):
        with tf.variable_scope("decoder"):
            
            #concat with labels because conditioned on the specific digit
            l_in=tf.concat([z,lbls], axis=1)

            l1=tf.contrib.layers.fully_connected(l_in, 512,activation_fn=tf.nn.relu,reuse=reusing,trainable=training,
                                         weights_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.04),scope='decode_l1')
            
            l2=tf.contrib.layers.fully_connected(l1, 14*14*16,activation_fn=tf.nn.relu,reuse=reusing,trainable=training,
                                         weights_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.04),scope='decode_l2')
            
        
            l2 = tf.reshape(l2, shape=[input_len, 14, 14, 16])     
            d_out = self.conv_transpose(l2, [input_len, 28, 28, 1], "d_out",reusing,training)
    

        return tf.squeeze(d_out)
    


    #train the variational encoder from the input images
    def train_vae(self,train_data,labels,alfa=1.0):        
        
        #reset and start session
        tf.reset_default_graph()
        sess = tf.InteractiveSession()

        #set up placeholders
        x=tf.placeholder('float',[None,28,28])
        lbls=tf.placeholder('float',[None,10])

        #make labels to one hot
        labels1h=self.make_onehot(labels)
            
        #encoder from image
        mu, logsigma_sq = self.encode(tf.reshape(x, shape=[-1, 28, 28, 1]),lbls)
        
        # Sampling of latent variable by re-parameterization technique
        z = self.sample_latent_var(mu, logsigma_sq)

        #decode back to image
        generated_image=self.decode(z,self.batch_size,lbls)

        #reconstruction loss
        loss_reconst_b = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=generated_image,labels=x),axis=[1,2])
        loss_reconst=tf.reduce_mean(loss_reconst_b)

        #the KL -divergence. our prior and recognition networks are gaussion so we have an explicit form for the loss
        loss_KLD_b = -0.5 * tf.reduce_sum(1 + logsigma_sq - tf.square(mu) - tf.exp(logsigma_sq), axis=1)
        loss_KLD = tf.reduce_mean(loss_KLD_b) #the mean over batch (initial shape batch, latent dim)

        #total loss is the sum of the reconstruction and kl divergence from the prior
        loss=loss_reconst+alfa*loss_KLD
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        #create a saver to save the model, keep 1 latest versions
        saver = tf.train.Saver(max_to_keep=1)  
        sess.run(tf.global_variables_initializer())

        print('Starting training')        
        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_loss_kld = 0
            

            #shuffle the train data for each epoch with similar manner
            rng_state = np.random.get_state()
            np.random.shuffle(train_data)
            np.random.set_state(rng_state)
            np.random.shuffle(labels1h)

            #the epoch loop
            for index in range(int(train_data.shape[0]/self.batch_size)):
                batch_x = train_data[index*self.batch_size:(index+1)*self.batch_size]
                batch_lbls= labels1h[index*self.batch_size:(index+1)*self.batch_size]

                #record losses
                _, c,kld = sess.run([optimizer, loss,loss_KLD], feed_dict={x: batch_x, lbls: batch_lbls})
                epoch_loss += c
                epoch_loss_kld += kld
                
            #print the total loss and the KL-loss
            print('Epoch', epoch+1, ' / ',self.epochs,'loss:',epoch_loss,', KLD %:',100*epoch_loss_kld/epoch_loss)
            
        #save the model
        save_path=saver.save(sess, 'tf_vae_files/vae',global_step=0)                
        print("Model saved in file: %s"+save_path)
        
        return None
    
    
     #outpout the latent variables for data
    def give_latent(self,digitarray,labelarray):        
        
        #reset and start session
        tf.reset_default_graph()
        sess = tf.InteractiveSession()

        #set up placeholders
        x=tf.placeholder('float',[None,28,28])
        
        #make labels to one hot
        labels1h=self.make_onehot(labelarray)
            
        #encoder from image
        mu, logsigma_sq = self.encode(tf.reshape(x, shape=[-1, 28, 28, 1]),labels1h)
        
        # Sampling of latent variable by re-parameterization technique
        z = self.sample_latent_var(mu, logsigma_sq)
        
        #load parameters
        saver = tf.train.Saver(max_to_keep=2)
        print("Restoring saved parameters")
        saver_recover = tf.train.import_meta_graph('tf_vae_files/vae-0.meta')
        saver_recover.restore(sess, tf.train.latest_checkpoint('tf_vae_files'))

        #run
        latent_out=sess.run(z, feed_dict={x: digitarray})

        return latent_out
    
    
    #sample from the latent space of the conditional variational autoencoder
    def vae_generate(self,labellist):
        
        #make digitlist to one hot
        lbls=self.make_onehot(np.asarray(labellist))
        
        #reset and start session
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        
        #just sample with mu,sigma=0,0 , other methods could be implemented here
        mu, logsigma_sq = tf.zeros([len(labellist),self.n_z]),tf.zeros([len(labellist),self.n_z])
        z = self.sample_latent_var(mu, logsigma_sq)
        generated_image=self.decode(z,len(labellist),lbls)
    
        #outputs as sigmoids
        image_out = tf.nn.sigmoid(tf.reshape(generated_image, shape=[-1, 28, 28]))

        #load parameters
        saver = tf.train.Saver(max_to_keep=2)
        print("Restoring saved parameters")
        saver_recover = tf.train.import_meta_graph('tf_vae_files/vae-0.meta')
        saver_recover.restore(sess, tf.train.latest_checkpoint('tf_vae_files'))

        #run
        digits_out=sess.run(image_out)

        #make to list and return
        return list(digits_out)
    
    
    #run a list of digits through the autoencoder and return a list of digits
    #could be used for example from denoising
    def ff_autoencode(self,digitlist,labellist):
        
        #reset and start session
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        
        #onehot labels
        labels1h=self.make_onehot(labellist)

        #set up placeholders
        x=tf.placeholder('float',[None,28,28])

        #computational graph, encode, sample from latent and decode:
        mu, logsigma_sq = self.encode(tf.reshape(x, shape=[-1, 28, 28, 1]),labels1h)
        z = self.sample_latent_var(mu, logsigma_sq)
        generated_image=self.decode(z,len(labellist),labels1h)
    
        #outputs as sigmoids
        image_out = tf.nn.sigmoid(tf.reshape(generated_image, shape=[-1, 28, 28]))

        #load parameters
        saver = tf.train.Saver(max_to_keep=2)
        print("Restoring saved parameters")
        saver_recover = tf.train.import_meta_graph('tf_vae_files/vae-0.meta')
        saver_recover.restore(sess, tf.train.latest_checkpoint('tf_vae_files'))

        #convert to numpy array and generate the corresponding output
        digits_in=np.asarray(digitlist)   

        #run
        digits_out=sess.run(image_out,feed_dict={x: digits_in})

        #make to list and return
        return list(digits_out)
