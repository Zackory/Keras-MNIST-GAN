# Keras GAN for MNIST

Generative Adverserial Network (GAN) implementation using the [Keras](https://keras.io/ "Keras") library.  
Several of the tricks from [ganhacks](https://github.com/soumith/ganhacks) have been implemented.

`mnist_dcgan.py`: provides a Deep Convolutional Generative Adverserial Network (DCGAN) implementation.  
Each epoch takes approx. 1 minute on a NVIDIA Tesla K80 GPU (using Amazon EC2).

`mnist_gan.py`: a standard GAN using fully connected layers.


## Generated Images

![Generated MNIST images at epoch 50](images/generated_image_epoch_50.png "Generated MNIST images at epoch 50.")

## Loss per Batch

![Loss at every epoch for 50 epochs](images/loss_epoch_50.png "Loss at every epoch for 50 epochs.")

