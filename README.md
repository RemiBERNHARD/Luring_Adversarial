# Luring of adversarial perturbations

This repository contains pretrained models and codes files to reproduce the results presented in the article "Luring of Adversarial Perturbations".


## Environment and libraries

The python scripts were executed in the following environment:

* OS: CentOS Linux 7
* GPU: NVIDIA GeForce GTX 1080 
* Cuda version: 9.0.176
* Python version: 2.7.5

The following version of some Python packages are necessary: 

* Tensorflow: 1.12.0
* Cleverhans: 3.0.1
* Keras: 2.2.4
* Numpy: 1.16.12


## Files

### Model files
    
Unzip the "models.zip" file to get the "models" repository.
The "models" repository contains the pretrained models for the datasets MNIST, SVHN and CIFAR10. As an example, on MNIST:    

* models/MNIST_float.h5 is the base classifier trained.
* models/MNIST_stacked.h5 is the model corresponding to the Stacked architecture.
* models/MNIST_auto.h5 is the model corresponding to the Auto architecture.
* models/MNIST_ce.h5 is the model corresponding to the C_E architecture.
* models/MNIST_luring.h5 is the model corresponding to the Luring architecture.

Unzip the "models_p.zip" file to get the "models_p" repository.
The "models_p" repository contains the components prepended to the base classifier for the datasets MNIST, SVHN and CIFAR10. As
an example, on MNIST:

* models_p/MNIST_auto_p.h5 is the auto encoder corresponding ot the Auto architecture.
* models_p/MNIST_ce_p.h5 is the preprended component corresponding ot the C_E architecture.
* models_p/MNIST_luring_p.h5 is the preprended component corresponding to the Luring architecture.


### Training files
        
Although pretrained models are provided, some python files allow to train and save your own models, 
for the MNIST, SVHN and CIFAR10 datasets. As an example, on MNIST:

    # Train the base classifier
    python train_mnist.py float 
    # Train the Stacked model
    python train_mnist.py stacked
    # Train the Auto model
    python train_mnist.py auto
    # Train the C_E model
    python train_mnist.py ce
    # Train the Luring model
    python train_mnist.py luring

### Evaluation under attack

#### Verification of the *luring* effect

As an example, for a perturbation of ```epsilon=0.06``` (pixel values have been scaled between 0 and 1) on SVHN, the following command allows
to get the values to reproduce the figures of the part "Verification of the *luring* effect":

    python verification_mnist.py 0.3

#### Adversarial results

As an example, for a perturbation of ```epsilon=0.03``` (pixel values have been scaled between 0 and 1) on CIFAR10, the following command allows
to get the values to reproduce the tables "Adversarial results" for the Luring architecture:

    python attack_cifar10.py 0.03 luring

        
        
        
        
        
        
        
        
        
        
        
        
