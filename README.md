# Simpsons DCGAN

### Dataset

Unfiltered cropped Simpsons faces from this dataset
https://www.kaggle.com/kostastokis/simpsons-faces

### Architecture

I've used [Radford et al., 2015.](https://arxiv.org/abs/1511.06434) architecture
- Random normal initializers, not truncated
- No noise was added to real input
- Aggressive learning rates for generator

```
Discriminator
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 64, 64, 64)        4864      
_________________________________________________________________
batch_normalization_1 (Batch (None, 64, 64, 64)        256       
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 64, 64, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 128)       204928    
_________________________________________________________________
batch_normalization_2 (Batch (None, 32, 32, 128)       512       
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 32, 32, 128)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 256)       819456    
_________________________________________________________________
batch_normalization_3 (Batch (None, 16, 16, 256)       1024      
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 16, 16, 256)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 16, 16, 512)       3277312   
_________________________________________________________________
batch_normalization_4 (Batch (None, 16, 16, 512)       2048      
_________________________________________________________________
leaky_re_lu_4 (LeakyReLU)    (None, 16, 16, 512)       0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 1024)        13108224  
_________________________________________________________________
batch_normalization_5 (Batch (None, 8, 8, 1024)        4096      
_________________________________________________________________
leaky_re_lu_5 (LeakyReLU)    (None, 8, 8, 1024)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 65536)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 65537     
_________________________________________________________________
activation_1 (Activation)    (None, 1)                 0         
=================================================================
Total params: 17,488,257
Trainable params: 17,484,289
Non-trainable params: 3,968
_________________________________________________________________
Generator
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_2 (Dense)              (None, 65536)             6619136   
_________________________________________________________________
reshape_1 (Reshape)          (None, 8, 8, 1024)        0         
_________________________________________________________________
leaky_re_lu_6 (LeakyReLU)    (None, 8, 8, 1024)        0         
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 16, 16, 512)       13107712  
_________________________________________________________________
batch_normalization_6 (Batch (None, 16, 16, 512)       2048      
_________________________________________________________________
leaky_re_lu_7 (LeakyReLU)    (None, 16, 16, 512)       0         
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 32, 32, 256)       3277056   
_________________________________________________________________
batch_normalization_7 (Batch (None, 32, 32, 256)       1024      
_________________________________________________________________
leaky_re_lu_8 (LeakyReLU)    (None, 32, 32, 256)       0         
_________________________________________________________________
conv2d_transpose_3 (Conv2DTr (None, 64, 64, 128)       819328    
_________________________________________________________________
batch_normalization_8 (Batch (None, 64, 64, 128)       512       
_________________________________________________________________
leaky_re_lu_9 (LeakyReLU)    (None, 64, 64, 128)       0         
_________________________________________________________________
conv2d_transpose_4 (Conv2DTr (None, 128, 128, 64)      204864    
_________________________________________________________________
batch_normalization_9 (Batch (None, 128, 128, 64)      256       
_________________________________________________________________
leaky_re_lu_10 (LeakyReLU)   (None, 128, 128, 64)      0         
_________________________________________________________________
conv2d_transpose_5 (Conv2DTr (None, 128, 128, 3)       4803      
_________________________________________________________________
activation_2 (Activation)    (None, 128, 128, 3)       0         
=================================================================
Total params: 24,036,739
Trainable params: 24,034,819
Non-trainable params: 1,920
_________________________________________________________________
```

### Keras Results
<img src="https://raw.githubusercontent.com/dredwardhyde/Simpsons-DCGAN/master/results-keras.PNG" width="900"/>

### Losses
<img src="https://raw.githubusercontent.com/dredwardhyde/Simpsons-DCGAN/master/losses-keras.png" width="640"/>

### Performance

#### Keras backed by Tensorflow - nVidia GeForce RTX2080 Ti EVGA XC2 ULTRA
<img src="https://raw.githubusercontent.com/dredwardhyde/Simpsons-DCGAN/master/tensorflow-keras.PNG" width="900"/>

#### Keras backed by PlaidML - AMD Radeon RX Vega 64 Sapphire Nitro+
<img src="https://raw.githubusercontent.com/dredwardhyde/Simpsons-DCGAN/master/plaidml-keras.png" width="900"/>

### Tensorflow Results
<img src="https://raw.githubusercontent.com/dredwardhyde/Simpsons-DCGAN/master/results-tensorflow.PNG" width="900"/>

### Losses
<img src="https://raw.githubusercontent.com/dredwardhyde/Simpsons-DCGAN/master/losses-tensorflow.png" width="640"/>
