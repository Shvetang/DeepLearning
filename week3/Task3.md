# Assignment 3    
    
## Loading the dataset       
    
The zip file was uploaded to google drive then extracted on colab    
Then the contents of all folders were shifted to the parent directory to pass to the ImageFolder class of torchvision.datasets using shutil and os modules        
All images were resized and cropped to 64x64 pixels   
The pixel values were resized with a mean and standard deviation of 0.5 for each channel (pixels fall in range [-1, 1] for each channel which helps train D)   
   

## Specs of the discriminator and generator (D and G)     
     
In accordance with the [paper](https://arxiv.org/pdf/1511.06434),    
1. There are no pooling layers, only convolution layers in both D and G    
2. Use batchnorm in both G and D    
3. Remove fully connected hidden layers for deeper architectures     
4. Use ReLU activation in G for all layers except for the output, which uses Tanh    
5. Use LeakyReLU activation (with leak=0.2) in D for all layers    
6. Fractionally-strided convolutions are implemeted using nn.ConvTranspose2d    
    
Also learning rate kept at `lr = 0.0002`      
All models were trained with a batch size of **128**     
  
Both models were moved to the GPU   
   
d = (d−1)xS − 2P + K for upsampling   
d = (d−K+2P)/S+ 1 for downsampling   
where d is the dimension, S is the stride, P is the padding and K is the kernel size    
From these formulae it is clear that the size of the image exactly doubles during upsampling in the generator and becomes exactly half during downsampling in the discriminator     
This justifies the choice of 64x64 for the image size 
  

## Training the GAN  
  
During each epoch, both the generator and discriminator are trained together  
  
Each time, batches of real images pushed to the GPU are sent to the discriminator from which the loss is calculated by setting the target as a tensor full of ones   
Also, batches of images produced by the generator from random inputs are fed in from which another loss is calculated by setting a target full of zeros  

For the generator, random inputs are fed in batch-wise, which are then passed into the discriminator, and the loss is calculated using the prediction of the discriminator and the target is a tensor full of ones (because we wish to fool the discriminator)  

In either case, once the losses are calculated, the new gradients are calculated using Stochastic Gradient Descent (SGD) and updated   
   
At the end of each epoch, the images produced by the generator are saved  
    
