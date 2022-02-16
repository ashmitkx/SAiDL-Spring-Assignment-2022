<!--
#

This is the implementation of

## Model Architechture

```python

```

loss: `mae`
metric: `PSNR`
Total trainable params: ``

<details>
    <summary>View model graph</summary>
    <img src='/model graph.png' alt=' model graph'/>
</details>

## Hyperparamteres

Filter count: ``

Filter size: ``

## Training Graph and Test PSNR

#### PSNR vs Epoch count

<image src='\epoch graph.jpg' alt=' epoch graph'/>

Training was done with an early stopping callback, with the model finally using the weights of the best performing epoch.

Avg Test PSNR: ``

## Image Comparison

<details>
    <summary>Show Images where the model performed better than traditional upscaling</summary>
    <image src='' alt=''>
</details>

<details>
    <summary>Show Image where the model performed worse than traditional upscaling</summary>
    <image src='' alt=''>
</details>

-->

# Dataset Preparation

The images were randomly cropped to size `196x196`. This gave the ground truth images.
These ground images were blurred using gaussian blur, and downsampled to size `96x96`.
Finally, the downsampled images were upsampled using bicubic interpolation to size `196x196`.

Dataset size: `3000 images`

# Super Resolution using CNN (SRCNN)

This is the implementation of the model suggested by the paper: [Image Super-Resolution Using Deep Convolutional Networks - Dong, et. al.](https://arxiv.org/pdf/1501.00092v3.pdf)

## Model Architechture

```python
Conv2D(filters=32, size=(9,9), activation='relu', padding='same')
Conv2D(filters=64, size=(5,5), activation='relu', padding='same')
Conv2D(filters=3, size=(5,5), activation='linear', padding='same')
```

loss: `mae`
metric: `PSNR`
Total trainable params: `63,875`

<details>
    <summary>View model graph</summary>
    <img src='srcnn-paper/model graph.jpg' alt=' model graph'/>
</details>

## Hyperparamteres

Filter count: `32-64-3`
The linked SRCNN paper suggests that a filter count of 32-64-3 is a good compromise between the model's ability to capture information vs the speed of the training process.

Filter size: `9-5-5`
Again, the linked SRCNN paper suggests that a filter size of 9-5-5 is optimal for allowing the model to pick up details from the dataset.

## Training Graph and Test PSNR

#### PSNR vs Epoch count

<image src='srcnn-paper\epoch graph.jpg' alt='srcnn-paper epoch graph'/>

Training was done with an early stopping callback, with the model finally using the weights of the best performing epoch.

Avg Test PSNR: `25.9 dB`

## Image Comparison

<details>
    <summary>Show Images where the model performed better than traditional upscaling</summary>
    <image src='srcnn-paper\image-comparison\cat_better.jpg' alt='cat_better'>
    <image src='srcnn-paper\image-comparison\cat_better2.jpg' alt='cat_better2'>
    <image src='srcnn-paper\image-comparison\dog_better.jpg' alt='dog_better'>
</details>

<details>
    <summary>Show Image where the model performed worse than traditional upscaling</summary>
    <image src='srcnn-paper\image-comparison\dog_worse.jpg' alt='dog_worse'>
    This is likely the result of the image being noisy, due to the pet's fur.
</details>


# Super Resolution using ResNets (SRResNet)

This is the implementation of a modified version of the SRCNN, with the addition of 6 ResNet blocks.

## Model Architechture

```python
# Typical ResNet blocks are used, with 2 convolution layers and a skip.
Conv2D(filters=16, size=(9,9), activation='relu', padding='same')
ResNetBlock(filters=16, size=(3,3), activation='relu', padding='same')
ResNetBlock(filters=16, size=(3,3), activation='relu', padding='same')
ResNetBlock(filters=16, size=(3,3), activation='relu', padding='same')

Conv2D(filters=32, size=(3,3), activation='relu', padding='same')
ResNetBlock(filters=32, size=(3,3), activation='relu', padding='same')
ResNetBlock(filters=32, size=(3,3), activation='relu', padding='same')
ResNetBlock(filters=32, size=(3,3), activation='relu', padding='same')

Conv2D(filters=3, size=(3,3), activation='linear', padding='same')
```

loss: `mae`
metric: `PSNR`
Total trainable params: `78,819`

<details>
    <summary>View model graph</summary>
    <img src='srres6-mae/model graph.jpg' alt=' model graph'/>
</details>

## Hyperparamteres

Filter count: `16-32`
Since the model is quite deep compared to the SRCNN, I chose filter counts of 16-32, instead of 32-64. Reducing the depth (number of ResNet blocks) and increasing the width (filter count) slowed down training and gave worse results. <!-- TODO: Add Evidence -->

Filter size: `9-3`
Choosing bigger 5x5 filters would've greatly increased training time, hence mainly I went with 3x3 filters.

## Training Graph and Test PSNR

#### PSNR vs Epoch count
<image src='srres6-mae\epoch graph.jpg' alt='srresnet6 epoch graph'/>

Training was done with an early stopping callback, with the model finally using the weights of the best performing epoch.

Avg Test PSNR: `25.95 dB`
This is similar to the SRCNN.

## Image Comparison

<details>
    <summary>Show Images where the model performed better than traditional upscaling</summary>
    <image src='srres6-mae\image-comparison\dog_better.jpg' alt='dog_better'>
    <image src='srres6-mae\image-comparison\dog_better2.jpg' alt='dog_better2'>
</details>

<details>
    <summary>Show Image where the model performed worse than traditional upscaling</summary>
    <image src='srres6-mae\image-comparison\cat_worse.jpg' alt='cat_worse'>
    Again, this is likely due the image mainly consisting of noisy fur.
</details>

# Super Resolution using CNN with Transposed Convolution

This is the implementation of the SRCNN which uses learnable upsampling. This implementation accepts a low resolution 96x96 image, upscales it using transposed convolution, and then attempts to apply super resolution. This is unlike the SRCNN suggested by the paper, which accepts an image that is pre-upsampled by bicubic upsampling.

## Model Architechture

```python
# bring 96x96 -> 194x194
Conv2DTranspose(filters=64, size=3, strides=2, output_padding=1, activation='relu') 
# bring 194x194 -> 196x196
Conv2DTranspose(64, size=3, activation='relu')
Conv2D(filters=32, size=9, activation='relu', padding='same')
Conv2D(filters=32, size=5, activation='relu', padding='same')
Conv2D(filters=3, size=5, activation='linear', padding='same')
```

loss: `mae`
metric: `PSNR`
Total trainable params: `232,675`

## Hyperparamteres

Filter count: `64-32-3` 

Filter size: `3-3-9-5-5`
I used the previous 9-5-5 SRCNN and attached 2 Transposed Convolution layers at the start of it. I attached the TransConv layers at the start of the network, because this operation adds zero values to the image, which may lead to vanishing gradients during backpropagation. 

## Training Graph and Test PSNR

#### PSNR vs Epoch count

<image src='srcnn-paper-trans\epoch graph.jpg' alt=' epoch graph'/>

Training was done with an early stopping callback, with the model finally using the weights of the best performing epoch.

Avg Test PSNR: `23.74 dB`
The avg test PSNR of this model is lower than the SRCNN/SRResNet models that use pre-upsampling. I was unable to get this model to perform better, in terms of PSNR, than even nearest neighbour upsampling. 
Despite this, the images produced by the model look much better than nearest neighbour upsampling, and sometimes better than bilinear upsampling. This shows that PSNR may not be the best metric to gauge super resolution performance. 

## Image Comparison

<details>
    <summary>Show Images</summary>
    <image src='srcnn-paper-trans/image-comparison/cat.jpg' alt='cat'>
    <image src='srcnn-paper-trans/image-comparison/dog.jpg' alt='dog'>
    <image src='srcnn-paper-trans/image-comparison/dog2.jpg' alt='dog2'>
</details>


# Super Resolution using CNN with Sub-Pixel Convolution

This is the implementation of the SRCNN which uses learnable upsampling. This implementation accepts a low resolution 96x96 image, upscales it using sub-pixel convolution, and then attempts to apply super resolution. This is unlike the SRCNN suggested by the paper, which accepts an image that is pre-upsampled by bicubic upsampling.

## Model Architechture

```python
Conv2DTranspose(filters=16, size=3, activation='relu')  # increase 96x96 -> 98x98

Conv2D(filters=32, size=(9,9), activation='relu', padding='same')
Conv2D(filters=64, size=(5,5), activation='relu', padding='same')
Conv2D(filters=64, size=(5,5), activation='relu', padding='same')

# sub-pixel conv
Conv2D(filters=3 * (2 ** 2), size=(5,5), activation='relu', padding='same')(x)
DepthToSpace(upscale_factor=2)
```

loss: `mae`
metric: `PSNR`
Total trainable params: `214,892`

<details>
    <summary>View model graph</summary>
    <img src='srcnn-paper-subpix-3\model graph.jpg' alt=' model graph'/>
</details>

## Hyperparamteres

Filter count: `32-64-64`
I used 32-64-64 as opposed to a thinner 16-32-32 model, as the former performed better than the latter. <!-- TODO: Add evidence -->

Filter size: `3-9-5-5-12`
I used the previous 9-5-5 SRCNN. The network has a Transposed Convolution layer attached at the start of it to increase the image size to 98x98. At the end, a Sub-Pixel Convolution layer is added to upscale the image by 2x.


## Training Graph and Test PSNR

#### PSNR vs Epoch count

<image src='srcnn-paper-subpix-3\epoch-graph.jpg' alt=' epoch graph'/>

Training was done with an early stopping callback, with the model finally using the weights of the best performing epoch.

Avg Test PSNR: `24.56 dB`
Though much better than the SRCNN with Transposed convolution, the avg test PSNR of this model is also lower than the SRCNN/SRResNet models that use pre-upsampling. The PSNR values are close to the nearest neighbour upsampled images.
Yet again, the images produced by the model look much better than nearest neighbour upsampling.

## Image Comparison

<details>
    <summary>Show Images</summary>
    <image src='srcnn-paper-subpix-3\image-comparison\cat.jpg' alt='cat'>
    <image src='srcnn-paper-subpix-3\image-comparison\dog.jpg' alt='dog'>
    <image src='srcnn-paper-subpix-3\image-comparison\dog2.jpg' alt='dog2'>
</details>

# Super Resolution using ResNets with Sub-Pixel Convolution

This is the implementation of the SRResNet which uses learnable upsampling. This implementation accepts a low resolution 96x96 image, upscales it using sub-pixel convolution, and then attempts to apply super resolution. This is unlike the SRResnet shown earlier, which accepts an image that is pre-upsampled by bicubic upsampling.

## Model Architechture

```python
# Typical ResNet blocks are used, with 2 convolution layers and a skip.
Conv2DTranspose(filters=8, size=(3,3), activation='relu')  # increase 96x96 -> 98x98
Conv2D(filters=16, size=(9,9), activation='relu', padding='same')

AddSkip(
    ResNetBlock(filters=16, size=(3,3), activation='relu', padding='same')
    ResNetBlock(filters=16, size=(3,3), activation='relu', padding='same')
    ResNetBlock(filters=16, size=(3,3), activation='relu', padding='same')

    Conv2D(filters=32, size=(3,3), activation='relu', padding='same')
    ResNetBlock(filters=32, size=(3,3), activation='relu', padding='same')
    ResNetBlock(filters=32, size=(3,3), activation='relu', padding='same')
    ResNetBlock(filters=32, size=(3,3), activation='relu', padding='same')

    Conv2D(filters=16, size=(5,5), activation='relu', padding='same')
)

Conv2D(filters=32, size=(5,5), activation='relu', padding='same')

# sub-pixel conv
Conv2D(filters=3 * (2 ** 2), size=(5,5), activation='relu', padding='same')(x)
DepthToSpace(upscale_factor=2)

Conv2D(filters=3, size=(3,3), activation='linear', padding='same')
```

loss: `mae`
metric: `PSNR`
Total trainable params: `128,192`

<details>
    <summary>View model graph</summary>
    <img src='srrres6-subpix\model graph.jpg' alt=' model graph'/>
</details>

## Hyperparamteres

Filter count: `8-16-32-12`
Since the network is quite deep, using a wider 32-64 filter count would significantly increase training time.

Filter size: `9-3-5-5-12`
I simply added resnet layers to the sub-pixel SRCNN model, since it performed well. The network has a Transposed Convolution layer attached at the start of it to increase the image size to 98x98. At the end, a Sub-Pixel Convolution layer is added to upscale the image by 2x.

## Training Graph and Test PSNR

#### PSNR vs Epoch count

<image src='srrres6-subpix\epoch graph.jpg' alt='srres6-subpix epoch graph'/>

Training was done with an early stopping callback, with the model finally using the weights of the best performing epoch.

Avg Test PSNR: `25.32 dB`
This is a good improvement over the SRCNN with sub-pixel convolution, which itself was an improvement over the SRCNN with transposed convolution. The avg test PSNR of this model is still slightly lower than the SRCNN/SRResNet models that use pre-upsampling. But, it is atleast able to beat bicubic interpolation, in terms of PSNR, in some of the cases.

## Image Comparison

<details>
    <summary>Show Images where the model performed better than traditional upscaling</summary>
    <image src='srrres6-subpix/image-comparison/cat_better.jpg' alt='cat_better'>
    <image src='srrres6-subpix/image-comparison/cat_better2.jpg' alt='cat_better2'>
</details>

<details>
    <summary>Show Image where the model performed worse than traditional upscaling</summary>
    <image src='srrres6-subpix/image-comparison/cat_worse.jpg' alt='cat_worse'>
    <image src='srrres6-subpix/image-comparison/cat_worse2.jpg' alt='cat_worse2'>
</details>
