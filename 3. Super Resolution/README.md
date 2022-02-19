# Table of Contents

-   [Dataset Preparation and Training](#dataset-preparation-and-training)

#### Task 1 - SRCNN and SRResNet

-   [Super Resolution using CNN (SRCNN)](#super-resolution-using-cnn-srcnn)
-   [Super Resolution using ResNets (SRResNet)](#super-resolution-using-resnets-srresnet)

#### Task 3 - Learnable Upsampling

-   [Super Resolution using CNN with Transposed Convolution](#super-resolution-using-cnn-with-transposed-convolution)
-   [Super Resolution using CNN with Sub-Pixel Convolution](#super-resolution-using-cnn-with-sub-pixel-convolution)
-   [Super Resolution using ResNets with Sub-Pixel Convolution](#super-resolution-using-resnets-with-sub-pixel-convolution)

#### Task 4 - Loss and Metrics

-   [MSE vs MAE](#mse-vs-mae)
-   [Thoughts on PSNR](#thoughts-on-psnr)
-   [Evaluation Metrics used in Recent Literature](#evaluation-metrics-used-in-recent-literature)

#### Footnotes

-   [Using a Shallower SRResNet](#using-a-shallower-srresnet)
-   [Using a Thinner CNN with Sub-Pixel Convolution](#using-a-thinner-srcnn-with-sub-pixel-convolution)

**Note**: Task 2 (interpolation comparison) has been done inside each individual implementation.

# Dataset Preparation and Training

The images were randomly cropped to size `196x196`. This gave the ground truth images.\
These ground images were blurred using gaussian blur, and downsampled to size `96x96`.\
Finally, the downsampled images were upsampled using bicubic interpolation to size `196x196`.

[Link to the Dataset](https://drive.google.com/drive/folders/1-612nU5UY8UaSFAfdGTqnZcUTBt1sosg?usp=sharing)\
Dataset size used for training: `3000 images`

Training was done with an early stopping callback, with the model finally using the weights of the best performing epoch.\
The saved models and their weights have been provided. The code to load them is given in the `srcnn.ipynb`/`srresnet.ipynb` files.

# Super Resolution using CNN (SRCNN)

This is the implementation of the model suggested by the paper: [Image Super-Resolution Using Deep Convolutional Networks - Dong, Loy et. al. (2015)](https://arxiv.org/pdf/1501.00092v3.pdf).

## Model Architecture

```python
Conv2D(filters=32, size=(9,9), activation='relu', padding='same')
Conv2D(filters=64, size=(5,5), activation='relu', padding='same')
Conv2D(filters=3, size=(5,5), activation='linear', padding='same')
```

loss: `mae`
metric: `PSNR`\
Total trainable params: `63,875`

<details>
    <summary><b>View model graph</b></summary>
    <img src='SRCNN/model graph.jpg' alt=' model graph'/>
</details>

## Hyperparameters

Filter count: `32-64-3`\
The linked SRCNN paper suggests that a filter count of `32-64-3` is a good compromise between the model's ability to capture information vs the speed of the training process.

Filter size: `9-5-5`\
Again, the linked SRCNN paper suggests that a filter size of `9-5-5` is optimal for allowing the model to pick up details from the dataset.

## Training Graph and Test PSNR

#### PSNR vs Epoch count

<image src='SRCNN/epoch graph.jpg' alt='srcnn-paper epoch graph'/>

Avg Test PSNR: `25.9 dB`

## Image Comparison

<details>
    <summary><b>Show Images where the model performed better than traditional upscaling</b></summary>
    <image src='SRCNN/image-comparison/cat_better.jpg' alt='cat_better'>
    <image src='SRCNN/image-comparison/cat_better2.jpg' alt='cat_better2'>
    <image src='SRCNN/image-comparison/dog_better.jpg' alt='dog_better'>
</details>

<details>
    <summary><b>Show Image where the model performed worse than traditional upscaling</b></summary>
    <image src='SRCNN/image-comparison/dog_worse.jpg' alt='dog_worse'>
    This is likely the result of the image being noisy, due to the pet's fur.
</details>

# Super Resolution using ResNets (SRResNet)

This is the implementation of a modified version of the SRCNN, with the addition of 6 ResNet blocks.

## Model Architecture

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
metric: `PSNR`\
Total trainable params: `78,819`

<details>
    <summary><b>View model graph</b></summary>
    <img src='SRResNet/model graph.jpg' alt=' model graph'/>
</details>

## Hyperparameters

Filter count: `16-16x3-32-32x3`\
The model performed better with a filter count of `16x3-32x3` instead `16x2-32x2`. [See footnote](#using-a-shallower-srresnet) for a comparison.\
Since the model is quite deep compared to the SRCNN, I chose filter counts of `16x3-32x3`, instead of `32x3-64x3` which would greatly increase the parameter count and slow down training.

Filter size: `9-3-3`\
Choosing bigger `5x5` filters would have greatly increased training time, hence mainly I went with `3x3` filters.

## Training Graph and Test PSNR

#### PSNR vs Epoch count

<image src='SRResNet/epoch graph.jpg' alt='srresnet6 epoch graph'/>

Avg Test PSNR: `25.95 dB`\
This is similar to the SRCNN.

## Image Comparison

<details>
    <summary><b>Show Images where the model performed better than traditional upscaling</b></summary>
    <image src='SRResNet/image-comparison/dog_better.jpg' alt='dog_better'>
    <image src='SRResNet/image-comparison/dog_better2.jpg' alt='dog_better2'>
</details>

<details>
    <summary><b>Show Image where the model performed worse than traditional upscaling</b></summary>
    <image src='SRResNet/image-comparison/cat_worse.jpg' alt='cat_worse'>
    Again, this is likely due to the image mainly consisting of noisy fur.
</details>

# Super Resolution using CNN with Transposed Convolution

This is the implementation of the SRCNN which uses learnable upsampling. This implementation accepts a low resolution `96x96` image, upscales it using transposed convolution, and then attempts to apply super resolution. This is unlike the SRCNN suggested by the paper, which accepts an image that is pre-upsampled by bicubic upsampling.

## Model Architecture

```python
# bring 96x96 -> 194x194
Conv2DTranspose(filters=64, size=3, strides=2, output_padding=1, activation='relu')

# bring 194x194 -> 196x196
Conv2DTranspose(filters=64, size=3, activation='relu')

Conv2D(filters=32, size=9, activation='relu', padding='same')
Conv2D(filters=32, size=5, activation='relu', padding='same')
Conv2D(filters=3, size=5, activation='linear', padding='same')
```

loss: `mae`
metric: `PSNR`\
Total trainable params: `232,675`

<details>
    <summary><b>View model graph</b></summary>
    <img src='SRCNN-TransConv/model graph.jpg' alt=' model graph'/>
</details>

## Hyperparameters

Filter count: `64x2-32x2-3`

Filter size: `3-3-9-5-5`\
I modified the previous `9-5-5` SRCNN and attached 2 Transposed Convolution layers at the start of it. I didn't attach the TransConv layers at the end of the network, because this operation adds zero values to the image, which may lead to vanishing gradients during backpropagation.

## Training Graph and Test PSNR

#### PSNR vs Epoch count

<image src='SRCNN-TransConv/epoch graph.jpg' alt=' epoch graph'/>

Avg Test PSNR: `23.74 dB`\
The avg test PSNR of this model is lower than the SRCNN/SRResNet models that use pre-upsampling. I was unable to get this model to perform better, in terms of PSNR, than even nearest neighbor upsampling.\
Despite this, the images produced by the model look much better than nearest neighbor upsampling, and sometimes better than bilinear upsampling. This shows that PSNR may not be the best metric to gauge super resolution performance.

## Image Comparison

<details>
    <summary><b>Show Images</b></summary>
    <image src='SRCNN-TransConv/image-comparison/cat.jpg' alt='cat'>
    <image src='SRCNN-TransConv/image-comparison/dog.jpg' alt='dog'>
    <image src='SRCNN-TransConv/image-comparison/dog2.jpg' alt='dog2'>
</details>

# Super Resolution using CNN with Sub-Pixel Convolution

This is the implementation of the SRCNN which uses learnable upsampling. This implementation accepts a low resolution `96x96` image, upscales it using sub-pixel convolution, and then attempts to apply super resolution. This is unlike the SRCNN suggested by the paper, which accepts an image that is pre-upsampled by bicubic upsampling.

## Model Architecture

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
metric: `PSNR`\
Total trainable params: `214,892`

<details>
    <summary><b>View model graph</b></summary>
    <img src='SRCNN-SubPixel/model graph.jpg' alt=' model graph'/>
</details>

## Hyperparameters

Filter count: `32-64-64-12`\
I used `32-64-64` as opposed to a thinner `16-32-32` model, as the former performed better than the latter. [See footnote](#using-a-thinner-cnn-with-sub-pixel-convolution) for a comparison.

Filter size: `3-9-5-5-5`\
I used the previous `9-5-5` SRCNN. The network has a Transposed Convolution layer attached at the start of it to increase the image size to `98x98`. At the end, a Sub-Pixel Convolution layer is added to upscale the image by `2x`.

## Training Graph and Test PSNR

#### PSNR vs Epoch count

<image src='SRCNN-SubPixel/epoch-graph.jpg' alt=' epoch graph'/>

Avg Test PSNR: `24.56 dB`\
Though much better than the SRCNN with Transposed convolution, the avg test PSNR of this model is also lower than the SRCNN/SRResNet models that use pre-upsampling. The PSNR values are close to the nearest neighbor upsampled images.\
Yet again, the images produced by the model look much better than nearest neighbor upsampling.

## Image Comparison

<details>
    <summary><b>Show Images</b></summary>
    <image src='SRCNN-SubPixel/image-comparison/cat.jpg' alt='cat'>
    <image src='SRCNN-SubPixel/image-comparison/dog.jpg' alt='dog'>
    <image src='SRCNN-SubPixel/image-comparison/dog2.jpg' alt='dog2'>
</details>

# Super Resolution using ResNets with Sub-Pixel Convolution

This is the implementation of the SRResNet which uses learnable upsampling. This implementation accepts a low resolution `96x96` image, upscales it using sub-pixel convolution, and then attempts to apply super resolution. This is unlike the SRResnet shown earlier, which accepts an image that is pre-upsampled by bicubic upsampling.

## Model Architecture

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
metric: `PSNR`\
Total trainable params: `128,192`

This model architecture is similar to the one given in [An Evolution in Single Image Super Resolution using Deep Learning](https://towardsdatascience.com/an-evolution-in-single-image-super-resolution-using-deep-learning-66f0adfb2d6b#6cb9).

<details>
    <summary><b>View model graph</b></summary>
    <img src='SRResNet-SubPixel/model graph.jpg' alt=' model graph'/>
</details>

## Hyperparameters

Filter count: `16-16x3-32-32x3`\
Since the network is quite deep, using a wider `32x3-64x3` filter count would significantly increase training time.

Filter size: `9-3-5-5`\
I simply added resnet layers to the sub-pixel SRCNN model, since it performed well. The network has a Transposed Convolution layer attached at the start of it to increase the image size to `98x98`. At the end, a Sub-Pixel Convolution layer is added to upscale the image by `2x`.

## Training Graph and Test PSNR

#### PSNR vs Epoch count

<image src='SRResNet-SubPixel/epoch graph.jpg' alt='srres6-subpix epoch graph'/>

Avg Test PSNR: `25.32 dB`\
This is a good improvement over the SRCNN with sub-pixel convolution, which itself was an improvement over the SRCNN with transposed convolution. The avg test PSNR of this model is still slightly lower than the SRCNN/SRResNet models that use pre-upsampling. But, it is at least able to beat bicubic interpolation, in terms of PSNR, in some of the cases.

## Image Comparison

<details>
    <summary><b>Show Images where the model performed better than traditional upscaling</b></summary>
    <image src='SRResNet-SubPixel/image-comparison/cat_better.jpg' alt='cat_better'>
    <image src='SRResNet-SubPixel/image-comparison/cat_better2.jpg' alt='cat_better2'>
</details>

<details>
    <summary><b>Show Image where the model performed worse than traditional upscaling</b></summary>
    <image src='SRResNet-SubPixel/image-comparison/cat_worse.jpg' alt='cat_worse'>
    <image src='SRResNet-SubPixel/image-comparison/cat_worse2.jpg' alt='cat_worse2'>
</details>

# Loss and Metrics

## MSE vs MAE

According to [Deep learning image enhancement insights on loss function engineering](https://towardsdatascience.com/deep-learning-image-enhancement-insights-on-loss-function-engineering-f57ccbb585d7#fcd8), Mean Squared Error (MSE) is prone to being affected by outliers, and has poor perceptual quality for a human viewer.

> A loss function that minimizes MSE encourages finding pixel averages of plausible solutions that are typically overly smoothed and although minimizing the loss, the generated images will have poor perceptual quality from a perspective of appealing to a human viewer.

Thus, I've instead used Mean Absolute Error (MAE), which according to [the same article](https://towardsdatascience.com/deep-learning-image-enhancement-insights-on-loss-function-engineering-f57ccbb585d7#ad7d), provides better perceptual quality.

> For Image Enhancement, MAE will likely result in an image which appears to be a higher quality from a human viewer’s perspective.

## Thoughts on PSNR

I've used Peak Signal to Noise Ratio (PSNR) to gauge the performance of all my implemented models, and compare them with traditional methods of upsampling.

PSNR seems to be a good metric for gauging the epoch training performance of a super resolution model.\
However, it is not very accurate for gauging the perceptual quality of a reconstructed image. This can be clearly seen in the [Super Resolution using CNN with Transposed Convolution](#super-resolution-using-cnn-with-transposed-convolution) and [Super Resolution using CNN with Sub-Pixel Convolution](#super-resolution-using-cnn-with-sub-pixel-convolution) models. These models yield a worse PSNR performance when compared to Nearest Neighbor (NN) upsampling. Yet, the images produced by them are perceptually better than those of NN upsampling.

## Evaluation Metrics used in Recent Literature

Since MSE/MAE/PSNR compute per-pixel distances, they cannot be expected to gauge perceptual performance like a human viewer would.

**Feature Loss** - This metric uses a pre-trained CNN to extract features from images in the form of feature maps. The Super Resolution model's output, and the ground truth images, are sent through this feature extractor and the euclidean distance between the feature maps is used to compute the loss. This metric compares high level differences such as content and style discrepancies, something that a human viewer would do too.

**Gram Loss** - At a high level, this metric aims to find the correlation between the features in the channels of an image. It computes style loss, and can be used for style transfer as well as recoloration of images. For example, this metric can help a model associate a certain texture with a certain range of color.

# Footnotes

# Using a Shallower SRResNet

This is the implementation of a shallower (i.e. a lesser number of ResNet layers) version of the SRResNet. The model uses a filter count of `16x2-32x2`, instead of `16x3-32x3`.

## Model Architecture

```python
# Typical ResNet blocks are used, with 2 convolution layers and a skip.
Conv2D(filters=16, size=(9,9), activation='relu', padding='same')
ResNetBlock(filters=16, size=(3,3), activation='relu', padding='same')
ResNetBlock(filters=16, size=(3,3), activation='relu', padding='same')

Conv2D(filters=32, size=(3,3), activation='relu', padding='same')
ResNetBlock(filters=32, size=(3,3), activation='relu', padding='same')
ResNetBlock(filters=32, size=(3,3), activation='relu', padding='same')

Conv2D(filters=3, size=(3,3), activation='linear', padding='same')
```

loss: `mae`
metric: `PSNR`\
Total trainable params: `55,683`

<details>
    <summary><b>View model graph</b></summary>
    <img src='SRResNet-Shallow/model graph.jpg' alt=' model graph'/>
</details>

## Hyperparameters

Filter count: `16-16x2-32-32x2`

Filter size: `9-3-3`\
This is the same as the deeper SRRestNet.

## Training Graph and Test PSNR

#### PSNR vs Epoch count

<image src='SRResNet-Shallow/epoch graph.jpg' alt=' epoch graph'/>\
The validation PSNR plot hovers around the `26.0 dB` mark.

#### Comparison with the Deeper Version

<image src='SRResNet/epoch graph.jpg' alt=' epoch graph'/>\
The validation PSNR plot crosses the `26.0 dB` mark, and even touches the `26.5 dB` mark.

Avg Test PSNR: `25.63 dB` (vs `25.95 dB` of the deeper version.)\
Thus, the difference between this model and its deeper counterpart is not too pronounced. It performs slightly worse than its deeper counterpart.

# Using a Thinner SRCNN with Sub-Pixel Convolution

This is the implementation of a thinner (i.e. a lesser number of filters per convolution layer) version of the SRCNN which uses sub-pixel convolution. The model uses a filter count of `16-32-32`, instead of `32-64-64`.

## Model Architecture

```python
Conv2DTranspose(filters=16, size=3, activation='relu')  # increase 96x96 -> 98x98

Conv2D(filters=16, size=(9,9), activation='relu', padding='same')
Conv2D(filters=32, size=(5,5), activation='relu', padding='same')
Conv2D(filters=32, size=(5,5), activation='relu', padding='same')

# sub-pixel conv
Conv2D(filters=3 * (2 ** 2), size=(5,5), activation='relu', padding='same')(x)
DepthToSpace(upscale_factor=2)
```

loss: `mae`
metric: `PSNR`\
Total trainable params: `69,276`

<details>
    <summary><b>View model graph</b></summary>
    <img src='SRCNN-SubPixel-Thin/model graph.jpg' alt=' model graph'/>
</details>

## Hyperparameters

Filter count: `16-32-32-12`

Filter size: `3-9-5-5-5`\
This is the same as the wider SRCNN.

## Training Graph and Test PSNR

#### PSNR vs Epoch count

<image src='SRCNN-SubPixel-Thin/epoch graph.jpg' alt=' epoch graph'/>\
The validation PSNR plot hovers around the `24.0 dB` mark.

#### Comparison with the Wider Version

<image src='SRCNN-SubPixel/epoch-graph.jpg' alt=' epoch graph'/>\
The validation PSNR plot clearly crosses the `24.0 dB` mark.

Avg Test PSNR: `24.03 dB` (vs `24.56 dB` of the wider version.)\
Thus, this model architecture performs worse than its wider counterpart.
