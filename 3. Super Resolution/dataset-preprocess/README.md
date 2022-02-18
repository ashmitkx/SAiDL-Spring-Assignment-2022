# Dataset Preprocessing

These files were used to process the Oxford-IIIT Pet dataset.\
[Link to the processed dataset.](https://drive.google.com/drive/folders/1-612nU5UY8UaSFAfdGTqnZcUTBt1sosg?usp=sharing)  

`preproc-random-crop.ipynb` takes random `196x196` crops from the pet dataset.\
`preproc-blur-downsample.ipynb` blurs the crops and downsamples them to `96x96`.\
`preproc-upsample.ipynb` upsamples the downsampled blurred images to `196x196`.

