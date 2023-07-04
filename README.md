# UNET-MultiClass-Segmentation
segmentation  of multiple classes 

## To run UNet- VGG

    python3 train.py --epochs 50 --batch-size 4 --imsize 224 224 --classes 3 --augmentations --unet-vgg

* Import image size should be 224
* No need to use --amp , --dropout, --bilinear flag
    

## To run UNet 
    python3 train.py --epochs 50 --batch-size 4 --imsize 968 732 --amp --classes 3 --dropout --augmentations --learning-rate 0.00001

* Use learning rate should be 0.00001