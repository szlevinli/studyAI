# ResNet-Tensorflow
Simple Tensorflow implementation of ***pre-activation*** ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
## Summary
### dataset
* [tiny_imagenet](https://tiny-imagenet.herokuapp.com/)
* cifar10, cifar100, mnist, fashion-mnist in `keras` 
(`pip install tensorflow==1.8`)
(`pip install keras=2.1.5`)

### Train
* python main.py --phase train --dataset mnist --res_n 18 --lr 0.1

### Test
* python main.py --phase test --dataset mnist --res_n 18 --lr 0.1
