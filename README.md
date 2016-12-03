# Generative Adversarial Network

A [Generative Adversarial Network](https://arxiv.org/pdf/1406.2661v1.pdf) that trains on mnist

- `python3 gan/main.py` - train the network
- `tensorboard --logdir runs` - track the training progress

This implementation has a slight improvement over the paper in that it dynamically balances the training regimes between the generator and the discriminator by ensuring the discriminator accuracy is never too far away from 50%
