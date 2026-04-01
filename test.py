import numpy as np
imgs = np.load('weights/test_images.npy')
labels = np.load('weights/test_labels.npy')
print("images shape:", imgs.shape)
print("images mean:", imgs.mean())
print("images std:", imgs.std())
print("first label:", labels[0])
print("first image sample:", imgs[0, :5])
