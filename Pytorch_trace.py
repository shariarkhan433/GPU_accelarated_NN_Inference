import torch, numpy as np

imgs  = torch.tensor(np.load('weights/test_images.npy'))
fc1_w = torch.tensor(np.load('weights/fc1_w.npy'))
fc1_b = torch.tensor(np.load('weights/fc1_b.npy'))
fc2_w = torch.tensor(np.load('weights/fc2_w.npy'))
fc2_b = torch.tensor(np.load('weights/fc2_b.npy'))
fc3_w = torch.tensor(np.load('weights/fc3_w.npy'))
fc3_b = torch.tensor(np.load('weights/fc3_b.npy'))

x = imgs[0]
x = x @ fc1_w.T + fc1_b
print("fc1 pre-act  [0:3]:", x[:3].tolist())
x = torch.relu(x)
print("fc1 post-relu[0:3]:", x[:3].tolist())

x = x @ fc2_w.T + fc2_b
print("fc2 pre-act  [0:3]:", x[:3].tolist())
x = torch.relu(x)
print("fc2 post-relu[0:3]:", x[:3].tolist())

x = x @ fc3_w.T + fc3_b
print("fc3 pre-act  [0:3]:", x[:3].tolist())
x = torch.softmax(x, dim=0)
print("softmax (all 10):", x.tolist())

labels = np.load('weights/test_labels.npy')
print("Predicted:", x.argmax().item(), " True:", int(labels[0]))