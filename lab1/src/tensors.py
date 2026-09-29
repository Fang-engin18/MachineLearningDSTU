import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
X_tf = tf.constant(data, dtype=tf.float32)

import torch
X_pt = torch.tensor(data, dtype=torch.float32)

n, m = X_tf.shape
k, p = 5, 4

A = tf.random.uniform((k, n), minval=0, maxval=10, dtype=tf.int32)
W = tf.random.normal((m, p))
B = tf.random.uniform((k, p))

AX = tf.matmul(tf.cast(A, tf.float32), X_tf)
AXW = tf.matmul(AX, W)
result_tf = AXW + B

A_pt = torch.randint(0, 10, (k, n), dtype=torch.float32)
W_pt = torch.randn(m, p)
B_pt = torch.rand(k, p)

result_pt = A_pt @ X_pt @ W_pt + B_pt

T = np.random.rand(3, 10)
P = np.random.rand(3, 10)
Q = np.random.rand(3, 10)

import keras
from keras import ops

T_k = ops.convert_to_tensor(T)
P_k = ops.convert_to_tensor(P)
Q_k = ops.convert_to_tensor(Q)

V_k = ops.abs(ops.sin(T_k) - ops.exp(P_k) * ops.sqrt(Q_k))

T_t = torch.tensor(T)
P_t = torch.tensor(P)
Q_t = torch.tensor(Q)
V_t = torch.abs(torch.sin(T_t) - torch.exp(P_t) * torch.sqrt(Q_t))

