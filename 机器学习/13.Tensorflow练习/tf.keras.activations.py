import tensorflow as tf
import matplotlib.pyplot as plt

print("relu alpha: ", tf.keras.activations.relu(-0.1, alpha=0.5))
print("relu alpha: ", tf.keras.activations.relu(0.5, alpha=0.5))
