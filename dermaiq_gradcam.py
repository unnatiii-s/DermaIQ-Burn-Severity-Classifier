import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load model
model = load_model("models/dermaiq_trained_model.h5")

# Load image
img_path = "test_images/sample_burn.jpg"  # Replace with your image
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

# Grad-CAM model
grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[
        model.get_layer("Conv_1_bn").output,
        model.output  # <-- Keep it exactly like this
    ]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    class_index = tf.argmax(predictions[0])
    loss = predictions[:, class_index]

# Compute Grad-CAM
grads = tape.gradient(loss, conv_outputs)[0]
pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
conv_outputs = conv_outputs[0]
heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

# Normalize
heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap)
heatmap = cv2.resize(heatmap.numpy(), (224, 224))
heatmap = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Overlay
original_img = cv2.imread(img_path)
original_img = cv2.resize(original_img, (224, 224))
superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

# Display
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title(f"Grad-CAM for class {class_index.numpy()}")
plt.axis("off")
plt.tight_layout()
plt.savefig("gradcam_output.png")
plt.show()