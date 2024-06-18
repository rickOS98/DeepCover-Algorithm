import numpy as np
import cv2
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt

class DeepCover:
    def __init__(self, model_weights='imagenet', layer_name='block5_conv3'):
        self.model = VGG16(weights=model_weights)
        self.layer_name = layer_name

    def preprocess_img(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def apply_gradcam(self, img_array, class_idx=None):
        grad_model = Model([self.model.inputs], [self.model.get_layer(self.layer_name).output, self.model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if class_idx is None:
                class_idx = np.argmax(predictions[0])
            class_channel = predictions[:, class_idx]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy(), predictions

    def superimpose_heatmap(self, img_path, heatmap):
        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        return superimposed_img

    def classify_and_explain(self, img_path):
        img_array = self.preprocess_img(img_path)
        heatmap, predictions = self.apply_gradcam(img_array)

        # Calculate responsibilities (simplified example)
        superpixels = {
            'A': img_array[0, :112, :112, :],
            'B': img_array[0, :112, 112:, :],
            'C': img_array[0, 112:, :112, :],
            'D': img_array[0, 112:, 112:, :]
        }

        responsibilities = {}
        for name, sp in superpixels.items():
            intensity = np.mean(sp)
            if intensity > 0.5:
                responsibility = 1 / (sp.size + 1)
            else:
                responsibility = 0
            responsibilities[name] = responsibility
        
        return responsibilities, predictions, heatmap

    def display_heatmap(self, img_path, heatmap):
        superimposed_img = self.superimpose_heatmap(img_path, heatmap)
        plt.imshow(superimposed_img)
        plt.axis('off')
        plt.show()

# Example usage:
if __name__ == "__main__":
    deep_cover = DeepCover()
    img_path = 'path_to_your_image.jpg'
    responsibilities, predictions, heatmap = deep_cover.classify_and_explain(img_path)

    # Print responsibilities and classification
    print("Responsibilities:", responsibilities)
    print("Predicted class:", decode_predictions(predictions, top=1)[0])

    # Display heatmap
    deep_cover.display_heatmap(img_path, heatmap)