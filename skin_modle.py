import tensorflow as tf 
from PIL import Image
import numpy as np
import os 
load_model = tf.keras.models.load_model
KerasDepthwiseConv2D = tf.keras.layers.DepthwiseConv2D



class RandomWidth(tf.keras.layers.Layer):
    def __init__(self, factor=0.0, interpolation='bilinear', seed=None, **kwargs):
        super().__init__(**kwargs) 
        self.factor = factor
        self.interpolation = interpolation
        self.seed = seed
        
    def call(self, inputs):
        return inputs
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "factor": self.factor,
            "interpolation": self.interpolation,
            "seed": self.seed,
        })
        return config

class RandomHeight(tf.keras.layers.Layer):
    def __init__(self, factor=0.0, interpolation='bilinear', seed=None, **kwargs):
        super().__init__(**kwargs) 
        self.factor = factor
        self.interpolation = interpolation
        self.seed = seed
        
    def call(self, inputs):
        return inputs
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "factor": self.factor,
            "interpolation": self.interpolation,
            "seed": self.seed,
        })
        return config

class FixedDepthwiseConv2D(KerasDepthwiseConv2D):
    
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)



print("Loading your own model: my_model.h5")


custom_objects = {
    'RandomWidth': RandomWidth,
    'RandomHeight': RandomHeight,
    'DepthwiseConv2D': FixedDepthwiseConv2D
}
try:
   
    model = load_model("my_model.h5", custom_objects=custom_objects)
    print("Your model loaded successfully!")
except Exception as e:
    print(f"!!! FATAL ERROR LOADING MODEL: {e}")
    
    raise e 


CLASS_NAMES = [
    "Eczema", "Melanoma", "Atopic Dermatitis", "Basal Cell Carcinoma", 
    "Melanocytic Nevi (NV)", "Benign Keratosis", "Psoriasis", 
    "Seborrheic Keratoses", "Tinea Ringworm Candidiasis", "Warts Molluscum"
]

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

def predict_skin_disease(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed, verbose=0) 
    
    idx = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][idx] * 100
    
    disease = CLASS_NAMES[idx]
    
    return disease, confidence