#library
import tensorflow as tf
from tensorflow.keras.applications import mobilenet
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, History
import numpy as np
import os, pickle


class ModelConfig:
    pass

class GenderAgeLoader:
    model_path="./app/weights/mobileNet.hdf5
    
    @staticmethod
    def multitask_activation(inputs):
        """
        Define multitask activation
        """
        sigmoid = K.sigmoid(inputs[:, 0])
        relu = K.relu(inputs[:, 1])
        result = K.stack([sigmoid, relu], axis=-1)
        return result
    
    @classmethod
    def build_age_gender_model(cls):
        model = mobilenet.MobileNet(include_top = False, input_shape = (160, 160, 3), weights = "imagenet"
                                    , pooling="avg")
        x = layers.Dense(2)(model.layers[-1].output)
        x = layers.Lambda(cls.multitask_activation)(x)
        
        return Model(model.layers[0].input, x)
        
    @classmethod
    def gender_age(cls):
        model = cls.build_age_gender_model()
        model.load_weights(cls.model_path)
        return model
        

class ResidualBlock(nn.Module):
    def __init__(self, features_in):
        super().__init__()
        
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(features_in, features_in, 3),
            nn.InstanceNorm2d(features_in),
            nn.ReLU(inplace = True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features_in, features_in, 3),
            nn.InstanceNorm2d(features_in)
        ]
        
        self.conv_block = nn.Sequential(*conv_block)
    
    def forward(self, x):
        return x + self.conv_block(x)
        
        
### GAN series for convert face by gender ###
### The below code build GAN model and load the weights ###

class Generator(nn.Module):
    def __init__(self, input_n, output_n, residual_blocks = 9):
        super().__init__()
        
        #Initial convolution block
        model = [ nn.ReflectionPad2d(3),
                 nn.Conv2d(input_n, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)
        ]
        
        #Downsampling
        features_in = 64
        features_out = features_in * 2
        for _ in range(2):
            model += [ nn.Conv2d(features_in, features_out, 3, stride = 2, padding = 1),
                       nn.InstanceNorm2d(features_out),
                       nn.ReLU(inplace = True)                
            ]
            features_in = features_out
            features_out = features_in * 2
        
        #Residual block
        for _ in range(residual_blocks):
            model += [ResidualBlock(features_in) 
            ]
        
        #Upsampling
        features_out = features_in // 2
        for _ in range(2):
            model += [ nn.ConvTranspose2d(features_in, features_out, 3, stride = 2, padding = 1, output_padding=1),
                       nn.InstanceNorm2d(features_out),
                       nn.ReLU(inplace = True)
            ]
            features_in = features_out
            features_out = features_in // 2
        
        #output layer
        model += [ nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_n, 7),
                  nn.Tanh()
                 ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


class GANLoader:
    
    def __init__(model_path):
        self.model_path = model_path
    
    @property
    def gen_model(self):
        gen_Model = Generator(input_n = 3, output_n =3)
        gen_Model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))

        gen_Model.eval()
        
        return gen_Model

    def get_model(self):
        return self.gen_model