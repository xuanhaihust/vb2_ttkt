import numpy as np
import time

from tensorflow.keras.applications import mobilenet

from app.model.model_loader import ModelLoader
from app.model.image_loader import ImageLoader


#  predict
class AgeGenderModel:
    def __init__(self, threshold=0.5):
        self.model = ModelLoader.gender_age()
        self.threshold = threshold

    def result_infer(self, pred) -> dict:
        """
        Result inference and contructor
        """
        prob, age = pred[0]
        gender = "Male"  # default
        
        if prob < self.threshold:
            gender = "Female"
            prob = 1 - prob
            
        results = {"gender": gender,
                   "age": age,
                   "confidence": prob
        }
        
        return results
        
    def predict(self, img, verbose=True):
        tic = time.time()
        
        #pre-process image
        feature = mobilenet.preprocess_input(np.expand_dims(img, axis = 0))
        
        #predict and construct results dict
        pred = self.model.predict(feature)
        results = self.result_infer(pred)
        
        if verbose:
            print("Gender : %s" %results['gender'], \
                  "Age : %s" %results['age'], "Confidence: %.2f"%results['confidence'], 
                  "Prediction time: %.3fs"%(time.time() - tic),
                  sep='\n')
        
        return results