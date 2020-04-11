from fastai2.vision.all import *

class BearClassifier:
    
    def __init__(self, model_path):
        self.learn_inf = load_learner(model_path)
        
    def predict(self, image_file):
        img = PILImage.create(image_file)
        pred,pred_idx,probs = self.learn_inf.predict(img)
        return pred,probs[pred_idx].item()
