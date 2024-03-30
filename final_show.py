# Import all the libraries
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as TF
import pandas as pd
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import joblib

class ImageClassifier():
    def __init__(self):
        self.input_dict = {}
        self.label_dict = {0: 'Wet Asphalt', 1: 'Wet Concrete', 2: 'Wet Gravel'}
        # Load the three models
        
        model_name_or_path = r'D:\College\Semester_2\CT5135\checkpoint\checkpoint-17000'
        self.feature_extractor_finetuned = ViTFeatureExtractor.from_pretrained(model_name_or_path)
        self.model_1 = ViTForImageClassification.from_pretrained(model_name_or_path)
        
        self.model_2 = models.efficientnet_v2_s(pretrained=True)
        self.model_2.load_state_dict(torch.load(r'D:\College\Semester_2\CT5135\Model_2_best_model.pth')) 
        self.model_3 = models.resnet50(pretrained=True)
        self.model_3.load_state_dict(torch.load(r'D:\College\Semester_2\CT5135\Model_3_best_model.pth')) 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_1.to(self.device)
        self.model_2.to(self.device)
        self.model_3.to(self.device)
        
        self.model_2.eval()
        self.model_3.eval()
        

        self.transform = TF.Compose([
            TF.Resize((224, 224)),
            TF.ToTensor(),
            TF.Normalize(mean=[0.4672, 0.4943, 0.4919],std=[0.1296, 0.1288, 0.1283])
        ])

        # Load the random forest model
        self.model = joblib.load(r'D:\College\Semester_2\CT5135\best_rf_model.joblib')
        
        
    def inference(self, image_obj):
        self.image = image_obj.copy()
        inputs = self.feature_extractor_finetuned(self.image, return_tensors="pt").to(self.device)
        self.image = self.transform(self.image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # model 1 inference
            logits = self.model_1(**inputs).logits
            p_model_1 = nn.functional.softmax(logits, dim=1)
            self.input_dict['class_1_True'] = p_model_1[:, 1].item()
            self.input_dict['class_1_False'] = p_model_1[:, 0].item()
            self.input_dict['class_1_binary'] =logits.argmax(-1).item()
            
            # model 2 inference
            logit = self.model_2(self.image)
            p_model_2 = nn.functional.softmax(logit, dim=1)
            self.input_dict['class_2_True'] = p_model_2[:, 1].item()
            self.input_dict['class_2_False'] = p_model_2[:, 0].item()
            self.input_dict['class_2_binary'] = nn.functional.softmax(logit, dim=1).argmax(1).item()
            
            # model 3 inference
            logit = self.model_3(self.image)
            p_model_3 = nn.functional.softmax(logit, dim=1)
            self.input_dict['class_3_True'] = p_model_3[:, 0].item()
            self.input_dict['class_3_False'] = p_model_3[:, 1].item()
            self.input_dict['class_3_binary'] = 1 if nn.functional.softmax(logit, dim=1).argmax(1).item() == 0.0 else 0
            
            # Create the dataframe
            df = pd.DataFrame(self.input_dict, index=[0])
            df_reordered = df.iloc[:, [0, 1, 3, 4, 6, 7, 2, 5, 8]]
            # Random forest model inference
            result = self.model.predict(df_reordered)
            # print(result[0])
            return self.label_dict[result[0]]

@st.cache_resource()
def load_model():
    ImageClassifierObj = ImageClassifier()
    return ImageClassifierObj

# @st.cache_resource()
def process_image(image):
    ImageClassifierObj = load_model()
    result = ImageClassifierObj.inference(image)
    return result

def main():
    st.title("Image Classifier")
    
    uploaded_file = st.file_uploader("Upload an Image to classify", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=False)

        if st.button("Process Image"):
            result = process_image(image)
            st.write(f"Image is classified as: {result}")

if __name__ == "__main__":
    main()