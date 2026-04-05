import torch
import tkinter as tk
from tkinter import filedialog
from efficientnet_pytorch import EfficientNet
from PIL import Image
import torchvision.transforms as transforms

# Load the model
model = EfficientNet.from_pretrained('efficientnet-b0')
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Function to select an image and perform inference
def select_image():
    root = tk.Tk()  
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title='Select an Image', filetypes=[('Image Files', '*.png;*.jpg;*.jpeg')])
    if file_path:
        image = Image.open(file_path).convert('RGB')
        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image).unsqueeze(0)
        
        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.max(torch.nn.functional.softmax(output, dim=1)).item()
            
            # Display the prediction result
            if prediction == 0:
                result = 'Normal'
            else:
                result = 'Glaucoma'
            print(f'Result: {result}, Confidence: {confidence * 100:.2f}%')

# Run the image selection and inference process
if __name__ == '__main__':
    select_image()