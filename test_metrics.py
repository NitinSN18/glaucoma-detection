import requests
import json

image_path = r'c:\Users\nitin\GitHub\glaucoma-detection\data\train\glaucoma\Im316_g_ACRIMA.jpg'

with open(image_path, 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5001/api/combined', files=files)
    
data = response.json()
if 'clinical_metrics' in data:
    metrics = data['clinical_metrics']
    print('Clinical Metrics Found:')
    print(f'  Glaucoma Percentage: {metrics["glaucoma_percentage"]}%')
    print(f'  Sensitivity: {metrics["sensitivity"]}%')
    print(f'  Specificity: {metrics["specificity"]}%')
    print(f'  Cup-to-Disc Ratio: {metrics["cup_to_disc_ratio"]}')
    print(f'  Severity: {metrics["severity"]}')
    print(f'  Recommendation: {metrics["recommendation"]}')
else:
    print('ERROR: clinical_metrics not found in response')
    print('Available keys:', list(data.keys()))
