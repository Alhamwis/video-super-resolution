import urllib.request
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# URLs for different models
model_urls = {
    # ESPCN models
    'ESPCN_x2': 'https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb',
    'ESPCN_x3': 'https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x3.pb',
    'ESPCN_x4': 'https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb',
    
    # FSRCNN models
    'FSRCNN_x2': 'https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb',
    'FSRCNN_x3': 'https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x3.pb',
    'FSRCNN_x4': 'https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb',
    
    # LapSRN models
    'LapSRN_x2': 'https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x2.pb',
    'LapSRN_x3': 'https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x3.pb',
    'LapSRN_x4': 'https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x4.pb',
}

# Download the models
for model_name, url in model_urls.items():
    output_path = os.path.join('models', f'{model_name}.pb')
    
    # Skip if model already exists
    if os.path.exists(output_path):
        print(f"Model {model_name} already exists, skipping...")
        continue
    
    print(f"Downloading {model_name} model to {output_path}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Successfully downloaded {model_name}")
    except Exception as e:
        print(f"Failed to download {model_name}: {e}")

print("Download completed.") 