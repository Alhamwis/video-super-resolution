#!/usr/bin/env python3
"""
Download PyTorch Model Weights for Super-Resolution
--------------------------------------------------
This script downloads the pre-trained model weights for ESRGAN, Real-ESRGAN, and SRCNN.
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
import shutil
from tqdm import tqdm

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_models():
    """Download the required model weights."""
    models = {
        # ESRGAN model
        'ESRGAN_x4.pth': {
            'url': None,
            'gdrive_id': '1MJFgqXNxZ8c-5ypnbQJhgK_U-m0VO0qr',  # Hosted on Google Drive
            'rename': False
        },
        # Real-ESRGAN model
        'RealESRGAN_x4plus.pth': {
            'url': None,
            'gdrive_id': '1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene',
            'rename': False
        },
        # SRCNN model
        'SRCNN_x4.pth': {
            'url': None,
            'gdrive_id': '1gZ8QWD4UTBPiuIJ7YLxDJNEpsGOdGxgw',  # Hosted on Google Drive
            'rename': False
        }
    }
    
    # Check if required packages are installed
    try:
        import gdown
    except ImportError:
        print("Installing required package: gdown")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown
    
    for model_name, info in models.items():
        output_path = os.path.join('models', model_name)
        
        # Skip if model already exists
        if os.path.exists(output_path):
            print(f"Model {model_name} already exists, skipping...")
            continue
        
        print(f"Downloading {model_name}...")
        
        if info['url']:
            # Download from URL
            try:
                download_url(info['url'], output_path if not info['rename'] else output_path + '.tmp')
                if info['rename']:
                    os.rename(output_path + '.tmp', output_path)
                print(f"Successfully downloaded {model_name}")
            except Exception as e:
                print(f"Failed to download {model_name} from URL: {e}")
                continue
        elif info['gdrive_id']:
            # Download from Google Drive
            try:
                gdown.download(id=info['gdrive_id'], output=output_path, quiet=False)
                print(f"Successfully downloaded {model_name}")
            except Exception as e:
                print(f"Failed to download {model_name} from Google Drive: {e}")
                continue
    
    print("\nDownload completed.")
    print("\nAvailable models:")
    for model in os.listdir('models'):
        if model.endswith('.pth'):
            print(f"- {model}")

if __name__ == "__main__":
    print("Downloading PyTorch model weights for super-resolution...")
    download_models() 