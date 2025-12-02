import requests
import threading
import time
import os
import sys
from PIL import Image
import io
import base64

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))
from app import app

def run_server():
    app.run(port=5001)

def test_endpoints():
    print("🧪 Starting Web App Endpoint Tests...")
    
    # Start server in background thread
    thread = threading.Thread(target=run_server)
    thread.daemon = True
    thread.start()
    time.sleep(5) # Wait for server to start
    
    base_url = 'http://127.0.0.1:5001'
    test_img_path = '../outputs/cutouts/original_1.jpg'
    
    if not os.path.exists(test_img_path):
        print(f"Test image not found: {test_img_path}")
        return
    
    # 1. Test /api/cutout
    print("\n1. Testing /api/cutout...")
    try:
        with open(test_img_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f'{base_url}/api/cutout', files=files)
            
        if response.status_code == 200:
            data = response.json()
            
            # Decode base64 image
            img_data = base64.b64decode(data['image'].split(',')[1])
            img = Image.open(io.BytesIO(img_data))
            
            print(f"   Success! Received image: {img.size}, Mode: {img.mode}")
            print(f"   Metrics received:")
            print(f"      - Confidence: {data['metrics']['confidence']:.4f}")
            print(f"      - Benchmark IoU: {data['metrics']['benchmark_iou']}")
            
            if 'alpha_matte' in data:
                print("   Alpha matte received for visualization")
            else:
                print("   Alpha matte missing")
            
            if img.mode == 'RGBA':
                print("   Image is RGBA (Transparent)")
            else:
                print("   Image is not RGBA")
        else:
            print(f"   Failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"   Exception: {e}")

    # 2. Test /api/composite
    print("\n2. Testing /api/composite...")
    try:
        # Use same image for fg and bg for simplicity
        with open(test_img_path, 'rb') as f1, open(test_img_path, 'rb') as f2:
            files = {
                'foreground': ('fg.jpg', f1, 'image/jpeg'),
                'background': ('bg.jpg', f2, 'image/jpeg')
            }
            response = requests.post(f'{base_url}/api/composite', files=files)
            
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            print(f"   Success! Received composite: {img.size}, Mode: {img.mode}")
        else:
            print(f"   Failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"   Exception: {e}")
        
    print("\nTests Completed")

if __name__ == '__main__':
    test_endpoints()
