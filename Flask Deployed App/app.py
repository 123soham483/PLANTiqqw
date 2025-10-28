import os
from flask import Flask, redirect, render_template, request, Response
import base64
import urllib.request
from PIL import Image
import torchvision.transforms.functional as TF  # pyright: ignore[reportMissingImports]
import CNN
import numpy as np  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = None
try:
    model = CNN.CNN(39)
    # Try latest file name; fallback to older filename if present
    model_path_candidates = [
        "plant_disease_model_1_latest.pt",
        "plant_disease_model_1.pt"
    ]
    loaded = False
    for candidate in model_path_candidates:
        if os.path.exists(candidate):
            model.load_state_dict(torch.load(candidate, map_location=torch.device('cpu')))
            loaded = True
            break
    if loaded:
        model.eval()
    else:
        print("[WARN] Model file not found. Place 'plant_disease_model_1_latest.pt' or 'plant_disease_model_1.pt' in the 'Flask Deployed App' folder.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")

def prediction(image_path):
    if model is None:
        raise RuntimeError("Model not loaded. Please add the model file and restart the server.")
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    with torch.no_grad():
        output = model(input_data)
        output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        if model is None:
            return "Model not loaded on server. Please add 'plant_disease_model_1_latest.pt' to the Flask Deployed App folder and reload.", 503
        image = request.files.get('image')
        if image is None or image.filename == '':
            return redirect('/index')
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            uploaded_image=file_path,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link,
        )
    # For GET, take the user back to the AI Engine upload page
    return redirect('/index')

@app.route('/market', methods=['GET'])
def market():
    # Build product list from CSV. Only keep rows with a valid image/link
    csv_products = []
    for name, image, link in zip(supplement_info['supplement name'], supplement_info['supplement image'], supplement_info['buy link']):
        if pd.isna(image) or pd.isna(link) or str(image).strip() == '' or str(link).strip() == '':
            continue
        csv_products.append({
            'name': str(name),
            'type': 'Supplements',
            'image_url': str(image),
            'buy_link': str(link),
            'price': None,  # price not available in CSV; shown as "Check price"
            'disease_description': 'Recommended supplement or fertilizer for plant health.'
        })

    # Curated defaults to guarantee content even if CSV or images are blocked
    curated = [
        {
            'name': 'Organic NPK 10-10-10 Fertilizer',
            'type': 'Healthy',
            'image_url': 'https://via.placeholder.com/600x400?text=Organic+NPK+10-10-10',
            'buy_link': 'https://www.amazon.in/s?k=npk+fertilizer+10-10-10',
            'price': None,
            'disease_description': 'Balanced nutrients for overall plant health.'
        },
        {
            'name': 'Neem Oil Bio Pesticide',
            'type': 'Supplements',
            'image_url': 'https://via.placeholder.com/600x400?text=Neem+Oil+Bio+Pesticide',
            'buy_link': 'https://www.amazon.in/s?k=neem+oil+for+plants',
            'price': None,
            'disease_description': 'Natural control for many leaf diseases and pests.'
        },
        {
            'name': 'Potassium Bicarbonate Fungicide',
            'type': 'Supplements',
            'image_url': 'https://via.placeholder.com/600x400?text=Potassium+Bicarbonate+Fungicide',
            'buy_link': 'https://www.amazon.in/s?k=potassium+bicarbonate+fungicide',
            'price': None,
            'disease_description': 'Effective against powdery mildew and leaf spot.'
        }
    ]

    # Ensure we have a visible list; start with curated then fill/extend from CSV
    products: list[dict] = curated.copy()
    products.extend(csv_products)
    # Cap at 60 items for performance
    if len(products) > 60:
        products = products[:60]

    return render_template('market.html', supplements=products)

@app.route('/img', methods=['GET'])
def image_proxy():
    url = request.args.get('u', '').strip()
    if not url:
        return Response(status=404)
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Referer': ''
        })
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec - proxying trusted urls from our list
            data = resp.read()
            content_type = resp.headers.get('Content-Type', 'image/jpeg')
    except Exception:
        # Transparent 1x1 GIF fallback
        data = base64.b64decode('R0lGODlhAQABAPAAAP///wAAACwAAAAAAQABAAACAkQBADs=')
        content_type = 'image/gif'
    return Response(data, mimetype=content_type, headers={'Cache-Control': 'public, max-age=86400'})

if __name__ == '__main__':
    app.run(debug=True)
