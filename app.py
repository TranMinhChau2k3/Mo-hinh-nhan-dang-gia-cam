from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Tải mô hình đã huấn luyện

from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Tạo lại kiến trúc model (như lúc ban đầu)
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 6)


# Assuming you know the number of classes your model was trained for (6 in this case)
num_classes = 6
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("efficientnet-b0.pth", map_location=device))

# 3. Đưa model lên thiết bị (GPU nếu có)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# 4. Chuyển model về chế độ eval để inference
model.eval()

classes = ['Bird', 'Chicken', 'Duck', 'Goose', 'Peacock', 'Pigeon']

def transform_image(image_bytes):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
    image = Image.open(image_bytes).convert('RGB')
    return transform(image).unsqueeze(0)

@app.route('/')
def index():
    return render_template('TrangChu.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    with open(filepath, 'rb') as f:
        input_tensor = transform_image(f).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            label = classes[predicted.item()]
    # Trả về kết quả cùng đường dẫn ảnh
    return jsonify({
        'prediction': label,
        'image_url': f'/uploads/{filename}'
    })

# Để serve ảnh trong folder uploads
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)