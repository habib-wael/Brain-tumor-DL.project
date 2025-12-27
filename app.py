import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# --- 1. تعريف هياكل النماذج (يجب أن تتطابق مع كود التدريب) ---

class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

def get_vgg16(num_classes):
    model = models.vgg16(weights=None) # لا نحتاج التحميل المسبق هنا لأننا سنحمل الأوزان الخاصة بنا
    model.classifier[6] = nn.Linear(4096, num_classes) # تعديل الطبقة الأخيرة فقط للهيكل الأساسي
    # ملاحظة: إذا كنت غيرت هيكل الـ Classifier بالكامل كما في كودك، يجب استنساخه هنا بدقة
    model.classifier = nn.Sequential(
        nn.Linear(25088, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

def get_resnet50(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    return model

# --- 2. إعدادات الصفحة والمعالجة ---

st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['glioma', 'meningioma', 'notumor', 'pituitary'] # تأكد من مطابقة ترتيب الـ ImageFolder

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# دالة تحميل الموديل المختار
@st.cache_resource # لضمان عدم تحميل الموديل في كل مرة يتم فيها ضغط زر
def load_selected_model(model_name):
    if model_name == "ResNet50":
        m = get_resnet50(4)
        path = "resnet50_best.pth"
    elif model_name == "VGG16":
        m = get_vgg16(4)
        path = "vgg16_best.pth"
    else:
        m = EnhancedCNN(4)
        path = "cnn_best.pth"
    
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        m.eval()
        return m
    else:
        st.error(f"Model file {path} not found!")
        return None

# --- 3. تصميم الـ GUI ---

st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image and let the AI identify the tumor type.")

# شريط جانبي لاختيار النموذج
with st.sidebar:
    st.header("Settings")
    selected_model_type = st.selectbox("Choose Model Architecture", ["ResNet50", "VGG16", "CNN"])
    st.info("Make sure the .pth files are in the same folder as this app.")

# رفع الصورة
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # عرض الصورة المرفوعة
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
    
    if st.button('Predict'):
        model = load_selected_model(selected_model_type)
        
        if model:
            with st.spinner('Analyzing image...'):
                # المعالجة المسبقة والتنبؤ
                img_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    confidence, pred = torch.max(probabilities, 0)
                
                # عرض النتيجة
                result_class = classes[pred.item()]
                
                st.success(f"Prediction: **{result_class}**")
                st.progress(confidence.item())
                st.write(f"Confidence Level: {confidence.item()*100:.2f}%")