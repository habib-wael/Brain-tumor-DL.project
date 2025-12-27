import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

transform_train = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
data_dir = "/kaggle/input/brain-tumor-mri-dataset"

full_train = datasets.ImageFolder(os.path.join(data_dir,"Training"),
                                  transform=transform_train)

test_dataset = datasets.ImageFolder(os.path.join(data_dir,"Testing"),
                                    transform=transform_test)

train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size

train_dataset, val_dataset = random_split(full_train,[train_size,val_size])

batch_size = 32
train_loader = DataLoader(train_dataset,batch_size,shuffle=True)
val_loader   = DataLoader(val_dataset,batch_size,shuffle=False)
test_loader  = DataLoader(test_dataset,batch_size,shuffle=False)

classes = full_train.classes
print(classes)
class EnhancedCNN(nn.Module):
    def __init__(self,num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*32*32,256)
        self.fc2 = nn.Linear(256,num_classes)

    def forward(self,x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)
import torch.nn as nn
from torchvision import models

def get_vgg16(num_classes):
    model = models.vgg16(pretrained=True)

    # 1️⃣ نجمّد أغلب الـ convolution layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 2️⃣ نفك آخر block بس (fine-tuning خفيف)
    for param in model.features[24:].parameters():
        param.requires_grad = True

   
    model.classifier = nn.Sequential(
        nn.Linear(25088, 1024),   
        nn.BatchNorm1d(1024),    
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(1024, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(256, num_classes)
    )

    return model

import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes):
    model = models.resnet50(pretrained=True)

    
    for param in model.parameters():
        param.requires_grad = False

    
    for param in model.layer4.parameters():
        param.requires_grad = True

    
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(128, num_classes)
    )

    return model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, name, epochs=10):
    model.to(device)

    criterion = nn.CrossEntropyLoss()

   
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001  
    )

    best_acc = 0.0

    for epoch in range(epochs):

        # ================= TRAIN =================
        model.train()
        running_loss = 0
        correct_train = 0
        total_train = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_acc = correct_train / total_train
        train_loss = running_loss / len(train_loader)

        # ================= VALIDATION =================
        model.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val

        print(
            f"{name} | Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        # ================= SAVE BEST =================
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                f"/kaggle/working/{name}_best.pth"
            )

    print(f"✅ Best Validation Accuracy for {name}: {best_acc:.4f}")
    return best_acc
results = {}

# ================= CNN =================
cnn_model = EnhancedCNN(4)
results["CNN"] = train_model(cnn_model, "cnn")

# ================= VGG16 =================
vgg_model = get_vgg16(4)
results["VGG16"] = train_model(vgg_model, "vgg16")

# ================= ResNet50 =================
resnet_model = get_resnet50(4)
results["ResNet50"] = train_model(resnet_model, "resnet50")

# ================= BEST MODEL =================
best_model_name = max(results, key=results.get)
print("Best Model:", best_model_name)
print("Validation Accuracies:", results)

import random

model = get_resnet50(4) if best_model_name=="ResNet50" else \
        get_vgg16(4) if best_model_name=="VGG16" else EnhancedCNN(4)

model.load_state_dict(torch.load(f"{best_model_name.lower()}_best.pth"))
model.to(device)
model.eval()

indices = random.sample(range(len(test_dataset)),5)

plt.figure(figsize=(15,5))
for i,idx in enumerate(indices):
    img,label = test_dataset[idx]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax(1).item()

    plt.subplot(1,5,i+1)
    plt.imshow(img.permute(1,2,0))
    plt.title(f"P:{classes[pred]}\nT:{classes[label]}")
    plt.axis("off")
plt.show()
from PIL import Image
import matplotlib.pyplot as plt

path = "/kaggle/input/brain-tumor-mri-dataset/Testing/notumor/Te-noTr_0001.jpg"

def predict_image(path):
    img = Image.open(path).convert("RGB")
    img_tensor = transform_test(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(img_tensor).argmax(1).item()

    predicted_class = classes[pred]

   
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis("off")
    plt.show()

   
    print("Predicted class:", predicted_class)

    return predicted_class
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np

if best_model_name == "ResNet50":
    model = get_resnet50(4)
    model.load_state_dict(torch.load("/kaggle/working/resnet50_best.pth"))
elif best_model_name == "VGG16":
    model = get_vgg16(4)
    model.load_state_dict(torch.load("/kaggle/working/vgg16_best.pth"))
else:
    model = EnhancedCNN(4)
    model.load_state_dict(torch.load("/kaggle/working/cnn_best.pth"))

model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("Classification Report:\n")
print(classification_report(
    all_labels,
    all_preds,
    target_names=classes
))

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.savefig("/kaggle/working/resnet50_cm.png")
plt.show()