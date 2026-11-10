# EX 01 : Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

## PROBLEM STATEMENT:

Image classification is a fundamental task in computer vision, where the goal is to assign an input image to one of several predefined categories. Traditional machine learning approaches often struggle to capture complex patterns in images. Therefore, this project aims to develop a Convolutional Deep Neural Network (CNN) that can automatically learn hierarchical features from images and perform accurate classification.

The objective is to design and implement a CNN model using multiple layers such as convolutional layers, pooling layers, and fully connected layers to extract and learn important features from the dataset. The model should be trained and evaluated on a labeled image dataset to classify images into their respective classes.

The developed model should achieve high accuracy, effectively generalize to unseen data, and demonstrate the advantages of deep learning techniques in image classification tasks.

## Neural Network Model:

<img width="959" height="433" alt="567560573-ca98d4ce-65d5-42ab-b25a-f7c3f63eab61" src="https://github.com/user-attachments/assets/68d1429c-97ff-4148-ad9a-dbff338bdaef" />



## DESIGN STEPS
### STEP 1: 
Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.

### STEP 2: 
Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.


### STEP 3: 
Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.


### STEP 4: 
Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.


### STEP 5: 
Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.


### STEP 6: 

Evaluate the trained model on test images and verify the classification accuracy for new unseen images.



## PROGRAM

### Name: C J ROHIT

### Register Number:212224243005
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
transform = transforms.Compose([
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)
image, label = train_dataset[0]
print(image.shape)
print(len(train_dataset))
image, label = test_dataset[0]
print(image.shape)
print(len(test_dataset))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2 =nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
    def forward(self,x):
      x=self.pool(torch.relu(self.conv1(x)))
      x=self.pool(torch.relu(self.conv2(x)))
      x=self.pool(torch.relu(self.conv3(x)))
      x=x.view(x.size(0),-1)
      x=torch.relu(self.fc1(x))
      x=torch.relu(self.fc2(x))
      x=self.fc3(x)
      return x
from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name:HARSHANA M V   ')
print('Register Number: 212224240053     ')
summary(model, input_size=(1, 28, 28))
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
      optimizer.zero_grad()
      outputs=model(images)
      loss=criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()



  print('Name:HARSHANA M V        ')
  print('Register Number: 212224240053      ')
  print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
train_model(model, train_loader)
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name: HARSHANA M V       ')
    print('Register Number:212224240053       ')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    print('Name:HARSHANA M V       ')
    print('Register Number:  212224240053     ')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print('Name: HARSHANA M V        ')
    print('Register Number:  212224240053     ')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

test_model(model, test_loader)
import matplotlib.pyplot as plt
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
    class_names = dataset.classes

    # Display the image
    print('Name: HARSHANA M V       ')
    print('Register Number: 212224240053      ')
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')
predict_image(model, image_index=85, dataset=test_dataset)
predict_image(model, image_index=82, dataset=test_dataset)

```

### OUTPUT

## Training Loss per Epoch

<img width="660" height="606" alt="image" src="https://github.com/user-attachments/assets/843c5570-63f2-438d-b8c7-59f6c48f3f5f" />



## Confusion Matrix

<img width="1753" height="841" alt="image" src="https://github.com/user-attachments/assets/b00c5ae9-1ebe-46b8-8b1a-cf0d006b6504" />




## Classification Report

<img width="917" height="325" alt="image" src="https://github.com/user-attachments/assets/0388213c-65c9-4d8d-a221-801feb5239d8" />



### New Sample Data Prediction

<img width="402" height="826" alt="image" src="https://github.com/user-attachments/assets/bdd05b98-85a9-4e83-a630-7f5da42ce2ba" />


## RESULT
The Convolutional Neural Network (CNN) model was successfully trained and achieved good classification performance on the given image dataset.
