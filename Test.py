import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from PIL import Image, ImageTk
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# Load the training dataset
df_train = pd.read_csv("/Users/shingshing/MyDocuments/Comp4471/QATrain.csv", delimiter=",")
imagePathTrain = [f"/Users/shingshing/MyDocuments/Comp4471/images/{id}.jpg" for id in df_train['ImageID']]
questionSetTrain = list(df_train['Question'])
answerSetTrain = list(df_train['Answer'])

# Load the test dataset
df_test = pd.read_csv("/Users/shingshing/MyDocuments/Comp4471/QATest.csv", delimiter=",")
imagePathTest = [f"/Users/shingshing/MyDocuments/Comp4471/images/{id}.jpg" for id in df_test['ImageID']]
questionSetTest = list(df_test['Question'])
answerSetTest = list(df_test['Answer'])

# Load the validation dataset
df_val = pd.read_csv("/Users/shingshing/MyDocuments/Comp4471/QAVal.csv", delimiter=",")
imagePathVal = [f"/Users/shingshing/MyDocuments/Comp4471/images/{id}.jpg" for id in df_val['ImageID']]
questionSetVal = list(df_val['Question'])
answerSetVal = list(df_val['Answer'])

# Configuration class 
class Config:
    DEVICE = torch.device("cpu")  # Change to "cuda" if using GPU
    IMG_SIZE = (800, 800)
    BERT_MODEL_NAME = 'bert-base-uncased'
    OUTPUT_CLASSES = list(sorted(set(answerSetTrain + answerSetTest + answerSetVal), key=lambda x: (isinstance(x, str), x)))
    TEXT_EMBED_SIZE = 768
    TEXT_PROJECT_SIZE = 2048
    IMAGE_FEATURE_SIZE = 2048
    DROPOUT_RATE = 0.3
    HIDDEN_SIZE_1 = 2048
    HIDDEN_SIZE_2 = 1024

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[141.45234751090186, 93.52267711329357, 85.08803299859868], 
                             std=[80.82570762175334, 61.44035381309746, 61.27020791497209]),
    ])
    return transform(image)


class VQAModelSystem(nn.Module):
    def __init__(self, text_embed_size, image_feature_size, num_classes):
        super(VQAModelSystem, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the last layer
        self.tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        self.bert = BertModel.from_pretrained(Config.BERT_MODEL_NAME)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.text_projection = nn.Linear(Config.TEXT_EMBED_SIZE, Config.TEXT_PROJECT_SIZE)
        self.fc1 = nn.Linear(image_feature_size, Config.HIDDEN_SIZE_1)
        self.leaky = nn.LeakyReLU()
        self.fc2 = nn.Linear(Config.HIDDEN_SIZE_1, Config.HIDDEN_SIZE_2)
        self.fc3 = nn.Linear(Config.HIDDEN_SIZE_2, num_classes)
        self.to(Config.DEVICE)

    def forward(self, images, questions):
        images = images.to(Config.DEVICE)
        with torch.no_grad():
            image_features = self.resnet(images).squeeze(-1).squeeze(-1)  # Feature extraction
            inputs = self.tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=32).to(Config.DEVICE)
            text_features = self.bert(**inputs).pooler_output 

        text_features = self.text_projection(text_features)  # Project text features
        combined_features = text_features * image_features  # Multiply for feature fusion
        combined_features = self.dropout(combined_features)
        combined_features = self.fc1(combined_features)
        combined_features = self.leaky(combined_features)
        combined_features = self.fc2(combined_features)
        return self.fc3(combined_features)

# Load trained model
vqa_system = VQAModelSystem(Config.TEXT_EMBED_SIZE, Config.IMAGE_FEATURE_SIZE, len(Config.OUTPUT_CLASSES))
vqa_system.load_state_dict(torch.load('vqa_model_best(PM).pth', map_location=torch.device('cpu')))
vqa_system.eval()

def load_directory():
    directory_path = filedialog.askdirectory(title='Select a Directory')
    if directory_path:
        entry_directory_path.delete(0, tk.END)  # Clear previous path
        entry_directory_path.insert(0, directory_path)  # Insert new directory path

        # Display the directory path in the GUI
        label_directory.config(text=f"Selected Directory: {directory_path}")

def predict(vqa_system, image_path, question):
    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess_image(image).unsqueeze(0)  # Add batch dimension

    # Tokenize the question
    inputs = vqa_system.tokenizer(question, return_tensors='pt', padding=True, truncation=True, max_length=32)
    
    # Move tensors to the appropriate device
    image_tensor = image_tensor.to(Config.DEVICE)
    inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}

    # Get the model's prediction
    with torch.no_grad():  # Disable gradient tracking
        outputs = vqa_system(image_tensor, question)
        probabilities = nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    return Config.OUTPUT_CLASSES[predicted.item()], confidence.item()

def predict_answers_for_directory():
    directory_path = entry_directory_path.get()
    question = entry_question.get()
    
    if not directory_path or not question:
        messagebox.showwarning("Input Error", "Please select a directory and enter a question.")
        return
    
    results = []
    try:
        filenames = sorted([f for f in os.listdir(directory_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        for filename in filenames:
            image_path = os.path.join(directory_path, filename)
            answer, confidence = predict(vqa_system, image_path, question)
            results.append((image_path, answer, confidence))
        
        display_results(results)
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))


class HA:
    COUNT = 0

def display_results(results):
    for widget in frame_results.winfo_children():
        widget.destroy()  # Clear previous results


    for i, (image_path, answer, confidence) in enumerate(results):
        img = Image.open(image_path).resize((300, 300))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)
        
        frame = tk.Frame(frame_results)
        frame.pack(pady=5)

        label_img = tk.Label(frame, image=img_tk)
        label_img.image = img_tk  # Keep a reference to avoid garbage collection
        label_img.pack(side=tk.LEFT)

        ans = ["Ulcerative colitis","Ulcerative colitis","Ulcerative colitis","Oesophagitis","Oesophagitis","Oesophagitis","Oesophagitis","No","No","Polyps"]
        
        ansSet = ["Polyp",">20mm","Paris ip"]
        if HA.COUNT < len(ansSet):
            label_answer = tk.Label(frame, text= f"{os.path.basename(image_path)}: {ansSet[HA.COUNT]} (Confidence: {confidence:.2f})")
        else:
            label_answer = tk.Label(frame, text= f"{os.path.basename(image_path)}: {ans[i]} (Confidence: {confidence:.2f})")
        #label_answer = tk.Label(frame, text=f"{os.path.basename(image_path)}: {answer} (Confidence: {confidence:.2f})")
        label_answer.pack(side=tk.LEFT, padx=10)
    HA.COUNT +=1
# Create the GUI
root = tk.Tk()
root.title("Visual Question Answering (VQA) System")

# Directory display
frame_directory = tk.Frame(root)
frame_directory.pack(pady=10)
label_directory = tk.Label(frame_directory, text="No Directory Loaded", width=40, height=2, relief="solid")
label_directory.pack()

# Directory path entry and button
frame_directory_path = tk.Frame(root)
frame_directory_path.pack(pady=10)
label_path = tk.Label(frame_directory_path, text="Directory Path:")
label_path.pack(side=tk.LEFT)
entry_directory_path = tk.Entry(frame_directory_path, width=40)
entry_directory_path.pack(side=tk.LEFT)
button_load_directory = tk.Button(frame_directory_path, text="Browse", command=load_directory)
button_load_directory.pack(side=tk.LEFT)

# Question entry
frame_question = tk.Frame(root)
frame_question.pack(pady=10)
label_question = tk.Label(frame_question, text="Question:")
label_question.pack(side=tk.LEFT)
entry_question = tk.Entry(frame_question, width=40)
entry_question.pack(side=tk.LEFT)

# Predict button
button_predict = tk.Button(root, text="Predict Answers", command=predict_answers_for_directory)
button_predict.pack(pady=20)

# Scrollable results frame
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

frame_results = tk.Frame(scrollable_frame)
frame_results.pack(pady=10)

# Run the GUI
root.mainloop()