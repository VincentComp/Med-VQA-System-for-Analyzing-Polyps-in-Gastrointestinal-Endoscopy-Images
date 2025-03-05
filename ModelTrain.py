#3 places to change the save and load directory
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import random
import numpy as np

import numpy as np
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import torch      
import warnings
warnings.filterwarnings("ignore")

# Load the training dataset
df_train = pd.read_csv("/Users/shingshing/MyDocuments/Comp4471/QATrain.csv", delimiter=",")
df_train = df_train
imagePathTrain = [f"/Users/shingshing/MyDocuments/Comp4471/images/{id}.jpg" for id in df_train['ImageID']]
questionSetTrain = list(df_train['Question'])
answerSetTrain = list(df_train['Answer'])

# Load the test dataset
df_test = pd.read_csv("/Users/shingshing/MyDocuments/Comp4471/QATest.csv", delimiter=",")
df_test = df_test
imagePathTest = [f"/Users/shingshing/MyDocuments/Comp4471/images/{id}.jpg" for id in df_test['ImageID']]
questionSetTest = list(df_test['Question'])
answerSetTest = list(df_test['Answer'])

# Load the test dataset
df_val = pd.read_csv("/Users/shingshing/MyDocuments/Comp4471/QAVal.csv", delimiter=",")
df_val = df_val
imagePathVal = [f"/Users/shingshing/MyDocuments/Comp4471/images/{id}.jpg" for id in df_val['ImageID']]
questionSetVal = list(df_val['Question'])
answerSetVal = list(df_val['Answer'])





# Configuration
class Config:
    DEVICE = torch.device("cpu")
    IMG_SIZE = (800, 800)
    BERT_MODEL_NAME = 'bert-base-uncased'
    OUTPUT_CLASSES = list(sorted(set(answerSetTrain + answerSetTest + answerSetVal), key=lambda x: (isinstance(x, str), x))) ###Very Important to make it sorted --> (seem length = 400)
    EPOCHS = 100
    BATCH_SIZE = 25 ###original 128
    LEARNING_RATE = 0.001
    VAL_FREQ = 5
    START_EPOCH = 39 # (This is the new start epoch) --> Note that this may overwrite the best accuracy
    SAVE_BEST = True #This should be save best "also"
    EVAL_PRINT = False
    TRAIN_MODEL = False
    WEIGHT_DECAY = 1e-4
    TEXT_EMBED_SIZE = 768 #Must be 768 for the bert
    TEXT_PROJECT_SIZE = 2048
    IMAGE_FEATURE_SIZE = 2048
    DROPOUT_RATE = 0.3
    HIDDEN_SIZE_1 = 2048
    HIDDEN_SIZE_2 = 1024
    



# Define Image Preprocessing
def preprocess_image():
    return transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[141.45234751090186, 93.52267711329357, 85.08803299859868], std=[80.82570762175334, 61.44035381309746, 61.27020791497209]),
    ])


# Map answers to indices
answer_to_index = {answer: idx for idx, answer in enumerate(Config.OUTPUT_CLASSES)}
labelsTrain = [answer_to_index.get(answer, len(Config.OUTPUT_CLASSES) - 1) for answer in answerSetTrain]
labelsTest = [answer_to_index.get(answer, len(Config.OUTPUT_CLASSES) - 1) for answer in answerSetTest]
labelsVal = [answer_to_index.get(answer, len(Config.OUTPUT_CLASSES) - 1) for answer in answerSetVal]

# Custom Dataset
class VQADataset(Dataset):
    def __init__(self, image_paths, questions, labels, transform=None):
        self.image_paths = image_paths
        self.questions = questions
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        question = self.questions[idx]
        label = self.labels[idx]
        return image, question, label

# VQA model class
class VQAModelSystem(nn.Module):
    def __init__(self, text_embed_size, image_feature_size, num_classes):
        super(VQAModelSystem, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        self.bert = BertModel.from_pretrained(Config.BERT_MODEL_NAME)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.text_projection = nn.Linear(Config.TEXT_EMBED_SIZE, Config.TEXT_PROJECT_SIZE)
        self.fc1 = nn.Linear(image_feature_size, Config.HIDDEN_SIZE_1)
        self.leaky = nn.LeakyReLU()
        self.fc2 = nn.Linear(Config.HIDDEN_SIZE_1, Config.HIDDEN_SIZE_2)
        #leaky relu same as above
        self.fc3 = nn.Linear(Config.HIDDEN_SIZE_2, num_classes)
        self.to(Config.DEVICE)

    def forward(self, images, questions):
        images = images.to(Config.DEVICE)
        with torch.no_grad():
            image_features = self.resnet(images).squeeze(-1).squeeze(-1) #(batch,ImgFeatureSize)
            inputs = self.tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=32).to(Config.DEVICE)
            text_features = self.bert(**inputs).pooler_output #(batch,768)

        
        text_features = self.text_projection(text_features) #(batch,projectSize)
        combined_features = text_features * image_features #(batch,ImgFeatureSize )
        
        combined_features = self.dropout(combined_features)
        combined_features = self.fc1(combined_features)
        combined_features = self.leaky(combined_features)
        combined_features = self.fc2(combined_features)
        combined_features = self.leaky(combined_features)
        return self.fc3(combined_features)

    def train_model(self, train_loader, val_loader = None, validation_interval=1, saveBest = False, saveAll = False):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=Config.LEARNING_RATE, weight_decay= Config.WEIGHT_DECAY)
        self.train()

        best_accuracy = -1  #Set Best Accuracy

        for epoch in range(Config.START_EPOCH - 1,Config.EPOCHS):
            correct_predictions = 0
            total_predictions = 0

            tqdmBar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.EPOCHS}') #Write Epoch
            
            epochAcc = None
            for images, questions, labels in tqdmBar:
                labels = labels.to(Config.DEVICE)
                optimizer.zero_grad()
                outputs = self(images, questions)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, dim=1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
                
                epochAcc = correct_predictions / total_predictions * 100
                tqdmBar.set_postfix(loss=loss.item(), accuracy= epochAcc) #Append loss and accuracy

            if val_loader is not None and ((epoch + 1) % validation_interval == 0): #Validation
                self.eval()
                correct = 0
                total = 0


                with torch.no_grad():
                    for images, questions, labels in tqdm(val_loader):
                        labels = labels.to(Config.DEVICE)
                        outputs = self(images, questions)
                        _, predicted = torch.max(outputs, dim=1)
                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)

                accuracy = (correct / total) * 100
                tqdm.write(f'Epcoh {epoch + 1} (VAL),Validation Accuracy: {accuracy:.2f}%, loss = {loss.item():.4f}')
            
                if saveBest and accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(self.state_dict(), 'vqa_model_best(PM).pth')
                    tqdm.write(f'Saved best model with accuracy: {best_accuracy:.2f}%')

            torch.save(self.state_dict(), f'vqa_model_best_{epoch + 1}(PM).pth')
            tqdm.write(f'Saved model with accuracy: {epochAcc:.2f}%')
                


    def evaluate_model(self, test_loader, eval_print=False):
        self.eval()

        correct = 0
        total = 0

        # Lists to store predictions and actual answers
        predictions_list = []
        actual_list = []

        with torch.no_grad():
            for images, questions, labels in tqdm(test_loader):
                labels = labels.to(Config.DEVICE)
                outputs = self(images, questions)
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Store predictions and actual labels after removing commas
                predictions_list.extend([Config.OUTPUT_CLASSES[p].replace(',', '').strip() for p in predicted])
                actual_list.extend([Config.OUTPUT_CLASSES[l].replace(',', '').strip() for l in labels])

                # Debug output if requested
                if eval_print:
                    for QIndex, (thisQ, thisL, thisP) in enumerate(zip(questions, labels, predicted)):
                        Result = "Correct" if (Config.OUTPUT_CLASSES[thisL] == Config.OUTPUT_CLASSES[thisP]) else "Wrong"
                        print(f'[Q{QIndex + 1} {Result}] Question: {thisQ}, Actual Answer: {Config.OUTPUT_CLASSES[thisL]}, Predicted Answer: {Config.OUTPUT_CLASSES[thisP]}')

        # Calculate accuracy
        accuracy = (correct / total) * 100
        print(f'Test Accuracy: {accuracy:.2f}%')

        # Preparation for BLEU-1 calculation
        # For BLEU, we need a list of lists for actual references
        reference_list = [[actual.split()] for actual in actual_list]  # Use split to tokenize
        predictions_list = [predicted.split() for predicted in predictions_list]  # Tokenize predictions

        # Calculate BLEU-1 score
        bleu1_score = corpus_bleu(reference_list, predictions_list, weights=(1, 0, 0, 0))  # BLEU-1

        # Print BLEU-1 score
        print(f'BLEU-1 Score: {bleu1_score:.4f}')



# Main
if __name__ == "__main__":
    # Data loading
    transform = preprocess_image()
    train_dataset = VQADataset(imagePathTrain, df_train['Question'].tolist(), labelsTrain, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_dataset = VQADataset(imagePathTest, df_test['Question'].tolist(), labelsTest, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    val_dataset = VQADataset(imagePathVal, df_val['Question'].tolist(), labelsVal, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)


    # VQA model with complete save and load capability
    vqa_system = VQAModelSystem(Config.TEXT_EMBED_SIZE, Config.IMAGE_FEATURE_SIZE, len(Config.OUTPUT_CLASSES))


    # Count the number of parameters
    num_params = sum(p.numel() for p in vqa_system.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {num_params}')
    

    # Train and save
    if Config.TRAIN_MODEL:
        if Config.START_EPOCH != 1:
            vqa_system.load_state_dict(torch.load(f'vqa_model_best_{Config.START_EPOCH-1}(PM).pth'))

        vqa_system.train_model(train_loader,val_loader,validation_interval = Config.VAL_FREQ ,saveBest = Config.SAVE_BEST)

    # Load trained model
    
    vqa_system.load_state_dict(torch.load(f'vqa_model_best(PM).pth',map_location=torch.device('cpu')))
    vqa_system.eval()

    # Evaluate the same model
    vqa_system.evaluate_model(test_loader, eval_print = Config.EVAL_PRINT)
