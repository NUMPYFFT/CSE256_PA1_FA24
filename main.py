# models.py

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, WordEmbeddings, read_word_embeddings
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import DAN, SentimentDatasetDAN
from utils import Indexer


# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()
        # print(batch, X.shape, y.shape) # batch index, 16x512, 16

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradient norms for each parameter
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient norm for {name}: {param.grad.norm().item()}")
                
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        # print(train_accuracy)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}, train loss {train_loss:.3f}, dev loss {test_loss:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load dataset
    start_time = time.time()

    train_data = SentimentDatasetBOW("data/train.txt")
    dev_data = SentimentDatasetBOW("data/dev.txt")
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")


    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        # TODO:  Train and evaluate your DAN
        # print("DAN model not implemented yet")
        print("Training DAN model...")
    
        start_time = time.time()
        
        # Assume you have a GloVe embedding file and you've built an indexer
        word_indexer = Indexer()
        word_indexer.add_and_get_index("PAD")  # Add PAD token
        word_indexer.add_and_get_index("UNK")  # Add UNK token

        # Load GloVe embeddings
        glove_file = "data/glove.6B.300d-relativized.txt"

        # Create the WordEmbeddings object by loading the embeddings from the file
        word_embeddings = read_word_embeddings(glove_file)
        print(f"Vocabulary size: {len(word_embeddings.word_indexer)}")
        
        # Simulate loading a small subset of the GloVe file manually for testing
        glove_vocab = {
            "the": 2,
            ".": 3,
            ",": 4
        }
        # Add these words to the indexer to simulate the GloVe embeddings being loaded
        for word, idx in glove_vocab.items():
            word_indexer.add_and_get_index(word)

        # Example sentence: "the bruehsbayreuyhr . , the"
        sentence = "the bruehsbayreuyhr . , the"
        words = sentence.split()

        # Translate the sentence into indices using the word indexer
        sentence_indices = [word_indexer.index_of(word) for word in words]

        # Expected output: [2, 1, 3, 4, 2]
        print("Sentence:", sentence)
        print("Word indices:", sentence_indices)

        # Verify the embeddings for PAD, UNK, "the", ".", and ","
        print("PAD index:", word_indexer.index_of("PAD"))  # Should print 0
        print("UNK index:", word_indexer.index_of("UNK"))  # Should print 1
        print("Index of 'the':", word_indexer.index_of("the"))  # Should print 2
        print("Index of '.':", word_indexer.index_of("."))  # Should print 3
        print("Index of ',':", word_indexer.index_of(","))  # Should print 4

        # Create the dataset and dataloader for training
        train_dataset = SentimentDatasetDAN("data/train.txt", word_indexer)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_dataset = SentimentDatasetDAN("data/dev.txt", word_indexer)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

        # Now train_loader can be used to train your DAN model
        for batch_idx, (X, y) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"Input (X) shape: {X.shape}")  # Should be (batch_size, max_seq_len)
            print(f"Labels (y) shape: {y.shape}")  # Should be (batch_size,)
            break

        # Initialize the DAN model using the loaded embeddings
        # dan_model = DAN(word_embeddings, hidden_dim=128, output_dim=2, frozen=True)
        # print(dan_model)

        # Train and evaluate the DAN model
        dan_train_accuracy, dan_test_accuracy = experiment(DAN(word_embeddings, hidden_dim=512, output_dim=2, frozen=False), 
                                                           train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'dan_train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dan_dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

if __name__ == "__main__":
    main()
