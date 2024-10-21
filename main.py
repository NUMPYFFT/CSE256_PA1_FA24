# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, WordEmbeddings, read_word_embeddings, RandomWordEmbeddings
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import DAN, SentimentDatasetDAN
from BPE import BPE, read_sentiment_examples_BPE, SentimentDatasetBPE, DAN_BPE
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

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
        
        # Load GloVe embeddings
        # 1a with pre-trained embeddings
        print("Loading GloVe embeddings...")
        glove_file = "data/glove.6B.50d-relativized.txt" # 50 or 300

        # Create the WordEmbeddings object by loading the embeddings from the file
        word_embeddings = read_word_embeddings(glove_file)
        print(f"Vocabulary size: {len(word_embeddings.word_indexer)}")

        # Create the dataset and dataloader using the word indexer from word_embeddings
        train_dataset = SentimentDatasetDAN("data/train.txt", word_embeddings)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_dataset = SentimentDatasetDAN("data/dev.txt", word_embeddings)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Train and evaluate the DAN model using GloVe embeddings
        dan_train_accuracy, dan_test_accuracy = experiment(DAN(word_embeddings, hidden_dim=512, output_dim=2, frozen=False), train_loader, test_loader)
        
        # 1b with random embeddings
        # Train and evaluate the DAN model using random embeddings
        print("Training DAN model with random embeddings...")
        word_indexer = Indexer()
        word_indexer.add_and_get_index("PAD")  # Index 0 for PAD token
        word_indexer.add_and_get_index("UNK")  # Index 1 for UNK token

        # Function to add words from the dataset to the indexer
        def add_words_to_indexer(sentences, word_indexer):
            for sentence in sentences:
                words = sentence.split()  # Split sentence into words
                for word in words:
                    word_indexer.add_and_get_index(word)  # Add word to the indexer if not already added

        # Load the dataset (this function should return a list of examples where each example has a sentence)
        train_sentences = [ex.words for ex in read_sentiment_examples("data/train.txt")]
        test_sentences = [ex.words for ex in read_sentiment_examples("data/dev.txt")]

        # Combine train and test sentences to ensure all words are added to the indexer
        all_sentences = train_sentences + test_sentences

        # Add all words from the combined dataset to the word indexer
        add_words_to_indexer([" ".join(sentence) for sentence in all_sentences], word_indexer)

        # Check the size of the vocabulary
        embedding_dim = 50
        random_word_embeddings = RandomWordEmbeddings(word_indexer, embedding_dim)
        
        rand_train_accuracy, rand_test_accuracy = experiment(DAN(random_word_embeddings, hidden_dim=512, output_dim=2, frozen=False), 
                                                             train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='DAN with GloVe embeddings (50D)')
        plt.plot(rand_train_accuracy, label='DAN with random embeddings (50D)')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'dan_train_accuracy_50d.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_accuracy, label='DAN with GloVe embeddings (50D)')
        plt.plot(rand_test_accuracy, label='DAN with random embeddings (50D)')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dan_dev_accuracy_50d.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")
        
    elif args.model == "BPE":
        # part 2
        # Read the training data and split sentences
        train_sentences, train_labels = read_sentiment_examples_BPE("data/train.txt")
        # Lowercase the corpus before training BPE
        train_sentences = [sentence.lower() for sentence in train_sentences]

        # Check if train_sentences is a list
        print(f"Type of train_sentences: {type(train_sentences)}")  # Should output: <class 'list'>

        # Check the first few elements in the list to ensure they are strings (sentences)
        print("First 5 sentences from train_sentences:")
        for i, sentence in enumerate(train_sentences[:5]):
            print(f"Sentence {i+1}: {sentence}")
        
        # Initialize and train the BPE model
        bpe_model = BPE(vocab_size=500)  # Set the desired subword vocabulary size
        print("Training BPE model...")
        bpe_model.train(train_sentences)
        print(f"merging rules: {bpe_model.bpe_codes}")
        
        print(f"BPE Vocabulary Size: {len(bpe_model.vocab)}")
        # Display the final subword vocabulary
        # bpe_model.display_final_vocab()
        
        # Save the final subword vocabulary to a file
        bpe_model.save_final_vocab_to_file("final_subword_vocab.txt")
        
        word = "cat"
        encoded_tokens = bpe_model.encode(word)
        print(f"Encoded tokens for '{word}': {encoded_tokens}")

        # Create dataset and dataloader
        train_dataset = SentimentDatasetBPE("data/train.txt", bpe_model)  # Use the updated dataset class
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        test_dataset = SentimentDatasetBPE("data/dev.txt", bpe_model)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Initialize the DAN_BPE model with the appropriate parameters
        vocab_size = len(bpe_model.vocab)  # Get the size of the learned BPE vocabulary
        embedding_dim = 50  # Example embedding dimension
        hidden_dim = 512  # Example hidden layer size
        output_dim = 2  # Binary classification (positive or negative sentiment)

        # Initialize the DAN_BPE model
        dan_model = DAN_BPE(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        
        # Train and evaluate the DAN model
        print("Training DAN model with BPE...")
        dan_train_accuracy, dan_test_accuracy = experiment(dan_model, train_loader, test_loader)


if __name__ == "__main__":
    main()
