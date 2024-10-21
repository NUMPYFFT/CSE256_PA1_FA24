import torch
import torch.nn as nn
import torch.nn.functional as F
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset


class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_embeddings, max_seq_len=512):
        """
        Dataset class that uses the word indexer from the word_embeddings object to convert
        words into indices.
        
        :param infile: Path to the input file containing the dataset.
        :param word_embeddings: WordEmbeddings object that contains the word indexer and embedding vectors.
        :param max_seq_len: Maximum sequence length for padding/truncation.
        """
        self.examples = read_sentiment_examples(infile)  # List of examples (sentences and labels)
        self.word_indexer = word_embeddings.word_indexer  # Access the indexer from the WordEmbeddings object
        self.max_seq_len = max_seq_len
        self.sentence_indices, self.labels = self._prepare_data()

    def _prepare_data(self):
        sentences = [" ".join(ex.words) for ex in self.examples]  # Extract sentences
        labels = [ex.label for ex in self.examples]  # Extract labels
        
        sentence_indices = []
        for sentence in sentences:
            # Convert each word to its index using the word_indexer from WordEmbeddings
            word_indices = [self.word_indexer.index_of(word) for word in sentence.split()]
            
            # Pad or truncate to max_seq_len
            if len(word_indices) < self.max_seq_len:
                word_indices += [self.word_indexer.index_of("PAD")] * (self.max_seq_len - len(word_indices))
            else:
                word_indices = word_indices[:self.max_seq_len]
            
            sentence_indices.append(word_indices)
        
        # Convert sentences and labels to tensors
        sentence_indices = torch.tensor(sentence_indices, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return sentence_indices, labels

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.sentence_indices[idx], self.labels[idx]

    
class DAN(nn.Module):
    def __init__(self, word_embeddings, hidden_dim, output_dim, frozen=True):
        super(DAN, self).__init__()
        
        # Initialize the embedding layer using the pre-trained word embeddings from WordEmbeddings class
        self.embedding = word_embeddings.get_initialized_embedding_layer(frozen=frozen)

        # Fully connected layers
        self.fc1 = nn.Linear(word_embeddings.get_embedding_length(), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence_indices):
        """
        :param sentence_indices: Tensor of shape (batch_size, sentence_len) containing word indices for each sentence
        """
        # Ensure that sentence_indices are of type LongTensor (required by nn.Embedding)
        sentence_indices = sentence_indices.long()  # Convert to LongTensor if not already
        
        # Debugging: Check if any index is out of bounds
        max_index = torch.max(sentence_indices)
        min_index = torch.min(sentence_indices)
        vocab_size = self.embedding.num_embeddings
        
        # print(f"Max index in input: {max_index}")
        # print(f"Min index in input: {min_index}")
        # print(f"Vocabulary size: {vocab_size}")
        
        # Check if any indices are out of range
        if max_index >= vocab_size:
            print(f"Error: Index {max_index} is out of range! Max allowed index is {vocab_size - 1}.")
        if min_index < 0:
            print(f"Error: Index {min_index} is out of range! Indices must be >= 0.")

        # Get word embeddings for the sentence
        embedded = self.embedding(sentence_indices)  # Shape: (batch_size, sentence_len, embedding_dim)
        # print(f"Shape of embedded: {embedded.shape}")

        # Average the embeddings across the sentence
        avg_embedding = torch.mean(embedded, dim=1)  # Shape: (batch_size, embedding_dim)
        # print(f"Shape of avg_embedding: {avg_embedding.shape}")

        # Pass through the fully connected layers
        x = F.relu(self.fc1(avg_embedding))
        x = self.fc2(x)
        output = self.log_softmax(x)
        # print(f"Shape of output: {output.shape}")

        return output
