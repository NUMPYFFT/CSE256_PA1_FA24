from collections import defaultdict, Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import defaultdict, Counter


class BPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}  # Final BPE vocabulary (subwords)
        self.bpe_codes = {}  # BPE merge operations

    def get_stats(self, word_freqs):
        """
        Calculate frequencies of adjacent character pairs in the vocabulary.
        :param word_freqs: Dictionary of word frequencies.
        :return: Dictionary of adjacent character pair frequencies.
        """
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair, word_freqs):
        """
        Merge the most frequent pair in the vocabulary and update the vocabulary accordingly.
        :param pair: The pair of characters/subwords to merge.
        :param word_freqs: The current word frequencies.
        :return: The updated vocabulary with merged pairs.
        """
        bigram = ' '.join(pair)  # e.g., 'e </w>'
        replacement = ''.join(pair)  # e.g., 'e</w>'
        new_word_freqs = Counter()

        # Debug: Show what we are replacing
        # print(f"Replacing all instances of '{bigram}' with '{replacement}'")

        # Replace all instances of the bigram in every word
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)  # Merge the pair in the word

            # Debug: Check how each word is being transformed
            # if word != new_word:
                # print(f"Word '{word}' changed to '{new_word}'")

            new_word_freqs[new_word] += freq  # Accumulate frequencies for the new word

        # Debug: Show the size of the new vocabulary after merge
        print(f"New vocabulary size after merge: {len(new_word_freqs)}")
        return new_word_freqs

    def train(self, corpus):
        """
        Train the BPE model on the given corpus.
        """
        # Create initial vocabulary by splitting words into characters
        word_freqs = Counter(" ".join(list(word)) + " </w>" for sentence in corpus for word in sentence.split())

        # Debug: Print initial vocabulary size
        print(f"Initial vocab size: {len(word_freqs)}")
        prev_vocab_size = len(word_freqs)

        total_merges = 0  # To track the number of merges

        # Keep merging pairs until we reach the desired vocab size
        while total_merges < self.vocab_size and len(word_freqs) > self.vocab_size:
            # Step 1: Get the most frequent pairs
            pairs = self.get_stats(word_freqs)
            if not pairs:
                break  # No more pairs to merge

            # Step 2: Find the best pair to merge
            best_pair = max(pairs, key=pairs.get)
            print(f"Merging pair: {best_pair} with frequency {pairs[best_pair]}")

            # Step 3: Merge the pair in the vocabulary
            new_word_freqs = self.merge_vocab(best_pair, word_freqs)
            self.bpe_codes[best_pair] = len(self.bpe_codes)

            # Step 4: Check if vocabulary size reduces
            word_freqs = new_word_freqs  # Always update vocabulary to continue merging
            prev_vocab_size = len(word_freqs)  # Track the new size
            total_merges += 1  # Increment the merge count
            print(f"Vocab size after merge {total_merges}: {len(word_freqs)}")

        # Step 5: Store the final vocabulary and number of merges
        self.vocab = {word: idx for idx, word in enumerate(word_freqs)}
        print(f"Final vocab size: {len(self.vocab)}, total merges: {total_merges}")

        return self.vocab
    
    # def train(self, corpus):
    #     # Create initial vocabulary by splitting words into characters
    #     word_freqs = Counter(" ".join(list(word)) + " </w>" for sentence in corpus for word in sentence.split())
        
    #     print(f"Initial vocab size: {len(word_freqs)}")

    #     prev_vocab_size = len(word_freqs)
    #     total_merges = 0  # To track the number of merges

    #     # Keep merging pairs until we reach the desired vocab size
    #     while total_merges < self.vocab_size and len(word_freqs) > self.vocab_size:
    #         pairs = self.get_stats(word_freqs)
    #         if not pairs:
    #             print("No pairs to merge!")
    #             break  # No more pairs to merge

    #         # Find the best pair to merge
    #         best_pair = max(pairs, key=pairs.get)
    #         print(f"Merging pair: {best_pair} with frequency {pairs[best_pair]}")

    #         # Merge the pair in the vocabulary
    #         word_freqs = self.merge_vocab(best_pair, word_freqs)

    #         # Update the merge codes
    #         self.bpe_codes[best_pair] = len(self.bpe_codes)

    #         total_merges += 1
    #         current_vocab_size = len(word_freqs)
    #         print(f"Vocab size after merge {total_merges}: {current_vocab_size}")

    #         # Stop if no merges are reducing the vocab size
    #         if current_vocab_size == prev_vocab_size:
    #             print("No reduction in vocabulary size, stopping.")
    #             break
    #         prev_vocab_size = current_vocab_size

    #     # Assign final subword tokens to vocab
    #     self.vocab = {word: idx for idx, word in enumerate(word_freqs)}
    #     # print(f"Final vocab size: {len(self.vocab)}, total merges: {total_merges}")
    #     # print(f"BPE codes: {self.bpe_codes}")

    def encode(self, word):
        """
        Encode a word into its BPE subword units.
        :param word: The input word to encode.
        :return: List of subword token IDs (integers).
        """
        word = list(word) + ['</w>']  # End of word symbol
        while len(word) > 1:
            pairs = [(word[i], word[i+1]) for i in range(len(word) - 1)]
            if any(pair in self.bpe_codes for pair in pairs):
                # Find the best pair to merge
                best_pair = min((pair for pair in pairs if pair in self.bpe_codes), key=self.bpe_codes.get)
                # Merge the best pair
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                        new_word.append(word[i] + word[i+1])  # Merge the pair
                        i += 2  # Skip the next symbol since it's merged
                    else:
                        new_word.append(word[i])
                        i += 1
                word = new_word
            else:
                break

        # Return the subword token IDs
        return [self.vocab.get(" ".join(subword), 1) for subword in word]  # Use unknown token ID (1) for OOV subwords
    
    def display_final_vocab(self):
        print("Final Subword Vocabulary:")
        for subword, index in self.vocab.items():
            print(f"Subword: '{subword}' -> Index: {index}")
            
    def save_final_vocab_to_file(self, file_path):
        """
        Save the final subword vocabulary to a text file.
        :param file_path: The path to the file where the vocabulary will be saved.
        """
        with open(file_path, 'w') as f:
            for subword, index in self.vocab.items():
                f.write(f"{subword} -> {index}\n")
        print(f"Final subword vocabulary saved to {file_path}")


def read_sentiment_examples_BPE(filepath):
    """
    Read the sentiment dataset and return sentences and labels.
    """
    sentences = []
    labels = []
    
    with open(filepath, 'r') as f:
        for line in f:
            # Split each line into label and sentence, assuming they are separated by a tab or space
            parts = line.strip().split(maxsplit=1)  # Splits into two parts: label and sentence
            if len(parts) == 2:  # Ensure there's a label and a sentence
                label = int(parts[0])  # The first part is the label, convert it to an integer
                sentence = parts[1]  # The second part is the sentence
                labels.append(label)  # Store the label
                sentences.append(sentence)  # Store the sentence
    
    return sentences, labels


class SentimentDatasetBPE(Dataset):
    def __init__(self, infile, bpe_model, max_seq_len=512):
        """
        Dataset class that tokenizes sentences into subwords using BPE.
        
        :param infile: Path to the input file containing the dataset.
        :param bpe_model: The BPE model for subword tokenization.
        :param max_seq_len: Maximum sequence length for padding/truncation.
        """
        self.examples, self.labels = read_sentiment_examples_BPE(infile)  # List of sentences and labels
        self.examples = [sentence.lower() for sentence in self.examples]  # Lowercase all sentences
        self.bpe_model = bpe_model  # BPE tokenizer model
        self.max_seq_len = max_seq_len
        self.sentence_indices, self.labels = self._prepare_data()

    def _prepare_data(self):
        sentences = [sentence for sentence in self.examples]  # Extract raw sentences
        sentence_indices = []
        for sentence in sentences:
            subword_ids = []
            for word in sentence.split():  # Tokenize each word into subwords
                subword_ids.extend(self.bpe_model.encode(word))  # Encode the word into subword units
                
            # Pad or truncate to max_seq_len
            if len(subword_ids) < self.max_seq_len:
                subword_ids += [0] * (self.max_seq_len - len(subword_ids))  # Use 0 as padding index
            else:
                subword_ids = subword_ids[:self.max_seq_len]  # Truncate to max_seq_len
            
            sentence_indices.append(subword_ids)
        
        # Convert sentences (subword token ids) and labels to tensors
        sentence_indices = torch.tensor(sentence_indices, dtype=torch.long)  # Subword indices should be integers
        labels = torch.tensor(self.labels, dtype=torch.long)  # Convert labels to tensor
        
        return sentence_indices, labels

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.examples)  # This returns the length of the dataset (number of sentences)

    def __getitem__(self, idx):
        return self.sentence_indices[idx], self.labels[idx]


class DAN_BPE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, frozen=False):
        super(DAN_BPE, self).__init__()
        
        # Initialize the embedding layer for subwords
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, subword_indices):
        subword_indices = subword_indices.long()
        embedded = self.embedding(subword_indices)
        avg_embedding = torch.mean(embedded, dim=1)
        x = F.relu(self.fc1(avg_embedding))
        x = self.fc2(x)
        output = self.log_softmax(x)
        return output
