from transformers import AutoTokenizer, AutoModel
import conllu
import seaborn as sb
import streamlit as st
import time
def load_conllu(filename, encoding='utf-8'):
    with open(filename, encoding=encoding) as fp:
        data = conllu.parse(fp.read())
    sentences = [[token['form'] for token in sentence] for sentence in data]
    taggings = [[token['xpos'] for token in sentence] for sentence in data]
    return sentences, taggings
train_sentences, train_taggings = load_conllu('te_mtg-ud-train.conllu')
valid_sentences, valid_taggings = load_conllu('te_mtg-ud-dev.conllu')
test_sentences, test_taggings = load_conllu('te_mtg-ud-test.conllu')
import collections
tagset = collections.defaultdict(int)

for tagging in train_taggings:
    for tag in tagging:
        tagset[tag] += 1
# print count and tag sorted by decreasing count
for tag, count in sorted(tagset.items(), reverse=True, key=lambda x: x[1]):
  print(count, tag)
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

# tokenize an example sentence
tokenizer.tokenize('చూసేరా అండీ ?')
import re

def align_tokenizations(sentences, taggings):
    bert_tokenized_sentences = []
    aligned_taggings = []

    for sentence, tagging in zip(sentences, taggings):
        # first generate BERT-tokenization
        bert_tokenized_sentence = tokenizer.tokenize(' '.join(sentence))

        aligned_tagging = []
        current_word = ''
        index = 0  # index of the current word in the sentence and tagging
        for token in bert_tokenized_sentence:
            current_word += re.sub(r'^##', '', token)  # recompose the word with subtokens
            sentence[index] = sentence[index].replace('\xad', '')  # fix bug in data
            # note that some word factors correspond to unknown words in BERT
            assert token == '[UNK]' or sentence[index].startswith(current_word)

            if token == '[UNK]' or sentence[index] == current_word:  # if we completed a word
                current_word = ''
                aligned_tagging.append(tagging[index])
                index += 1
            else:  # otherwise insert padding
                aligned_tagging.append('<pad>')

        assert len(bert_tokenized_sentence) == len(aligned_tagging)

        bert_tokenized_sentences.append(bert_tokenized_sentence)
        aligned_taggings.append(aligned_tagging)

    return bert_tokenized_sentences, aligned_taggings


train_bert_tokenized_sentences, train_aligned_taggings = align_tokenizations(train_sentences, train_taggings)
valid_bert_tokenized_sentences, valid_aligned_taggings = align_tokenizations(valid_sentences, valid_taggings)
test_bert_tokenized_sentences, test_aligned_taggings = align_tokenizations(test_sentences, test_taggings)
import torch
device = torch.device( 'cpu')

import collections

label_vocab = collections.defaultdict(lambda: len(label_vocab))
label_vocab['<pad>'] = 0

def convert_to_ids(sentences, taggings):
    sentences_ids = []
    taggings_ids = []
    for sentence, tagging in zip(sentences, taggings):
        sentence_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(['[CLS]'] + sentence + ['SEP'])).long()
        tagging_tensor = torch.tensor([0] + [label_vocab[tag] for tag in tagging] + [0]).long()

        sentences_ids.append(sentence_tensor.to(device))
        taggings_ids.append(tagging_tensor.to(device))
    return sentences_ids, taggings_ids

train_sentences_ids, train_taggings_ids = convert_to_ids(train_bert_tokenized_sentences, train_aligned_taggings)
valid_sentences_ids, valid_taggings_ids = convert_to_ids(valid_bert_tokenized_sentences, valid_aligned_taggings)
test_sentences_ids, test_taggings_ids = convert_to_ids(test_bert_tokenized_sentences, test_aligned_taggings)
from torch.utils.data import Dataset

class PosTaggingDataset(Dataset):
    def __init__(self, sentences, taggings):
        assert len(sentences) == len(taggings)
        self.sentences = sentences
        self.taggings = taggings

    def __getitem__(self, i):
        return self.sentences[i], self.taggings[i]

    def __len__(self):
        return len(self.sentences)
def collate_fn(items):
    max_len = max(len(item[0]) for item in items)

    sentences = torch.zeros((len(items), max_len), device=items[0][0].device).long().to(device)
    taggings = torch.zeros((len(items), max_len)).long().to(device)

    for i, (sentence, tagging) in enumerate(items):
        sentences[i][0:len(sentence)] = sentence
        taggings[i][0:len(tagging)] = tagging

    return sentences, taggings

x, y = collate_fn([[torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])], [torch.tensor([1, 2]), torch.tensor([3, 4])]])

from torch.utils.data import DataLoader

batch_size = 64

train_loader = DataLoader(PosTaggingDataset(train_sentences_ids, train_taggings_ids), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
valid_loader = DataLoader(PosTaggingDataset(valid_sentences_ids, valid_taggings_ids), batch_size=batch_size, collate_fn=collate_fn)
test_loader = DataLoader(PosTaggingDataset(test_sentences_ids, test_taggings_ids), batch_size=batch_size, collate_fn=collate_fn)
import torch.nn as nn
import torch.nn.functional as F

class GRUClassifier(nn.Module):
    def __init__(self, num_labels, embed_size=300, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(tokenizer.vocab_size, embed_size, padding_idx=tokenizer.pad_token_id)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.decision = nn.Linear(1 * 2 * hidden_size, num_labels)  # size output by GRU is number of layers * number of directions * hidden size
        self.to(device)
  
    def forward(self, sentences):
        embed_rep = self.embedding(sentences)
        word_rep, sentence_rep = self.gru(embed_rep)
        return self.decision(F.dropout(F.gelu(word_rep), 0.3))
def perf(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # do not apply training-specific steps such as dropout
    total_loss = correct = num_loss = num_perf = 0
    for x, y in loader:
        with torch.no_grad():  # no need to store computation graph for gradients
            # perform inference and compute loss
            y_scores = model(x)
            loss = criterion(y_scores.view(-1, len(label_vocab)), y.view(-1))  # requires tensors of shape (num-instances, num-labels) and (num-instances)

            # gather loss statistics
            total_loss += loss.item()
            num_loss += 1

            # gather accuracy statistics
            y_pred = torch.max(y_scores, 2)[1]  # compute highest-scoring tag
            mask = (y != 0)  # ignore <pad> tags
            correct += torch.sum((y_pred == y) * mask)  # compute the number of correct predictions
            num_perf += torch.sum(mask).item()
    return total_loss / num_loss, correct.item() / num_perf


import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import accuracy_score

def fit(model, epochs, patience=5, initial_lr=1e-2, gamma=0.95):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = ExponentialLR(optimizer, gamma)  # Exponential learning rate scheduler
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = num = 0
        correct_train = num_train = 0
        for x, y in train_loader:
            optimizer.zero_grad()  # start accumulating gradients
            y_scores = model(x)
            loss = criterion(y_scores.view(-1, len(label_vocab)), y.view(-1))
            loss.backward()  # compute gradients through the computation graph
            optimizer.step()  # modify model parameters
            total_loss += loss.item()
            num += 1

            # Calculate training accuracy
            y_pred = torch.max(y_scores, 2)[1]
            mask = (y != 0)
            correct_train += torch.sum((y_pred == y) * mask)
            num_train += torch.sum(mask).item()

        train_acc = correct_train.item() / num_train  # Calculate training accuracy

        # Print the learning rate for this epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}: Loss: {total_loss / num}, Training Accuracy: {train_acc}, Learning Rate: {current_lr}")

        # Calculate validation loss and accuracy
        val_loss, val_acc = perf(model, valid_loader)
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

        # Step the learning rate scheduler
        scheduler.step()

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break
gru_model = GRUClassifier(len(label_vocab))
#fit(gru_model, 100,3)
#torch.save(gru_model.state_dict(), 'gru_model_weights.pth')
gru_model = GRUClassifier(len(label_vocab))
gru_model.load_state_dict(torch.load('gru_model_weights.pth'))
gru_model.eval() 
def predict_tags(user_input, model, tokenizer, label_vocab):
    # Tokenize user input
    tokenized_input = tokenizer.tokenize(user_input)
    
    # Convert tokenized input to tensor format
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenized_input + ['[SEP]'])]).to(device)
    
    # Get model predictions
    with torch.no_grad():
        model.eval()
        predicted_ids = model(input_ids)

    # Convert predicted tensor IDs to tag labels
    predicted_tags = []
    for tensor_id in predicted_ids.argmax(dim=-1).squeeze().tolist():
        tag = list(label_vocab.keys())[list(label_vocab.values()).index(tensor_id)]
        if tag != '<pad>':
            predicted_tags.append(tag)

    return tokenized_input, predicted_tags



def main():
    bg_img = '''
        <style>
        [data-testid="stAppViewContainer"] {
        background-image: url('https://w0.peakpx.com/wallpaper/560/855/HD-wallpaper-dark-blue-iphone-best-classic-edge-htc-lg-samsung-galaxy-style-display-summer.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        }
        </style>
        '''
    st.markdown(bg_img, unsafe_allow_html=True)
    st.title('POS Tagging with Machine Learning')
    user_input = st.text_input('Enter a sentence in Telugu:', value='చూసేరా అండీ ?')

    if st.button('Predict POS Tags'):
        with st.spinner('Predicting...'):
            tokenized_input, predicted_tags = predict_tags(user_input, gru_model, tokenizer, label_vocab)

            time.sleep(3)  # Simulating a delay to show the spinner

        st.success('Prediction Complete!')
        
        st.subheader("Predicted POS Tags")
        animated_html = f"""
        <style>
            [data-testid="stAppViewContainer"] {{
            background-image: url('https://w0.peakpx.com/wallpaper/560/855/HD-wallpaper-dark-blue-iphone-best-classic-edge-htc-lg-samsung-galaxy-style-display-summer.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            }}
            @keyframes typing {{
                from {{ width: 0 }}
                to {{ width: 100% }}
            }}
            .typewriter {{
                overflow: hidden;
                white-space: nowrap;
                animation: typing 5s steps(40, end);
                border-right: 3px solid;
                font-size: 18px;
                font-family: 'Courier New', monospace;
                color:white
            }}
        </style>
        <p class="typewriter">{ ' '.join(predicted_tags[:-1]) }</p>
        """
        st.components.v1.html(animated_html, height=100)

if __name__ == '__main__':
    main()