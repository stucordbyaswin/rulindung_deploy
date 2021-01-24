import re
import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from app import app

def setup():
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('GPU yang digunakan:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device


def text_cleansing(text):
    text = text.lower()
    text = text.strip()
    word_list = text.split()
    
    nu_words = []
    for words in word_list:
        words = re.sub(r'\W+', '', words)
        words = re.sub(r'http\S+', '', words)
        words = re.sub(r'@\S+', '', words)
        words = re.sub(r'#\S+', '', words)
        words = re.sub(r'\\\S+', '', words)
        words = re.sub(r'\burl\b', '', words)
        words = re.sub(r'\buser\b', '', words)
        words = re.sub(r'\bxd\b', '', words)
        words = re.sub(r'&amp\S+', 'dan', words)
        words = re.sub(r'&gt\S+', '', words)
        words = re.sub(r'&lt\S+', '', words)
        words = re.sub(r'\brt\b', '', words)
        words = re.sub(r'[^\w\s]', ' ', words) # remove punctuations
        words = re.sub(r'[_]', ' ', words) # remove more punctuation
        words = re.sub(r'ð', '', words)
        words = re.sub(r'â', '', words)

        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        words = emoji_pattern.sub(r'', words)

        # words = regex.sub(u'[^\p{Latin}]', '', words)

        nu_words.append(words)
  
    text = ' '.join(nu_words)
    text = text.split()
    text = ' '.join(text)
    text.strip()
    return text

def predict(text):
    device = setup()

    preprocessed_text = text_cleansing(text)
    print('Preprocessed text:', preprocessed_text)

    print('Loading model...')
    model = BertForSequenceClassification.from_pretrained(app.config["MODEL_2_PATH"])
    tokenizer = BertTokenizer.from_pretrained(app.config["MODEL_2_PATH"])

    # Copy the model to the GPU.
    model.to(device)
    print('Model has loaded.')

    encoded_dict = tokenizer.encode_plus(
                        preprocessed_text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,           # Pad & truncate all sentences.
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                        truncation = True,
                        padding = 'max_length'
                   )

    # encoded_dict = encoded_dict.to(device)
    
    # Add the encoded sentence to the list.    
    input_ids = encoded_dict['input_ids']
    input_ids = input_ids.to(device)
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks = encoded_dict['attention_mask']
    attention_masks = attention_masks.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
        # outputs.to(device)

    print('Outputs:', outputs)
    
    logits = outputs[0]
    softmax = torch.nn.functional.softmax(logits)
    
    logits = logits.detach().cpu().numpy()
    softmax = softmax.detach().cpu().numpy()


    print('Logits:', logits)
    print('Softmax:', softmax)

    label_id = np.argmax(logits, axis=1).flatten()
    percentage = np.max(softmax * 100)

    if label_id == 0:
        label_name = 'Non-Kekerasan'
    elif label_id == 1:
        label_name = 'Kekerasan'

    prediction = 'Text: {}. \n Konten ini adalah {} ({:.0f}%)'.format(preprocessed_text, label_name, percentage)
    print(prediction)

    
    return prediction # prediction