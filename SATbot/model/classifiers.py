import pytorch_lightning as pl
import textdistance as td
import numpy as np
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoModelWithLMHead,
    AutoTokenizer
)
from tokenizers import ByteLevelBPETokenizer, BertTokenizer

from tokenizers.processors import BertProcessing

#T5:
class T5FineTuner(pl.LightningModule):
  def __init__(self, hparams):
    super(T5FineTuner, self).__init__()
    self.hparams = hparams

    self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
    self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)

  def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
  ):
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        lm_labels=lm_labels,
    )

args_dict = dict(
    model_name_or_path='t5-base',
    tokenizer_name_or_path='t5-base',
)

args = argparse.Namespace(**args_dict)


#RoBERTa:
@torch.jit.script
def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def forward(self, input):
        return mish(input)

class EmoClassificationModel(nn.Module):
    def __init__(self, base_model, n_classes, base_model_output_size=768, dropout=0.05):
        super().__init__()
        self.base_model = base_model

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_model_output_size, base_model_output_size),
            Mish(),
            nn.Dropout(dropout),
            nn.Linear(base_model_output_size, n_classes)
        )

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, input_, *args):
        X, attention_mask = input_
        hidden_states = self.base_model(X, attention_mask=attention_mask)

        return self.classifier(hidden_states[0][:, 0, :])

#labels for emotion classification
labels = [ "sadness", "joy", "anger", "fear", "love", "instability", "disgust", "disappointment", "shame", "guilt", "envy", "jealous" ]
label2int = dict(zip(labels, list(range(len(labels)))))

#load emotion classifier (Megatron-BERT)
#Megatron's tokniser and pretrained model have been downloaded
#from a publicly availably checkpoint and are located in the
#nvidia folder
with torch.no_grad():
    bert_model = EmoClassificationModel(AutoModelWithLMHead.from_pretrained("nvidia/megatron-bert-cased-345m").base_model, len(labels))
    bert_model.load_state_dict(torch.load('/mnt/c/Users/Rado/Desktop/Individual Project/SATbot/NLP models/Emotion classification/Megatron_BERT_finetuned.pt', map_location=torch.device('cpu'))) #change path

#load emotion classifier (T5)
# with torch.no_grad():
#    emomodel = T5FineTuner(args)
#    emomodel.load_state_dict(torch.load('/mnt/c/Users/Rado/Desktop/Individual Project/SATbot/NLP models/Emotion classification/T5_finetuned_syn.pt', map_location=torch.device('cpu')))

#load empathy classifier (T5)
with torch.no_grad():
    t5model = T5FineTuner(args)
    t5model.load_state_dict(torch.load('/mnt/c/Users/Rado/Desktop/Individual Project/SATbot/NLP models/Empathy classification/T5_student_RoBERTa_teacher.pt', map_location=torch.device('cpu'))) #change path

#Load pre-trained GPT2 language model weights
with torch.no_grad():
    gptmodel = GPT2LMHeadModel.from_pretrained('gpt2')
    gptmodel.eval()

#Load pre-trained GPT2 tokenizer
gpttokenizer = GPT2Tokenizer.from_pretrained('gpt2')

#simple tokenizer + stemmer
regextokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stemmer = nltk.stem.PorterStemmer()


#def get_emotion(text):
#  text = re.sub(r'[^\w\s]', '', text)
#  text = text.lower()
#  with torch.no_grad():
#      input_ids = emomodel.tokenizer.encode(text + '</s>', return_tensors='pt')
#      output = emomodel.model.generate(input_ids=input_ids, max_length=2)
#      dec = [emomodel.tokenizer.decode(ids) for ids in output]
#  label = dec[0]
#  return label


def get_emotion(text):
  '''
  Classifies and returns the underlying emotion of a text string
  '''
  text = re.sub(r'[^\w\s]', '', text)
  text = text.lower()
  tokenizer = BertTokenizer.from_pretrained("nvidia/megatron-bert-cased-345m")
  encoded = tokenizer.encode(text)
  sequence_padded = torch.tensor(encoded.ids).unsqueeze(0)
  attention_mask_padded = torch.tensor(encoded.attention_mask).unsqueeze(0)
  with torch.no_grad():
      output = bert_model((sequence_padded, attention_mask_padded))
  top_p, top_class = output.topk(1, dim=1)
  label = int(top_class[0][0])
  label_map = {v: k for k, v in label2int.items()}
  return label_map[label]


def empathy_score(text):
  '''
  Computes a discrete numerical empathy score for an utterance (scale 0 to 2)
  '''
  with torch.no_grad():
      input_ids = t5model.tokenizer.encode(text + '</s>', return_tensors='pt')
      output = t5model.model.generate(input_ids=input_ids, max_length=2)
      dec = [t5model.tokenizer.decode(ids) for ids in output]
  label = dec[0]
  if label == 'no':
    score = 0.0
  elif label == 'weak':
    score = 1.0
  else:
    score = 2.0
  #normalise between 0 and 1 dividing by the highest possible value:
  return score/2


def perplexity(sentence):
  '''
  Computes the PPL of an utterance using GPT2 LM
  '''
  tokenize_input = gpttokenizer.encode(sentence)
  tensor_input = torch.tensor([tokenize_input])
  with torch.no_grad():
      loss = gptmodel(tensor_input, labels=tensor_input)[0]
  return np.exp(loss.detach().numpy())


def repetition_penalty(sentence):
  '''
  Adds a penalty for each repeated (stemmed) token in
  an utterance. Returns the total penalty of the sentence
  '''
  word_list = regextokenizer.tokenize(sentence.lower())
  filtered_words = [word for word in word_list if word not in stopwords.words('english')]
  stem_list = [stemmer.stem(word) for word in filtered_words]
  penalty = 0
  visited = []
  for w in stem_list:
    if w not in visited:
      visited.append(w)
    else:
      penalty += 0.005
  return penalty


def fluency_score(sentence):
  '''
  Computes the fluency score of an utterance, given by the
  inverse of the perplexity minus a penalty for repeated tokens
  '''
  ppl = perplexity(sentence)
  penalty = repetition_penalty(sentence)
  score = (1 / ppl) - penalty
  #normalise by the highest possible fluency computed on the corpus:
  normalised_score = score / 0.16
  if normalised_score < 0:
    normalised_score = 0
  return round(normalised_score, 2)


def get_distance(s1, s2):
  '''
  Computes a distance score between utterances calculated as the overlap
  distance between unigrams, plus the overlap distance squared over bigrams,
  plus the overlap distance cubed over trigrams, etc (up to a number of ngrams
  equal to the length of the shortest utterance)
  '''
  s1 = re.sub(r'[^\w\s]', '', s1.lower()) #preprocess
  s2 = re.sub(r'[^\w\s]', '', s2.lower())
  s1_ws = regextokenizer.tokenize(s1) #tokenize to count tokens later
  s2_ws = regextokenizer.tokenize(s2)
  max_n = len(s1_ws) if len(s1_ws) < len(s2_ws) else len(s2_ws) #the max number of bigrams is the number of tokens in the shorter sentence
  ngram_scores = []
  for i in range(1, max_n+1):
    s1grams = nltk.ngrams(s1.split(), i)
    s2grams = nltk.ngrams(s2.split(), i)
    ngram_scores.append((td.overlap.normalized_distance(s1grams, s2grams))**i) #we normalize the distance score to be a value between 0 and 10, before raising to i
  normalised_dis = sum(ngram_scores)/(max_n) #normalised
  return normalised_dis


def compute_distances(sentence, dataframe):
  '''
  Computes a list of distances score between an utterance and all the utterances in a dataframe
  '''
  distances = []
  for index, row in dataframe.iterrows():
    df_s = dataframe['sentences'][index] #assuming the dataframe column is called 'sentences'
    distance = get_distance(df_s.lower(), sentence)
    distances.append(distance)
  return distances


def novelty_score(sentence, dataframe):
  '''
  Computes the mean of the distances beween an utterance
  and each of the utterances in a given dataframe
  '''
  if dataframe.empty:
    score = 1.0
  else:
    d_list = compute_distances(sentence, dataframe)
    d_score = sum(d_list)
    score = d_score / len(d_list)
  return round(score, 2)


def get_sentence_score(sentence, dataframe):
  '''
  Calculates how fit a sentence is based on its weighted empathy, fluency
  and novelty values
  '''
  empathy = empathy_score(sentence)
  fluency = fluency_score(sentence)
  novelty = novelty_score(sentence, dataframe)
  score = empathy + 0.675*fluency + 2*novelty
  return score
