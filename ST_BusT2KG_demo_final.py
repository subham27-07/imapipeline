# import all packages
import streamlit as st
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# tokenizer
from transformers import AutoTokenizer, DistilBertTokenizerFast
# sequence tagging model + training-related 
from transformers import DistilBertForTokenClassification, Trainer, TrainingArguments
import numpy as np
import pandas as pd
import torch
import json
import sys
import os
from datasets import load_metric
from sklearn.metrics import classification_report
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import json
import re
import numpy as np 
import pandas as pd
import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import itertools
import json
import glob
from transformers import TextClassificationPipeline, TFAutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import csv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():

  st.title("Text to Causal Knowledge Graph")
  st.sidebar.title("Please upload your text documents in one file here:")

  uploaded_file = st.sidebar.file_uploader("Choose a file")

   # with open(uploaded_file, encoding='utf-8') as f:
  if uploaded_file is not None:
    # st.write(uploaded_file)
    # data = []
    # condition = '5'
    # ind_append = False

    # bytes_data = uploaded_file.getvalue()
    # data = bytes_data.decode('utf-8')#
    uploaded_file_df = pd.read_csv(uploaded_file, sep=".", header=None)


  k=2
  seed = 1
  k1= 5


  # GUI to upload PDFs
  # Inputs: 10K PDFs (Sreekar)

  # Output is an NLP df
  # Step 1: Extracting causal statements from PDFs  (Sreekar)



  # GUI to upload PDFs
  # Inputs: 10K PDFs (Sreekar)

  # Output is an NLP df
  # Step 1: Extracting causal statements from PDFs  (Sreekar)
  def clean_text(text):
      text = str(text).lower()
      text = re.sub('\[.*?\]', '', text)
      text = re.sub('https?://\S+|www\.\S+', '', text)
      text = re.sub('<.*?>+', '', text)
      text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
      text = re.sub('\n', '', text)
      text = re.sub('\w*\d\w*', '', text)
      text = [word for word in text.split(' ')]
      text=" ".join(text)
      text = [stemmer.stem(word) for word in text.split(' ')]
      text=" ".join(text)
      return text

  # class CustomDataset(torch.utils.data.Dataset):
  #     def __init__(self, encodings):
  #         self.encodings = encodings
  #         #self.labels = labels

  #     def __getitem__(self, idx):
  #         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
  #         #item['labels'] = torch.tensor(self.labels[idx])
  #         return item

  #     def __len__(self):
  #         return len(self.encodings)

  # with open('/content/ALL.txt','r', encoding="utf-8") as a:
  #         file_content = a.read()
  #         review = re.sub('[^a-zA-Z.]', ' ', file_content)
  #         review = review.split('.')
  #         review = [x for x in review if "us gaap" not in x]
  #         review = '.'.join(review)
  #         json_content = {}
  #         json_content['raw-text'] = review
          
  #         with open('/content/ALL.json', 'w+', encoding="utf-8") as outfile:
  #             json.dump(json_content, outfile,sort_keys=True, indent=4)

  # company_count_dict = {}
  # #company_name = 'ALL'
  # sent_list = []
  # with open('ALL.json', 'r') as f:
  #     data = json.load(f)



  final_data = uploaded_file_df.to_string()
  #final_data = data['raw-text']
  #print(type(final_data))
  #remove the numbers from the file

  result = re.sub(r'\d+', '', final_data)

  #convert all the text into lower case
  result = result.lower()

  final_data1=result.split('.')

  #remove punctuations
  result =[]
  for i in final_data1:
    result.append(re.sub(r'[^\w\s]','',i))

  #remove non words
  import nltk
  nltk.download('words')
  new_result =[]
  words = set(nltk.corpus.words.words())

  def clean_sent(sent):
      return " ".join(w for w in nltk.wordpunct_tokenize(sent) \
       if w.lower() in words or not w.isalpha())

  #result = clean_sent(result)
  for i in result:
    new_result.append(clean_sent(i))

  new_result1 =[]
  for i in new_result:
    if i:
      new_result1.append(i)
  df = pd.DataFrame(new_result1,columns=['text'])
  df['text']=df['text'].apply(clean_text)

  test = df['text'].to_list()

  pipeline('text-classification')
  model_name = "bert-base-uncased"
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

  test_data = tokenizer(df['text'].tolist(), padding="max_length", truncation=True, return_attention_mask=True)

  model_path = "checkpoint-2850"
  model = AutoModelForSequenceClassification.from_pretrained(model_path, id2label={0:'non-causal',1:'causal'}) #, id2label={0:'non-causal',1:'causal'}

  pipe1 = pipeline("text-classification", model=model,tokenizer=tokenizer)
  causal_sents = []
  for sent in test:
    pred = pipe1(sent)
    #print(pred)
    for lab in pred:
      #print(lab)
      if lab['label'] == 'causal':
        causal_sents.append(sent)

  pipeline('sentiment-analysis')
  model_name = "distilbert-base-cased"
  tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
  # # convenience function for wordpiece tokenization of a list of tokens
  tokenize = lambda ds: tokenizer(ds, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, return_tensors="pt")
  RANDOM_SEED = 42

  X_test= tokenize(causal_sents)

  class SimpleDataset:
      def __init__(self, tokenized_texts):
          self.tokenized_texts = tokenized_texts
      
      def __len__(self):
          return len(self.tokenized_texts["input_ids"])
      
      def __getitem__(self, idx):
          return {k: v[idx] for k, v in self.tokenized_texts.items()}

      # def __len__(self):
      #     return len(self.encodings)

  test_dataset = SimpleDataset(causal_sents)

  for X in [X_test]:
     X.pop("offset_mapping")

  model_path = 'DistilBertforTokenClassification'
  model = DistilBertForTokenClassification.from_pretrained(model_path, id2label={0:'B-E',1:'B-C',2:'I-C',3:'O',4:'I-CT',5:'I-E',6:'B-CT'}) #len(unique_tags),, num_labels= 7,
  #loader_collate = DataLoader(X_test, shuffle=True, batch_size=5, collate_fn=dummy_data_collector)
  pipe = pipeline('ner', model=model, tokenizer=tokenizer,grouped_entities=True) #grouped_entities=True


  #pred= pipe(flat_list)
  #test_data = pd.DataFrame(class_list)

  sentence_pred = []
  class_list = []
  entity_list = []
  for k in causal_sents:
    pred= pipe(k)
    for i in pred:
      #for j in i:
      
      sentence_pred.append(k)
      class_list.append(i['word'])
      entity_list.append(i['entity_group'])

  filename = 'Checkpoint-classification.sav'
  count_vect = CountVectorizer(ngram_range=[1,3])
  tfidf_transformer=TfidfTransformer()
  loaded_model = pickle.load(open(filename, 'rb'))
  loaded_vectorizer = pickle.load(open('vectorizefile_classification.pickle', 'rb'))

  pipeline_test_output = loaded_vectorizer.transform(class_list)
  predicted = loaded_model.predict(pipeline_test_output)

  #print(np.shape(predicted))
  pred1 = predicted
  level0 = []
  count =0
  for i in predicted:
    if i == 3:
      level0.append('Non-Performance')
      count +=1
    else:
      level0.append('Performance')
      count +=1

  list_pred = {0: 'Customers',1:'Employees',2:'Investors',3:'Non-performance',4:'Society',5:'Unclassified'}
  pred_val = [list_pred[i] for i in pred1]

  #print('count',count)

  sent_id, unique = pd.factorize(sentence_pred) 

  final_list = pd.DataFrame(
      {'Id': sent_id,
       'Full sentence': sentence_pred,
       'Component': class_list,
       'cause/effect': entity_list,
       'Label level1': level0,
       'Label level2': pred_val
      })
  s = final_list['Component'].shift(-1)
  m = s.str.startswith('##', na=False)
  final_list.loc[m, 'Component'] += (' ' + s[m])


  final_list1 = final_list[~final_list['Component'].astype(str).str.startswith('##')]

  li = []
  uni = final_list1['Id'].unique()
  for i in uni:
    df_new = final_list1[final_list1['Id'] == i]
    uni1 = df_new['Id'].unique()
    if 'E' not in df_new.values:
      li.append(uni1)
  out = np.concatenate(li).ravel()
  li_pan = pd.DataFrame(out,columns=['Id'])
  df3 = pd.merge(final_list1, li_pan[['Id']], on='Id', how='left', indicator=True) \
              .query("_merge == 'left_only'") \
              .drop('_merge',1)
    
  df = df3.groupby(['Id','Full sentence','cause/effect', 'Label level1', 'Label level2'])['Component'].apply(', '.join).reset_index()

  cause = len(final_list1[final_list1['cause/effect'] == 'C'])
  effect = len(final_list1[final_list1['cause/effect'] == 'E'])
  eff = final_list1[final_list1['cause/effect'] == 'E']
  cau = final_list1[final_list1['cause/effect'] == 'C']

  Nperf_cau = len(cau[cau['Label level1'] == 'Non-Performance'])
  Nperf_eff = len(eff[eff['Label level1'] == 'Non-Performance'])
  perf_eff = len(eff[eff['Label level1'] == 'Performance'])
  perf_cau = len(cau[cau['Label level1'] == 'Performance'])

  Inv_eff = len(eff[eff['Label level2'] == 'Investors'])
  Cus_eff = len(eff[eff['Label level2'] == 'Customers'])
  Emp_eff = len(eff[eff['Label level2'] == 'Employees'])
  Soc_eff = len(eff[eff['Label level2'] == 'Society'])

  Inv_cau = len(cau[cau['Label level2'] == 'Investors'])
  Cus_cau = len(cau[cau['Label level2'] == 'Customers'])
  Emp_cau = len(cau[cau['Label level2'] == 'Employees'])
  Soc_cau = len(cau[cau['Label level2'] == 'Society'])

  Cau_eff_NP = Nperf_cau + Nperf_eff
  cau_NP_eff_inv = Nperf_cau + Inv_eff
  cau_NP_eff_cus = Nperf_cau + Cus_eff
  cau_NP_eff_Emp = Nperf_cau + Emp_eff
  cau_NP_eff_soc = Nperf_cau + Soc_eff

  cau_inv_eff_NP = Inv_cau + Nperf_eff
  cau_inv_eff_inv = Inv_cau + Inv_eff
  cau_inv_eff_cus = Inv_cau + Cus_eff
  cau_inv_eff_Emp = Inv_cau + Emp_eff
  cau_inv_eff_soc = Inv_cau + Soc_eff

  cau_cus_eff_NP = Cus_cau + Nperf_eff
  cau_cus_eff_inv = Cus_cau + Inv_eff
  cau_cus_eff_cus = Cus_cau + Cus_eff
  cau_cus_eff_emp = Cus_cau + Emp_eff
  cau_cus_eff_soc = Cus_cau + Soc_eff

  cau_Emp_eff_NP = Emp_cau + Nperf_eff
  cau_Emp_eff_inv = Emp_cau + Inv_eff
  cau_Emp_eff_cus = Emp_cau + Cus_eff
  cau_Emp_eff_Emp = Emp_cau + Emp_eff
  cau_Emp_eff_Soc = Emp_cau + Soc_eff

  cau_Soc_eff_NP = Soc_cau + Nperf_eff
  cau_Soc_eff_inv = Soc_cau + Inv_eff
  cau_Soc_eff_cus = Soc_cau + Cus_eff
  cau_Soc_eff_Emp = Soc_cau + Emp_eff
  cau_Soc_eff_Soc = Soc_cau + Soc_eff

  df_tab = pd.DataFrame({
      'Non-performance': [Cau_eff_NP, cau_NP_eff_inv, cau_NP_eff_cus, cau_NP_eff_Emp, cau_NP_eff_soc],
      'Investors': [cau_inv_eff_NP, cau_inv_eff_inv, cau_inv_eff_cus, cau_inv_eff_Emp, cau_inv_eff_soc],
      'Customers': [cau_cus_eff_NP, cau_cus_eff_inv, cau_cus_eff_cus, cau_cus_eff_emp, cau_cus_eff_soc],
      'Employees': [cau_Emp_eff_NP, cau_Emp_eff_inv, cau_Emp_eff_cus, cau_Emp_eff_Emp, cau_Emp_eff_Soc],
      'Society': [cau_Soc_eff_NP, cau_Soc_eff_inv, cau_Soc_eff_cus, cau_Soc_eff_Emp, cau_Soc_eff_Soc]},
       index=['Non-performance', 'Investors', 'Customers', 'Employees', 'Society'])

  df_tab.to_csv('final_data.csv')

  df["cause/effect"].replace({"C": "cause", "E": "effect"}, inplace=True)
  df_final = df[df['cause/effect'] != 'CT']
  df['New string'] = df_final['Component'].replace(r'[##]+', ' ', regex=True)
  df_final = df_final.drop('Component',1)
  df_final.insert(2, "Component", df['New string'], True)

  # print(final_list)
  # print(final_list1)
  df_final.to_csv('predictions.csv')
  #print(df_final)

  # st.download_button(
  #      "Download causal knowledge data",
  #      csv,
  #      "/Users/seetha/Desktop/Individualstudy/RA/ProtoType_final/predictions.csv",
  #      "text/csv",
  #      key='download-final'
  #   )

  def convert_df(df):

  #IMPORTANT: Cache the conversion to prevent computation on every rerun

    return df.to_csv().encode('utf-8')

               

  csv = convert_df(report.astype(str))

 
  st.download_button(label="Download the result table",data=csv,file_name='results.csv',mime='text/csv')


if __name__ == '__main__':
    main()
