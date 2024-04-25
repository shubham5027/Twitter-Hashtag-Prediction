#TechVidvan Project
#import all the required libraries
import numpy as np
import pandas as pd
import pickle
from statistics import mode
import nltk
from nltk import word_tokenize
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input,LSTM,Embedding,Dense,Concatenate,Attention
from sklearn.model_selection import train_test_split
nltk.download('wordnet')
nltk.download('punkt')

#read the dataset file
train=pd.read_csv("train.csv")
#tweet column is input 
inp_data=train["tweet"]
#target data is sentiment(s1,s2,s3,s4,s5) , 
#when (w1,w2,w3,w4) and kind(k1,k2,k3...k15)
tar_data=train.iloc[:,4:].values

#get the column name of target
tar_lab=train.iloc[:,4:].columns.tolist()
#value of the target label like
#s1="I can't tell" , s2="Negative" and so on till s5
#w1="current weather", w2=future forecast and so on till w4
#k1="clouds", k2="cold", k3="dry" and so on till k15
tar_lab_val=[
"I can't tell","Negative","Neutral","Positive","Tweet not related to weather condition",
"current (same day) weather","future (forecast)","I can't tell","past weather",
"clouds","cold","dry","hot","humid","hurricane","I can't tell","ice","other","rain",
"snow","storms","sun","tornado","wind"]

inp_texts=[]
tar_texts=[]
inp_words=[]
tar_words=[]
contractions= pickle.load(open("contractions.pkl","rb"))['contractions']

#clean the tweets
def clean(tweet):
  #replace and lower the tweets
  tweet=tweet.replace(":","").lower()
  #get only words that contains alphabets
  words= list(filter(lambda w:(w.isalpha()),tweet.split(" ")))
  #expand the shortened words
  words= [contractions[w] if w in contractions else w for w in words ]
  #return all the words
  return words

#iterate over input data
for tweet in inp_data:
  #clean the tweets
  inpt_words= clean(tweet)
  #store the input texts and words
  inp_texts+= [' '.join(inpt_words)]
  inp_words+= inpt_words

#iterate over target data
for lab in tar_data:
  #get index of maximum value from sentiment data(s1 to s5)
  #with the help of this index get label value
  senti=tar_lab[np.argmax(lab[:5])]
  #get index of maximum value from when data(w1 to w4)
  #with the help of this index get label value
  when=tar_lab[np.argmax(lab[5:9])+5] 
  #get index of values greater than 0.5 and get label value from it
  kind=[tar_lab[ind] for ind,ele in enumerate(lab[9:len(lab)],9) if ele>=0.5]
  #store the target text which is combination of sentiment,when and kind data
  #add sos at start and eos at end of text
  tar_texts+=["sos "+" ".join([senti]+[when]+kind)+" eos"]

#only store unique words from the input and target word lists
inp_words = sorted(list(set(inp_words)))
num_inp_words = len(inp_words) 
num_tar_words = len(tar_lab)+2
 
#get the length of the input and the target texts which appears most frequently
max_inp_len = mode([len(i) for i in inp_texts])
max_tar_len = mode([len(i) for i in tar_texts])
 
print("number of input words : ",num_inp_words)
print("number of target words : ",num_tar_words)
print("maximum input length : ",max_inp_len)
print("maximum target length : ",max_tar_len)

#split the input and target text into 90:10 ratio or testing size of 10%=0.1.
x_train,x_test,y_train,y_test=train_test_split(inp_texts,tar_texts,test_size=0.1,random_state=42)

#Use all of the words from training input and output to train the tokenizer.
inp_tokenizer = Tokenizer()
inp_tokenizer.fit_on_texts(x_train)
tar_tokenizer = Tokenizer()
tar_tokenizer.fit_on_texts(y_train)
 
#convert text to an integer sequence where the integer represents the word index
x_train= inp_tokenizer.texts_to_sequences(x_train) 
y_train= tar_tokenizer.texts_to_sequences(y_train)

#If the length is less than the maximum length, pad the array with 0s. 
enc_inp_data= pad_sequences(x_train, maxlen=max_inp_len, padding='post',dtype="float32")
dec_data= pad_sequences(y_train, maxlen=max_tar_len, padding='post',dtype="float32")
 
#The last word, ie 'eos,' will not be included in the decoder input data.
dec_inp_data = dec_data[:,:-1]
 
#decoder target data will be one time step ahead as it will not include the first initial word i.e 'sos'
dec_tar_data = dec_data.reshape(len(dec_data),max_tar_len,1)[:,1:]

from keras import backend as K 
K.clear_session() 
latent_dim = 500
 
#create input object with the shape equal to the maximum number of input words
enc_inputs = Input(shape=(max_inp_len,)) 
enc_embedding = Embedding(num_inp_words+1, latent_dim)(enc_inputs)
 
 
#create 3 stacked LSTM layer
#1st LSTM layer keep only output  
enc_lstm1= LSTM(latent_dim, return_state=True, return_sequences=True) 
enc_outputs1, *_ = enc_lstm1(enc_embedding) 
 
#2nd LSTM layer keep only output 
enc_lstm2= LSTM(latent_dim, return_state=True, return_sequences=True) 
enc_outputs2, *_ = enc_lstm2(enc_outputs1) 
 
#3rd LSTM layer keep output as well as its states 
enc_lstm3= LSTM(latent_dim,return_sequences=True,return_state=True)
enc_outputs3 , state_h3 , state_c3= enc_lstm3(enc_outputs2)
 
#encoder states
enc_states= [state_h3, state_c3]

# Decoder. 
dec_inputs = Input(shape=(None,)) 
dec_emb_layer = Embedding(num_tar_words+1, latent_dim) 
dec_embedding = dec_emb_layer(dec_inputs) 
 
#initialize the LSTM layer of the decoder with the encoder's output states
dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
dec_outputs, *_ = dec_lstm(dec_embedding,initial_state=enc_states) 
 
#Attention layer
attention =Attention()
attn_out = attention([dec_outputs,enc_outputs3])
 
#Merge the attention output with the decoder outputs
merge=Concatenate(axis=-1, name='concat_layer1')([dec_outputs,attn_out])

#fully connected Dense layer for the output
dec_dense = Dense(num_tar_words+1, activation='softmax') 
dec_outputs = dec_dense(merge) 

#Model class and model summary
model = Model([enc_inputs, dec_inputs], dec_outputs) 
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#compile the model using RMSProp optimizer
model.compile( 
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 
 
#train the model with input and target data from encoder and decoder 
model.fit( 
    [enc_inp_data, dec_inp_data],dec_tar_data, 
    batch_size=500,epochs=10)

#Save model with the name as “s2s”
model.save("s2s")

#encoder inference
latent_dim=500

#load the model
model = models.load_model("s2s")

#construct an encoder model from the output of the 6th layer of LSTM 
enc_outputs,state_h_enc,state_c_enc = model.layers[6].output
enc_states=[state_h_enc,state_c_enc]
 
#add input data and state data from the layer
enc_model = Model(model.input[0],[enc_outputs]+enc_states)

# decoder inference model
#create Input object of hidden state and cell state for decoder
dec_state_input_h = Input(shape=(latent_dim,))
dec_state_input_c = Input(shape=(latent_dim,))
dec_hidden_state_input = Input(shape=(max_inp_len,latent_dim))
 
#Get the all the layers from the model
dec_inputs = model.input[1]
dec_emb_layer = model.layers[5]
dec_lstm = model.layers[7]
dec_embedding= dec_emb_layer(dec_inputs)
 
#add input and initialize the LSTM layer with decoder’s hidden and cell state
dec_outputs2, state_h2, state_c2 = dec_lstm(dec_embedding, initial_state=[dec_state_input_h,dec_state_input_c])
 
#Attention layer
attention = model.layers[8]
attn_out1 = attention([dec_outputs2,dec_hidden_state_input])

merge2 = Concatenate(axis=-1)([dec_outputs2, attn_out1])

#Dense layer for decoder output
dec_dense = model.layers[10]
dec_outputs2 = dec_dense(merge2)
 
# Finally define the Decoder model Class
dec_model = Model(
[dec_inputs] + [dec_hidden_state_input,dec_state_input_h,dec_state_input_c],
[dec_outputs2] + [state_h2, state_c2])

#create a dictionary with all indexes as key and respective target label as values
reverse_tar_word_index = tar_tokenizer.index_word
reverse_inp_word_index = inp_tokenizer.index_word
tar_word_index = tar_tokenizer.word_index
reverse_tar_word_index[0]=' '
 
def decode_sequence(inp_seq):
    #get the encoder outputs and states(hidden and cell) by passing the input sequence
    enc_out, enc_h, enc_c= enc_model.predict(inp_seq)
 
    #target sequence with starting initial word as 'sos'
    tar_seq = np.zeros((1, 1))
    tar_seq[0, 0] = tar_word_index['sos']
 
    #Stop the iteration if the iteration reaches end of the text
    stop_condition = False
    #merge every predicted word in decoded sentence
    decoded_sentence = ""
    while not stop_condition: 
      #get predicted output words, hidden and cell state for the model
      output_words, dec_h, dec_c= dec_model.predict([tar_seq] + [enc_out,enc_h, enc_c])

      #Using index get the word from the dictionary
      word_index = np.argmax(output_words[0, -1, :])
      text_word = reverse_tar_word_index[word_index]
      decoded_sentence += text_word +" "

      # Stop when we either hit max length or reach the terminal word i.e. eos.
      if text_word == "eos" or len(decoded_sentence) > max_tar_len:
          stop_condition = True

      #update target sequence with the current word index.
      tar_seq = np.zeros((1, 1))
      tar_seq[0, 0] = word_index
      enc_h, enc_c = dec_h, dec_c
 
    
    #return the decoded sentence string
    return decoded_sentence

#dict with key as label and value as target label value
lab_val=dict((i,v) for i,v in zip(tar_lab,tar_lab_val))

for i in range(0,20,3):
  #tokenize the x_test and convert into integer sequence
  inp_x= inp_tokenizer.texts_to_sequences([x_test[i]]) 
  #pad array of 0's 
  inp_x= pad_sequences(inp_x,  maxlen=max_inp_len, padding='post')
  #reshape the input x_test
  tag=decode_sequence(inp_x.reshape(1,max_inp_len)).replace('eos','')
  print("Tweet:",x_test[i])
  print("Predicted Hashtag:"," ".join(["#"+lab_val[i] for i in word_tokenize(tag)]))
  print("Actual Hashtag:"," ".join(["#"+lab_val[i] for i in y_test[i][4:-4].split(" ")]))
  print("\n")
