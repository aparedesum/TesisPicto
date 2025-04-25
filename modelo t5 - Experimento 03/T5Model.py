#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##Experimento 03
## En este experimento agregamos los identificadores de arasaac al modelo
## Formato identificados: pict_id

import os

os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


# In[1]:


#get_ipython().system('python --version')

import torch
print("Torch version",torch.__version__)
print("Device count", torch.cuda.device_count())
print("Cuda is available?", torch.cuda.is_available())

#get_ipython().system('pip install transformers==4.50.0')
#get_ipython().system('pip install -U datasets')
#get_ipython().system('pip install tensorboard')
#get_ipython().system('pip install sentencepiece')
#get_ipython().system('pip install accelerate')
#get_ipython().system('pip install evaluate==0.4.0')
#get_ipython().system('pip install bleu')
#get_ipython().system('pip install -U scikit-learn')
#get_ipython().system('pip install nltk datasets')
#get_ipython().system('pip install sacrebleu')


# In[2]:

import sacrebleu
import numpy as np
import nltk
import evaluate
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

#os.environ["CUDA_VISIBLE_DEVICES"]="2"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

print("Empty Cache", torch.cuda.empty_cache())
print("Current device", torch.cuda.current_device())

parser = argparse.ArgumentParser(description="Parameters for running training")

parser.add_argument("--training_data", type=str, required=True,
                        help="Ruta al archivo de datos de entrenamiento.")
parser.add_argument("--model_output", type=str, required=True,
                        help="Ruta para guardar el modelo entrenado.")
    
args = parser.parse_args()


import json
lista_pictogramas_ids=[]
with open("ids_only_id.txt", 'r', encoding='utf-8') as archivo:
  lineas = archivo.readlines()
  for linea in lineas:
    lista_pictogramas_ids.append(linea.strip())

print(len(lista_pictogramas_ids))


# In[4]:


dataset_train = load_dataset('json', data_files=args.training_data)['train']
dataset_test = load_dataset('json', data_files='test_data.json')['train']
dataset_valid = load_dataset('json', data_files='validation_data.json')['train']

# Mostrar estadísticas básicas
print(dataset_train)
print(dataset_test)
print(dataset_valid)


# In[7]:


model_checkpoint = "vgaraujov/t5-base-spanish" #"flax-community/spanish-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")

total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# Agregar nuevos tokens al tokenizer
existing_tokens = set(tokenizer.get_vocab().keys())
new_tokens = [token for token in lista_pictogramas_ids if token not in existing_tokens]
tokenizer.add_tokens(new_tokens)

model.resize_token_embeddings(len(tokenizer))

# Validar parámetros actualizados
print("After adding new tokens:")
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)
print(f"Sample IDs of new tokens: {new_token_ids[:10]}")

new_embeddings = model.get_input_embeddings().weight.data[new_token_ids]
print("Sample embeddings for new tokens:", new_embeddings[:5])

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Convertir los nuevos tokens a sus IDs
new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)

# Mostrar ejemplos de tokens agregados y sus IDs
for token, token_id in zip(new_tokens[:10], new_token_ids[:10]):
    print(f"Token: {token}, ID: {token_id}")


# In[8]:


model = model.to(torch.device('cuda'))
print("device", model.device)


# In[9]:


max_input_length = 40
max_target_length = 40

batch_size=32
metric ="bleu"
model_name = "t5-xgen-finetuned"
evaluation_strategy = "epoch"
save_strategy="epoch"
overwrite_output_dir=True
learning_rate=10e-5#10e-5
gradient_accumulation_steps=1
weight_decay=0.01
do_train=True
do_eval=True
save_total_limit=20
num_train_epochs=20
seed=42
predict_with_generate=True
fp16=True
metric_for_best_model="bleu"
load_best_model_at_end=True
generation_max_length = max_target_length
logging_strategy="epoch"
eval_accumulation_steps=1


# In[10]:


def preprocess_function(examples):
    inputs = ["translate: " + oracion for oracion in examples['oracion']]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")    
    labels = tokenizer(text_target=examples["traduccion"], max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# In[11]:


encoded_train = dataset_train.map(preprocess_function, batched=True)
encoded_test = dataset_test.map(preprocess_function, batched=True)
encoded_validation = dataset_valid.map(preprocess_function, batched=True)


# In[12]:


nltk.download("punkt", quiet=True)

bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load('chrf')

def compute_metrics_v2(eval_preds):
    print(f"metrics v2")
    predictions, labels = eval_preds

    # Reemplazar -100 con pad_token_id tanto en predicciones como en etiquetas
    pad_token_id = tokenizer.pad_token_id
    predictions = np.where(predictions != -100, predictions, pad_token_id)
    labels = np.where(labels != -100, labels, pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
     
    # Asegurarse de que BLEU reciba cadenas completas, no listas de palabras
    preds = [" ".join(pred.split()) for pred in decoded_preds]
    refs = [[" ".join(label.split())] for label in decoded_labels]  # Lista de listas para referencias
    
    print(f"bleu_preds {preds}")
    print(f"bleu_refs {refs}")
    
    # Calcular BLEU
    try:
        bleu_result = bleu_metric.compute(predictions=preds, references=refs)
        bleu_score = {"bleu": bleu_result["bleu"], "precisions": bleu_result["precisions"]}
    except Exception as e:
        print(f"Error al calcular BLEU: {e}")
        bleu_score = {"bleu": 0.0}

    # Calcular CHRF++
    try:
        chrf_result = chrf_metric.compute(predictions=preds, references=refs, word_order=2)
        #print(chrf_result)
        chrf_score = {"chrf": chrf_result["score"]}
    except Exception as e:
        print(f"Error al calcular CHRF++: {e}")
        chrf_score = {"chrf": 0.0}
        
    # Combinar resultados
    combined_results = {**bleu_score, **chrf_score}
    return combined_results


# In[13]:


from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
args = Seq2SeqTrainingArguments(
    output_dir=args.model_output,
    evaluation_strategy = evaluation_strategy,
    save_strategy = save_strategy,
    overwrite_output_dir = overwrite_output_dir,
    learning_rate = learning_rate,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size= batch_size,
    gradient_accumulation_steps= gradient_accumulation_steps,
    weight_decay= weight_decay,
    do_train= do_train,
    do_eval= do_eval,
    save_total_limit= save_total_limit,
    num_train_epochs= num_train_epochs,
    seed= seed,
    predict_with_generate= predict_with_generate,
    fp16= fp16,
    generation_max_length=generation_max_length,
    logging_strategy=logging_strategy,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    report_to="none",
    )

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=encoded_train,
    eval_dataset=encoded_validation,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_v2
)



# In[14]:


import numpy as np

import nltk
nltk.download('punkt')

trainer.train()

