"""
Train FinTextSim with BatchHardTripletLoss
"""

#########################
#from __future__ import annotations
import torch
import pickle
from tqdm import tqdm
import sys
import os
from collections import Counter
from sklearn.utils import resample
from sentence_transformers.evaluation import TripletEvaluator, SequentialEvaluator
from sentence_transformers.losses import TripletLoss, BatchAllTripletLoss, BatchHardSoftMarginTripletLoss, BatchHardTripletLoss, BatchSemiHardTripletLoss
from sentence_transformers.losses import TripletDistanceMetric
from sentence_transformers.losses import BatchHardTripletLossDistanceFunction
from sentence_transformers.training_args import BatchSamplers

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments

)

############Define the configuration of training
results_dict = {}
results_dict["config"] = {
    "n_epochs" : 5,
    "loss": "BatchHardTripletLoss",
    "margin": 5,
    "batch_size": 200,
    "eval_steps": 2000,
    "lr_scheduler": "linear",
    "learning_rate": 2e-5,
}
##############
results_dict["training_results"] = {}   #initialize training results

os.chdir("/mnt/sdb1/home/simonj") #set working directory


model_dir = "paper1/Fintextsim_Models/fintextsim_2016_2023_triplet_temporal"
embedding_dir = "paper1/Results/Embeddings/test_data/fintextsim_2016_2023_triplet_temporal.pkl"

n_epochs = results_dict["config"].get("n_epochs")
transition_steps = results_dict["config"].get("transition_steps")

batch_size = results_dict["config"].get("batch_size")
eval_steps = results_dict["config"].get("eval_steps")
lr_scheduler = results_dict["config"].get("lr_scheduler")
learning_rate = results_dict["config"].get("learning_rate")
print(f"Learning rate scheduler: {lr_scheduler}\n learning rate: {learning_rate}")

#----------import test and train sets
file_loc = "paper1/Data/train_test_sets/train_test_sets_2016_2023_masked.pkl"

with open(file_loc, "rb") as file:
    data = pickle.load(file)

train_triplets = data["train_triplets"]
test_triplets = data["test_triplets"]
train_dataset_triplets = data["train_dataset_triplets"]
train_dataset = data["train_dataset"]
train_dataset_masked = data["train_dataset_masked"]
test_dataset = data["test_dataset"]
test_dataset_masked = data["test_dataset_masked"]
test_sentences = data["test_sentences"] #unmasked test sentences
test_topics = data["test_topics"]
labeled_sentences = data["labeled_sentences"]
labeled_dataset_topics = data["labeled_dataset_topics"]
masked_test_set = data["masked_test_set"]

#extract masked sentences and corresponding labels
test_sentences_masked = [sent for sent, label in masked_test_set]
test_labels_masked = [label for sent, label in masked_test_set]

#-----------select the base model
from sentence_transformers import models

bert_transformer = models.Transformer('bert-base-uncased', do_lower_case = True) #max_seq_length = 256

bert_pooler = models.Pooling(
    bert_transformer.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True #mean pooling of tokens
)

## normalization layer
normalizer = models.Normalize()

# Combine them into a SentenceTransformer model
bert_base_model = SentenceTransformer(modules=[bert_transformer, bert_pooler, normalizer])

modern_bert = models.Transformer("answerdotai/ModernBERT-base", do_lower_case = True)

modern_bert_pooler = models.Pooling(
    modern_bert.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)



# Combine them into a SentenceTransformer model
modern_bert_base_model = SentenceTransformer(modules=[modern_bert, modern_bert_pooler, normalizer])

finbert = models.Transformer('yiyanghkust/finbert-pretrain', do_lower_case = True)


finbert_pooler = models.Pooling(
    finbert.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)


# Combine them into a SentenceTransformer model
finbert_base_model = SentenceTransformer(modules=[finbert, finbert_pooler, normalizer])


base_models = [
    modern_bert_base_model, 
    #finbert_base_model, 
    #bert_base_model
]
base_model_names = [
    "modern_bert", 
    #"finbert", 
    #"BERT"
]

#------------check GPU availability

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    
    # Ask the user if they want to continue or stop
    user_choice = input("GPU is not available. Do you want to continue using the CPU? (yes/no): ").strip().lower()
    
    if user_choice != 'yes':
        print("Ending the script")
        sys.exit()
    print("Using CPU")



#--------------train the model
from fintextsim.AdaptiveCircleLoss import CircleLossText
from fintextsim.ClasswiseEvaluator import ClasswiseEvaluator

for i, model in tqdm(enumerate(base_models), desc = "Model progress"):
    model_name = base_model_names[i]
    print(f"Training model: {model_name}")
    print(f"Batch size: {batch_size}; \nepochs: {n_epochs}")
    
    train_loss_1 = BatchHardTripletLoss(model = model, margin = results_dict["config"].get("margin"))

    model_loc = model_dir + "_" + model_name
    
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=model_loc,
        # Optional training parameters:
        num_train_epochs=n_epochs,
        per_device_train_batch_size= batch_size,
        per_device_eval_batch_size= batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type = lr_scheduler,
        warmup_ratio=0.1,
        #warmup_steps = 100,
        #fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        #bf16=False,  # Set to True if you have a GPU that supports BF16
        #train_sampler = sampler,
        #batch_sampler = BatchSamplers.GROUP_BY_LABEL,  # losses that use "in-batch negatives" benefit from no duplicates; GROUP_BY_LABEL ensures that each batch has 2+ samples from the same label
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=eval_steps,
        #save_strategy="steps",
        save_steps=results_dict["config"].get("eval_steps"),
        #save_total_limit=2,
        #logging_steps=100,
        #run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
        load_best_model_at_end = True,
    )


    classwise_evaluator = ClasswiseEvaluator(
        texts = test_sentences,
        labels = test_topics,
        show_progress_bar = True,
        name = "Classwise evaluator",
        batch_size = batch_size
    )


    # Now use the custom trainer for training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        loss= train_loss_1,
        evaluator=classwise_evaluator

    )
    print(f"------------------------Model training ({model})--------------------------------\n")
    print(f"Batch size: {batch_size}; \nmargin: {results_dict["config"].get("margin")}\nepochs: {n_epochs}")

    trainer.train()
    evaluation = trainer.evaluate()
    print(evaluation)


    # Save the trained model
    model.save_pretrained(model_loc)



    #-----------------create and save embeddings for comparing the sentence transformer models
    print(f"---------------Create test embeddings for {model_name}----------------------")
    test_embeddings_fintextsim = model.encode(test_sentences, show_progress_bar = True)
    test_embeddings_fintextsim_masked = model.encode(test_sentences_masked, show_progress_bar = True)

    #------------------create embeddings for the full labeled dataset
    print(f"----------------Create embeddings for the full labeled dataset with {model_name}-------------")
    labeled_dataset_embeddings = model.encode(labeled_sentences, show_progress_bar = True)

    #---------------reduce dimensionality of the test embeddings
    #print(f"--------------------Reduce dimensionality of test embeddings ({model_name})-----------------")
   
    #------------save results

    
    results_dict["training_results"][model_name] = {
        "test_embeddings_fintextsim": test_embeddings_fintextsim,
        "test_embeddings_fintextsim_masked": test_embeddings_fintextsim_masked,
        "labeled_dataset_embeddings": labeled_dataset_embeddings,
        "evaluation": evaluation,
        "test_topics": test_topics,
        "labeled_dataset_topics": labeled_dataset_topics,
        "test_sentences": test_sentences,
        "test_sentences_masked": test_sentences_masked
    }
    with open (embedding_dir, "wb") as file:
        pickle.dump(results_dict, file)

#----------------create test embeddings for AM
#define the base model
baseline_model1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device = device)
baseline_model2 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device = device)
test_embeddings_ots1 = baseline_model1.encode(test_sentences, show_progress_bar = True)
test_embeddings_ots2 = baseline_model2.encode(test_sentences, show_progress_bar = True)

#---baseline model - financial sentiment
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
baseline_model3 = AutoModelForSequenceClassification.from_pretrained(model_name)

#add model name for saving later on --> no backslashes, etc.
model_name = "distil_roberta"

from bertopic_models.utils_bertopic import generate_embeddings_classifier

test_embeddings_ots3 = generate_embeddings_classifier(baseline_model3, tokenizer, test_sentences, "roberta")

results_dict["basics"] = {
    "test_embeddings_ots1": test_embeddings_ots1,
    "test_embeddings_ots2": test_embeddings_ots2,
    "test_embeddings_ots3": test_embeddings_ots3,
    "test_topics": test_topics,
    "test_sentences": test_sentences,
    "labeled_dataset_topics": labeled_dataset_topics,
}


with open (embedding_dir, "wb") as file:
    pickle.dump(results_dict, file)

print(f"Relevant data saved to {embedding_dir}")
