import os
import time
import datetime

import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, TrainingArguments
from transformers import AdamW, get_linear_schedule_with_warmup

import nltk
 
model_path = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_path)
configuration = GPT2Config.from_pretrained(model_path, output_hidden_states=False)
batch_size = 2
max_length = 1024 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.

def load_file_to_array(file_path):
    file  = open(file_path, 'r')
    arr = []
    for line in file:
        arr.append(line)
        # get rough token count distribution
        tokens = nltk.word_tokenize(line)
    return arr

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

class GPT2Dataset(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []
    for txt in txt_list:
      encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx] 


def load_dataset(train_path,test_path,tokenizer):
    train_arr = load_file_to_array(train_path)
    test_arr = load_file_to_array(test_path)

    train_dataset = GPT2Dataset(train_arr, tokenizer, max_length=max_length)
    test_dataset = GPT2Dataset(test_arr, tokenizer, max_length=max_length)

    train_size = len(train_dataset)
    test_size = len(test_dataset)

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(test_size))

    train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

    test_dataloader = DataLoader(
            test_dataset, # The validation samples.
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

    return train_dataloader, test_dataloader

def main():
    # Load the GPT tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained(model_path, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium

    # loading data
    # for inquisitive
    train_path = "inquisitive_source_data/base_input_to_train.txt"
    test_path = "inquisitive_source_data/base_input_to_val.txt"
    # for dcqa complete
    # train_path = "dcqa_source_data/input_to_train.txt"
    # test_path = "dcqa_source_data/input_to_val.txt"
    # for dcqa part
    # train_path = "elab_questions/input_to_train.txt"
    # test_path = "elab_questions/input_to_test.txt"

    train_dataset,test_dataset = load_dataset(train_path,test_path,tokenizer)
    
    
    # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
    # otherwise the tokenizer and model tensors won't match up
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))

    # Tell pytorch to run this model on the GPU.
    device = torch.device("cuda")
    model.cuda()

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42 
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # some parameters I cooked up that work reasonably well

    epochs = 7
    learning_rate = 2e-5 #5e-4
    warmup_steps = 1e2
    epsilon = 5e-5
    sample_every = 100 # this produces sample output every 100 steps

    optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )

    # Total number of training steps is [number of batches] x [number of epochs]. 
    total_steps = len(train_dataset) * epochs
    # Create the learning rate scheduler.
    # This changes the learning rate as the training loop progresses
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = warmup_steps, 
                                                num_training_steps = total_steps)
    

    total_t0 = time.time()

    training_stats = []

    model = model.to(device)

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataset):

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            model.zero_grad()        

            outputs = model(  b_input_ids,
                            labels=b_labels, 
                            attention_mask = b_masks,
                            token_type_ids=None
                            )

            loss = outputs[0]  

            batch_loss = loss.item()
            total_train_loss += batch_loss

            # Get sample every x batches.
            if step % sample_every == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataset), batch_loss, elapsed))

                model.eval()

                sample_outputs = model.generate(
                                        bos_token_id=random.randint(1,30000),
                                        do_sample=True,   
                                        top_k=50, 
                                        max_length =max_length,
                                        top_p=0.95, 
                                        num_return_sequences=1
                                    )
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
                
                model.train()

            loss.backward()

            optimizer.step()

            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataset)       
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in test_dataset:
            
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)
            
            with torch.no_grad():        

                outputs  = model(b_input_ids, 
    #                            token_type_ids=None, 
                                attention_mask = b_masks,
                                labels=b_labels)
            
                loss = outputs[0]  
                
            batch_loss = loss.item()
            total_eval_loss += batch_loss        

        avg_val_loss = total_eval_loss / len(test_dataset)
        
        validation_time = format_time(time.time() - t0)    

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    

    # Display floats with two decimal places.
    pd.set_option('precision', 2)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    # A hack to force the column headers to wrap.
    #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

    # Display the table.
    print(df_stats)
    output_dir = 'models/inquisitive_golden'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()