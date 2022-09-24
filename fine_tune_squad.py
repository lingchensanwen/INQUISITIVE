import pandas as pd
import os
import torch
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

# initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def ConvertIdstoQueryAnswer(inputIds,startId,endId):
    answer=''
    results=[]
    for i in range(len(inputIds)):
        token=tokenizer.convert_ids_to_tokens(inputIds[i])      
        if endId[i] >= startId[i]:
            answer = token[startId[i]]
            for i in range(startId[i]+1, endId[i]+1):
                if token[i][0:2] == "##":               
                    answer += token[i][2:]                    
                else:
                    answer += " " + token[i]          
            results.append(answer)   
        else:
            results.append("None")  

    return results

def prepare_data_for_squad(question_df):
    # question_df = question_df.dropna(how='any')
    # question_df = question_df.reset_index()
    question_df.fillna('', inplace=True) 
    def compose_answer_info(question_df):
        answers = []
        for i in range(len(question_df)):
            span = question_df.loc[i, "span"]
            span_start_idx = question_df.loc[i, "span_start_in_anchor_idx"]
            span_end_idx = question_df.loc[i, "span_end_in_anchor_idx"]
            answer = {'text': span, 'answer_start': span_start_idx, 'answer_end': int(span_end_idx)}
            answers.append(answer)
        return answers


    target_setences = []
    previous_sentences = []
    answers = []
    target_setences = question_df["anchor_sentence"].tolist()
    previous_sentences = question_df["context"].tolist()
    answers = compose_answer_info(question_df)
    return target_setences, previous_sentences, answers

def add_token_positions(encodings, answers):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        # append start/end token position using char_to_token method
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        # end position cannot be found, char_to_token found space, so shift position until found
        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1
    # update our encodings object with the new token-based start/end positions
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        res = {}
        for key, val in self.encodings.items():
            if key not in ["anchor_sentence", "context", "true_span"]:
                res.update({key: torch.tensor(val[idx])})
            else:
                res.update({key: val[idx]})
        return res
        #return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def get_anchor_sentence(self, idx):
        self.encodings["anchor_sentence"][idx]
    def get_context(self, idx):
        self.encodings["context"][idx]
    def get_true_span(self, idx):
        self.encodings["true_span"][idx]
    def __len__(self):
        return len(self.encodings.input_ids)

def main():
    #preprocess data
    train_df = pd.read_csv('span_predict/train_span.csv', sep='\\t', on_bad_lines='skip')
    test_df = pd.read_csv('span_predict/test_span.csv', sep='\\t', on_bad_lines='skip')
    val_df = pd.read_csv('span_predict/val_span.csv', sep='\\t', on_bad_lines='skip')
    train_contexts, train_questions, train_answers = prepare_data_for_squad(train_df)
    test_contexts, test_questions, test_answers = prepare_data_for_squad(test_df)
    val_contexts, val_questions, val_answers = prepare_data_for_squad(val_df)
    print(train_contexts[:2])
    print(train_questions[:2])
    print(train_answers[:2])

    def encode(input):
        return tokenizer(input["anchor_sentence"], input["context"], truncation=True, padding=True)
    # tokenize
    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True, )
    test_encodings = tokenizer(test_contexts, test_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)
    
    train_encodings.update({
        "anchor_sentence": train_contexts,
        "context": train_questions,
        "true_span": train_answers
    })
    test_encodings.update({
        "anchor_sentence": test_contexts,
        "context": test_questions,
        "true_span": test_answers
    })
    val_encodings.update({
        "anchor_sentence": val_contexts,
        "context": val_questions,
        "true_span": val_answers
    })
    
    print(val_encodings[0])

    # apply function to our data
    add_token_positions(train_encodings, train_answers)
    add_token_positions(val_encodings, val_answers)
    add_token_positions(test_encodings, test_answers)
    
    print(val_encodings[0])

    # build datasets for both our training and validation sets
    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)
    test_dataset = SquadDataset(test_encodings)

    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

    #------fine-tune training------- 
    # setup GPU/CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # move model over to detected device
    model.to(device)
    # activate training mode of model
    model.train()
    # initialize adam optimizer with weight decay (reduces chance of overfitting)
    optim = AdamW(model.parameters(), lr=5e-5)

    # initialize data loader for training data
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    for epoch in range(3):
        # set model to train mode
        model.train()
        # setup loop (we use tqdm for the progress bar)
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all the tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            # train model on batch and return outputs (incl. loss)
            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)
            # extract loss
            loss = outputs[0]
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())


        #--------------eval model result-----------------
        # switch model out of training mode
        model.eval()
        # initialize validation set data loader
        val_loader = DataLoader(val_dataset, batch_size=16)
        # initialize list to store accuracies
        acc = []
        # loop through batches
        for batch in val_loader:
            # we don't need to calculate gradients as we're not training
            with torch.no_grad():
                # for i in range(16):
                #     print(batch["anchor_sentence"][0][i])
                # pull batched items from loader
                anchor_sentences = batch["anchor_sentence"]
                span = batch["true_span"]
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                # we will use true positions for accuracy calc
                start_true = batch['start_positions'].to(device)
                end_true = batch['end_positions'].to(device)
                # make predictions
                outputs = model(input_ids, attention_mask=attention_mask)
                # pull prediction tensors out and argmax to get predicted tokens
                start_pred = torch.argmax(outputs['start_logits'], dim=1)
                end_pred = torch.argmax(outputs['end_logits'], dim=1)
                # calculate accuracy for both and append to accuracy list
                acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
                acc.append(((end_pred == end_true).sum()/len(end_pred)).item())

        # calculate average accuracy in total
        acc = sum(acc)/len(acc)
        print("exact match acc is")
        print(acc)
        #---------------saving model-------------------
        output_dir = './model_save_span_predict/'
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