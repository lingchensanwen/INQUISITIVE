from turtle import mode
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from fine_tune_squad import SquadDataset, prepare_data_for_squad, add_token_positions, ConvertIdstoQueryAnswer
from transformers import DistilBertTokenizerFast
import pandas as pd
import torch

tokenizer = DistilBertTokenizerFast.from_pretrained('./model_save_span_predict/')

model = AutoModelForQuestionAnswering.from_pretrained("./model_save_span_predict/")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# move model over to detected device
model.to(device)
model.eval()

def run_model_and_predict_span(test_loader, FileName):
    outputFile = open(FileName, 'w')
    outputFile.write('index\ttrue_span\tpredict_span\tanchor_sentence\n')
    acc = []
    
    # loop through batches
    for batch in test_loader:
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
            model_out=ConvertIdstoQueryAnswer(input_ids,start_pred,end_pred)
            for i in range(len(batch['input_ids'])):
                # if i >= len(model_out):
                #     text = str(0)+'\t'+span['text'][0]+'\t'+''+'\t'+anchor_sentences[0]+'\n'
                # else:
                text = str(i) + '\t'+span['text'][i]+'\t'+model_out[i]+'\t'+anchor_sentences[i]+'\n'
                outputFile.write(text)
    # calculate average accuracy in total
    acc = sum(acc)/len(acc)
    print("exact match acc is")
    print(acc)
        
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

    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
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
    
    print("len of test dataset: %d" % len(test_df))
    print("len of test encoding: %d" % len(test_contexts))
    # apply function to our data
    add_token_positions(train_encodings, train_answers)
    add_token_positions(val_encodings, val_answers)
    add_token_positions(test_encodings, test_answers)

    # build datasets for both our training and validation sets
    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)
    test_dataset = SquadDataset(test_encodings)
    
    # initialize validation set data loader
    train_loader = DataLoader(train_dataset, batch_size=16)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    #predict span
    run_model_and_predict_span(train_loader, "result/predict_span_train.csv")
    run_model_and_predict_span(val_loader, "result/predict_span_val.csv")
    run_model_and_predict_span(test_loader, "result/predict_span_test.csv")
        


if __name__ == "__main__":
    main()