import os
import pandas as pd
import argparse

inquisitive_articles_path = "/work/07144/yw23374/ls6/DCQA-INQUISITIVE/examples/article_examples/inquisitive_docs/article_inquisitive/"

#read source df for fine-tune inquisitive on inquisitive data
def read_train_test_val_for_inquisitive(inqusitive_data_dir):
    inquisitive_df = pd.read_csv('questions.csv', sep='\\t', on_bad_lines='skip')
    val_df = inquisitive_df[(inquisitive_df["Article_Id"]<=100) | ((inquisitive_df["Article_Id"]>=1051) & (inquisitive_df["Article_Id"]<=1100))]
    test_df = inquisitive_df[((inquisitive_df["Article_Id"]>=101) & (inquisitive_df["Article_Id"]<=105)) | ((inquisitive_df["Article_Id"]>=501) & (inquisitive_df["Article_Id"]<=550)) |
    ((inquisitive_df["Article_Id"]>=1101) & (inquisitive_df["Article_Id"]<=1150))]
    train_df = inquisitive_df[(~inquisitive_df["Article_Id"].isin(test_df["Article_Id"])) & (~inquisitive_df["Article_Id"].isin(val_df["Article_Id"]))]
    train_df = train_df.reset_index()
    test_df = test_df.reset_index()
    val_df = val_df.reset_index()
    print("now reading train test val for inquisitive base")
    print("train num: "+str(len(train_df)))
    print("val num: "+str(len(val_df)))
    print("test num: "+str(len(test_df)))
    return train_df, val_df, test_df

def read_train_test_val_for_fine_tune(source_data_dir):
    train_df = pd.read_csv(source_data_dir+'/train.csv', sep='\\t', on_bad_lines='skip')
    test_df = pd.read_csv(source_data_dir+'/test.csv', sep='\\t', on_bad_lines='skip')
    val_df = pd.read_csv(source_data_dir+'/val.csv', sep='\\t', on_bad_lines='skip')
    print("now reading train test val for elab fine-tune")
    print("train num: "+str(len(train_df)))
    print("val num: "+str(len(val_df)))
    print("test num: "+str(len(test_df)))
    return train_df, val_df, test_df

def read_train_test_val_from_predict_span(span_prediction_dir):
    train_df = pd.read_csv(span_prediction_dir+'/predict_span_train.csv', sep='\\t', on_bad_lines='skip')
    test_df = pd.read_csv(span_prediction_dir+'/predict_span_test.csv', sep='\\t', on_bad_lines='skip')
    val_df = pd.read_csv(span_prediction_dir+'/predict_span_val.csv', sep='\\t', on_bad_lines='skip')
    print("now reading train test val from span predicted")
    print("train num: "+str(len(train_df)))
    print("val num: "+str(len(val_df)))
    print("test num: "+str(len(test_df)))
    return train_df, val_df, test_df

def read_inquisitive_article_content(file_num):
    articleContent = []
    file_num = str(file_num).zfill(4) #append zeros to file name
    article = open(inquisitive_articles_path+str(file_num)+".txt",'r')
    for line in article:
        #handle each sentence
        try:  
            # print("line: "+line)
            # sentence="".join(line.split('')[1:])
            sentence = line.strip()
            sentence = " ".join(sentence.split(' ')[1:])
            # print("line: "+sentence)
            articleContent.append(sentence)

        except:
            articleContent.append('')
    return articleContent

#generate train/val/test file for inquisitive model
def generate_text_file_for_inquisitive_base(question_df, filename):
    outputFile = open(filename, 'w')
    for i in range(len(question_df)):
        sentence_num = question_df.loc[i, "Sentence_Id"]
        file_num = question_df.loc[i, "Article_Id"]
        highlighted_span = question_df.loc[i, "Span"]
        essay = read_inquisitive_article_content(file_num)
        context = "".join(essay[:sentence_num+1])#include all sentences before and including sentence posted
        question = question_df.loc[i, "Question"]
        text = context +'<@ '+ str(highlighted_span) + ' (> || ' + question +"\n"
        outputFile.write(text)
        
        
def generate_text_file_for_inquisitive_fine_tune_on_predict_span(question_df, predict_span_df, filename):
    outputFile = open(filename, 'w')
    print("question df len is %d" % len(question_df))
    print("pred df len is %d" % len(predict_span_df))
    for i in range(min(len(question_df),len(predict_span_df))):
        file_num = question_df.loc[i, "Article_Id"]
        anchor_num = question_df.loc[i, "Sentence_Id"]
        essay = read_inquisitive_article_content(file_num)
        context = "".join(essay[:anchor_num+1])#include all sentences before and including anchor sentencor
        question = question_df.loc[i, "Question"]
        predict_span = predict_span_df.loc[i, "predict_span"]
        text = context +'<@ '+ str(predict_span) + ' (> || ' + question +"\n"
        outputFile.write(text)
        
def clean_invalid_span(span, anchor_sentence):
    span = " ".join(span.split())
    if ' ,' in span:
        span = span.replace(' ,', ',')
        if span in anchor_sentence:
            return span
    if ' - ' in span:
        span = span.replace(' - ', '-')
        if span in anchor_sentence:
            return span
    if '$ ' in span: 
        span = span.replace('$ ', '$')
        if span in anchor_sentence:
            return span
    if '% ' in span: 
        span = span.replace('% ', '%')
        if span in anchor_sentence:
            return span
    if ' %' in span: 
        span = span.replace(' %', '%')
        if span in anchor_sentence:
            return span
    if '\"' in span:
        span = span.replace('\"', '\'')   
        if span in anchor_sentence:
            return span 
    if ' , ' in span:
        span = span.replace(' , ', ',')
        if span in anchor_sentence:
            return span       
    if ' \' ' in span:
        span = span.replace(' \' ', '\'')
        if span in anchor_sentence:
            return span
    if '\' ' in span:
        span = span.replace('\' ', '\'')
        if span in anchor_sentence:
            return span
    if ' \'' in span:
        span = span.replace(' \'', '\'')
        if span in anchor_sentence:
            return span
    if ' .' in span:
        span = span.replace(' .', '.')
        if span in anchor_sentence:
            return span 
    if '. ' in span:
        span = span.replace('. ', '.')
        if span in anchor_sentence:
            return span 
    if 'U.S.' in span:
        span = span.replace('U.S.', 'U.S. ')
        if span in anchor_sentence:
            return span     
    return span

#generate files for span prediction        
def generate_text_file_for_span_predict(question_df, filename):
    outputFile = open(filename, 'w')
    context_arr = []
    question_arr = []
    answer_arr = []
    #calculate span start pos and span end pos in white space
    def get_span_start_end_pos(context, span):
        span = str(span)
        span_start = context[:context.index(span)].count(' ')
        span_len = span.count(' ')
        span_end = span_len+span_start+1
        return span_start, span_end

    #calculate span start pos and span end pos in char
    def get_span_start_end_pos_in_char(context, span):
        span_start = context.index(span)
        spand_end = int(span_start)+len(span)
        return span_start, spand_end
    #write span info to a file
    outputFile.write('file_num\tcontext_with_anchor\tcontext\tquestion\tanchor_sentence\tspan\tspan_start\tspan_end\t  \
    span_start_idx\tspan_end_idx\tspan_start_in_anchor_idx\tspan_end_in_anchor_idx\n')
    #write only useful info into squad file so that can use it to fine tune squad

    for i in range(len(question_df)):
        file_num = question_df.loc[i, "Article_Id"]
        anchor_num = question_df.loc[i, "Sentence_Id"]
        question = question_df.loc[i, "Question"]
        highlighted_span = str(question_df.loc[i, "Span"])
        essay = read_inquisitive_article_content(file_num)
        context = " ".join(essay[:anchor_num])#include all sentences before and not include anchor sentencor
        context_with_anchor = " ".join(essay[:anchor_num+1])
        anchor_sentence =  essay[anchor_num-1]

        if highlighted_span not in anchor_sentence:
            alter_highlighted_span = clean_invalid_span(highlighted_span, anchor_sentence)
            if alter_highlighted_span not in anchor_sentence:
                # print("highlighted_span is not labeled correctly(not inside anchor sentence)")
                # print("old span:" + highlighted_span)
                # print("alter span: " + alter_highlighted_span)
                # print("real anchor: "+anchor_sentence)
                continue
            else:
                highlighted_span = alter_highlighted_span

        span_start, span_end = get_span_start_end_pos(anchor_sentence, highlighted_span)
        span_start_idx, span_end_idx = get_span_start_end_pos_in_char(context_with_anchor, highlighted_span)
        span_start_in_anchor_idx, span_end_in_anchor_idx = get_span_start_end_pos_in_char(anchor_sentence, highlighted_span)
        text = str(file_num) +'\t'+context_with_anchor +'\t'+context+'\t'+question+'\t'+anchor_sentence+'\t'+str(highlighted_span)+'\t'+str(span_start)+'\t'+str(span_end) \
        +'\t'+str(span_start_idx)+'\t'+str(span_end_idx)+'\t'+str(span_start_in_anchor_idx)+'\t'+str(span_end_in_anchor_idx)+'\n'
        outputFile.write(text)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="questions.csv", help="data dir for inquisitive data")
    parser.add_argument('--prepare_question_generation', type=bool, default=False, help="choose to prepare data for question generation or not")
    parser.add_argument('--question_generation_dir', type=str, default = "inquisitive_source_data/", help="where to put data for question generation")
    parser.add_argument('--prepare_span_prediction', default=False, type=bool, help="choose to prepare data for span prediction or not")
    parser.add_argument('--tune_span_prediction_dir', default="span_predict/", type=str, help="where to put data for span prediction")
    parser.add_argument('--use_predict_span', type=bool, default=False, help="use predicted span to generate question")
    parser.add_argument('--predicted_span_dir', default="result/", type=str, help="where are the predicted span")
    args = parser.parse_args()
    print("args is")
    print(args)
    
    #preprocess data for fine-tune model on golden spans
    if args.prepare_question_generation == True:
        train_df, val_df, test_df = read_train_test_val_for_inquisitive(args.data_dir)
        generate_text_file_for_inquisitive_base(train_df, args.question_generation_dir+"/base_input_to_train.txt")
        generate_text_file_for_inquisitive_base(val_df, args.question_generation_dir+"/base_input_to_val.txt")
        generate_text_file_for_inquisitive_base(test_df, args.question_generation_dir+"/base_input_to_test.txt")

    #preprocess data for fine-tune model to predict spans
    if args.prepare_span_prediction == True:
        train_df, val_df, test_df = read_train_test_val_for_inquisitive(args.data_dir)
        generate_text_file_for_span_predict(train_df, args.tune_span_prediction_dir+"/train_span.csv")
        generate_text_file_for_span_predict(val_df, args.tune_span_prediction_dir+"/val_span.csv")
        generate_text_file_for_span_predict(test_df, args.tune_span_prediction_dir+"/test_span.csv")

    #preprocess data for fine-tune model on predicted spans
    if args.use_predict_span == True:      
        train_df, val_df, test_df = read_train_test_val_for_inquisitive(args.data_dir)
        train_df = train_df.reset_index()
        val_df = val_df.reset_index()
        test_df = test_df.reset_index()
        pred_span_train_df, pred_span_val_df, pred_span_test_df = read_train_test_val_from_predict_span(args.predicted_span_dir)
        # pred_span_train_df = pred_span_train_df.reset_index()
        # pred_span_val_df = pred_span_val_df.reset_index()
        # pred_span_test_df = pred_span_test_df.reset_index()
        generate_text_file_for_inquisitive_fine_tune_on_predict_span(train_df, pred_span_train_df, args.question_generation_dir+"/pred_span/input_to_train.txt")
        generate_text_file_for_inquisitive_fine_tune_on_predict_span(val_df, pred_span_val_df, args.question_generation_dir+"/pred_span/input_to_val.txt")
        generate_text_file_for_inquisitive_fine_tune_on_predict_span(test_df, pred_span_test_df, args.question_generation_dir+"/pred_span/input_to_test.txt")