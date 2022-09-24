# INQUISITIVE Dataset 
A dataset of about 20k questions that are elicited from readers as they naturally read through a document sentence by sentence.  Compared to existing datasets, INQUISITIVE questions target more towards high-level (semantic and discourse) comprehension of text. Because these questions are generated while the readers are pro-cessing the information, the questions directly communicate gaps between the reader’s and writer’s knowledge about the events described in the text, and are not necessarily answered in the document itself. This type of question reflects a real-world scenario: if one has questions during reading, some of them are answered by the text later on, the rest are not, but any of them would help further the reader’s understanding at the  particular point when they asked it.  This resource could enable question generation models to simulate human-like curiosity and cognitive processing, which may open up a new realm of applications. 

## Environment Setup
- The environment is saved in requirements.txt 
# Data preprocessing
## Question data generation
- Source inquisitive data - question.csv(rename the question.txt in the repo as it)
- Modify the source articles saving path in preprocess.py and build an empty folder to save train/val/test data
- Run <code>$preprocess.py prepare_question_generation=True</code>
- The generated data has a format of context(including the anchor sentence which contains the span) + highlighted span + question

## Span data generation
- Run <code>$python preprocess.py prepare_span_prediction=True</code>
- To fine tune bert for span prediction, run <code>$python fine_tune_squad.py</code>. It will save model as model_save_span_predict

# Train model
- Run <code>$python fine_tune_inquisitive.py</code> to fine-tune model to generate questions
- Run <code>$python fine_tune_squad.py</code> to fine-tune models to predict spans
- To generate span using prediction model, run <code>$python predict_span.py</code>. It will generate and save the result in result(please build an empty directory for result first)

# Generate Spans & Questions
- We use golden spans and predicted spans to generate questions
- To use predicted spans, we need to <code>$python preprocess.py --prepare_question_generation=True --use_predict_span=True</code> to generate desired input
- Then we can call <code>$python ./transformers/examples/text-generation/generate_for_fine_tune.py --model_type=gpt2 --model_name_or_path=<model path> --use_golden_span=True --golden_span_dir=<b>inquisitive data including golden span directory</b></code> to generate questions on golden span information
- We can call <code>$python /transformers/examples/text-generation/generate_for_fine_tune.py --model_type=gpt2 --model_name_or_path=<model path> --use_predict_span=True --predict_span_dir=<b>inquisitive data including predicted span directory path </b></code>to generate questions on predicted span information

**Citation:**
```
@InProceedings{ko2020inquisitive,
  author    = {Ko, Wei-Jen and Chen, Te-Yuan and Huang, Yiyan and Durrett, Greg and Li, Junyi Jessy},
  title     = {Inquisitive Question Generation for High Level Text Comprehension},
  booktitle = {Proceedings of EMNLP},
  year      = {2020},
}
```


# Data split
Validation: 1\~100, 1051\~1100

Test: 101\~150, 501\~550, 1101\~1150

The remaining articles are the training set.

# Article Sources
WSJ: 51\~259, 551\~590, 696\~900, 1446\~1491

Newsela: 1\~50, 260\~550, 901\~1050, 1492\~1500

AP: 591\~695, 1051\~1445

Since the articles are copyrighted, please send me an email to ask for the articles (wjko@outlook.com). Please obtain permission from newsela (https://newsela.com/data) first before emailing me.


# Note on span
The Span_Start_Position	and Span_End_Position are calculated by counting the white spaces. Note that in some sentences there are consecutive white spaces.
