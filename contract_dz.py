import numpy
import pandas as pd
import textract
import os
import torch
onlyfiles='Sample 1 - Garratt Callahan SPA 2020-2022.docx'

###############
def preprocess_string(txt):
    txt=txt.replace('\t',' ')
    txt=txt.replace('\n',' ')
    txt=txt.replace('“',' ')
    txt=txt.replace('”',' ')
    txt=txt.replace('supplier        salesforce.com','')
    txt=txt.replace('supplier       salesforce.com','')
    txt=txt.replace('supplier      salesforce.com','')
    txt=txt.replace('supplier     salesforce.com','')
    txt=txt.replace('supplier    salesforce.com','')
    txt=txt.replace('supplier   salesforce.com','')
    txt=txt.replace('supplier  salesforce.com','')
    txt=txt.replace('supplier salesforce.com','')
    #txt=txt.replace('SUPPLIER ','')
    #txt=txt.replace('supplier ','')
    txt=txt.replace('By:','')
    txt=txt.replace('by:','')
    txt=txt.replace('Name:','')
    txt=txt.replace('name:','')
    txt=txt.replace('Title:','')
    txt=txt.replace('title:','')
    txt=txt.replace('\\si2\\','')
    txt=txt.replace('\\si1\\','')
    txt=txt.replace('\\ti1\\','')
    txt=txt.replace('\\ti2\\','')
    txt=txt.replace('\\na2\\','')
    txt=txt.replace('\\na1\\','')
    txt=txt.replace('\\si2\\','')
    txt=txt.replace('\\ds1\\','')
    txt=txt.replace('\\ds2\\','')
    txt=txt.replace('(','')
    txt.replace(')','')
    return txt
###############
def get_dataframe_chunk(text, file_name):
    line_bucket = text
    start_len = 0
    sentence_len = 1500
    counter = sentence_len
    tect_bucket = []
    slider_len = 50
    flag_len = 1
    # print(counter)
    # print(len(ans))

    text_bucket = []
    data = {'text': text_bucket, 'File_Name': file_name}
    while (flag_len != 0):
        if ((len(line_bucket) - counter) > 0):
            line_new = line_bucket[start_len:counter]
            text_bucket.append(line_new)
            flag_len = 1

        else:
            line_new = line_bucket[start_len:len(line_bucket)]
            text_bucket.append(line_new)
            flag_len = 0

        start_len = counter - slider_len
        counter = counter + sentence_len - slider_len
    # print(text_bucket)
    df_temp = pd.DataFrame(data)

    return (df_temp)
#########

df_final = pd.DataFrame()

text_dump=""
text_dump_str=""
text_dump_str_lower=""
text_dump= textract.process(onlyfiles)
tmp_text_dump=text_dump.decode('utf-8')
text_dump_str_lower=tmp_text_dump.lower()
text_dump_str_final=preprocess_string(text_dump_str_lower)
df_temp=pd.DataFrame()
df_temp = get_dataframe_chunk(text_dump_str_final,onlyfiles)
df_final=pd.concat([df_final,df_temp])

df_final.to_csv("Contract_Digitization.csv")
data =pd.read_csv("Contract_Digitization.csv")
data.drop('Unnamed: 0',axis=1,inplace=True)
#data.head()
data_temp=data.copy()

####
from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
###
def get_token_length(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
        '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)
    return(len(input_ids))

######
question="What is the name of the Agreement Title?"
token_len=[]
token_len=[get_token_length(question,data_temp.text[row]) for row in range(0,data_temp.shape[0])]
data_temp['Token Length']=token_len
#####
def bert_prediction_1(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
    # print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example through the model.
    outputs = model(torch.tensor([input_ids]),  # The tokens representing our input text.
                    token_type_ids=torch.tensor([segment_ids]),
                    # The segment IDs to differentiate question from answer_text
                    return_dict=True)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

            # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

        # print('Answer: "' + answer + '"')
    return (answer)

########
def contract_digitization(quest, type_of_quest="a"):
    output_con_dig = []
    answer_Flag_List = []
    k = data_temp.shape[0]
    for row in range(0, k):
        text = preprocess_string(data_temp.text[row])
        # print(text)
        pred = bert_prediction_1(question, text)
        answer_Flag_List.append(pred)
    data_temp['Answer'] = answer_Flag_List
    data_temp_filtered = data_temp[~data_temp['Answer'].str.contains('CLS')]

    output_con_dig.append(data_temp_filtered.Answer.iloc[0])
    output_con_dig.append(data_temp_filtered.text.iloc[0])
    output_con_dig.append(question)
    return output_con_dig
output=contract_digitization(question)
print(output)