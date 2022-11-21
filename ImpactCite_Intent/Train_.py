
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import jsonlines
import shutil

from string import punctuation
import string
import numpy as np


# In[34]:


#lets create the folder structure
cwd = os.getcwd()
root='aclImdb'
p = Path(root)
if not p.exists():
    os.mkdir(root)
    os.makedirs(cwd + '/'+root+'/test/result')
    os.makedirs(cwd + '/'+root+'/test/method')
    os.makedirs(cwd + '/'+root+'/test/background')
    os.makedirs(cwd + '/'+root+'/train/result')
    os.makedirs(cwd + '/'+root+'/train/method')
    os.makedirs(cwd + '/'+root+'/train/background')
    

test_result_folder=cwd + '/'+root+'/test/result/'
test_method_folder=cwd + '/'+root+'/test/method/'
test_background_folder=cwd + '/'+root+'/test/background/'
train_result_folder=cwd + '/'+root+'/train/result/'
train_method_folder=cwd + '/'+root+'/train/method/'
train_background_folder=cwd + '/'+root+'/train/background/'


# In[35]:


def create_folder(folder_name):
    p = Path(folder_name)
    if p.exists():
        shutil.rmtree(p)
        print('Deleteing old LSTM_SMOTE folder')
    try:
        print("Creating_folder")
        os.mkdir(folder_name)
    except OSError:
        print("Creation of the directory %s failed" % 'output')
folder_name='XLNET_intent_output'
create_folder(folder_name)


# In[36]:


def print_and_write(folder_name,text):
    file_path=folder_name+"/output.txt"
    p = Path(file_path)
    if p.exists():
        f=open(file_path, "a+")
        f.write("\n"+text)
        print(text)
        f.close()
    else:
        f= open(file_path,"w+")
        f.write("\n"+text)
        print(text)
        f.close()


# In[37]:


def load_citation_intent_corpus(file_path,do_print=False,folder_name=''):

    result_list = []
    method_list = []
    background_list = []
    result_num = 0
    method_num = 0
    backgroud_num = 0

    for ex in jsonlines.open(file_path):
        text = ex.get('string')
        label = ex.get('label')

        if label == 'result':
            result_num += 1
            result_list.append(text)

        if label == 'method':
            method_num += 1
            method_list.append(text)

        if label == 'background':
            backgroud_num += 1
            background_list.append(text)

    if do_print:
        print_and_write(folder_name,"Number of examples of type :result= "+ str(result_num))
        print_and_write(folder_name,"Number of examples of type :method= "+str(method_num))
        print_and_write(folder_name,"Number of examples of type :background="+str(backgroud_num))

    return result_list,method_list,background_list


# In[38]:


train_path='./scicite_data/train.jsonl'
val_path='./scicite_data/dev.jsonl'
test_path='./scicite_data/test.jsonl'
print_and_write(folder_name,"\nDetails of training folder")
train_result_list,train_method_list,train_background_list=load_citation_intent_corpus(train_path,True,folder_name)
print_and_write(folder_name,"\nDetails of validation folder")
val_result_list,val_method_list,val_background_list=load_citation_intent_corpus(val_path,True,folder_name)
print_and_write(folder_name,"\nDetails of testing folder")
test_result_list,test_method_list,test_background_list=load_citation_intent_corpus(test_path,True,folder_name)



train_result_list=train_result_list+val_result_list
train_method_list=train_method_list+val_method_list
train_background_list=train_background_list+val_background_list



# In[39]:


print(len(train_result_list))
print(len(train_method_list))
print(len(train_background_list))


# In[40]:


from sklearn.model_selection import train_test_split
def write_data(list_to_write,folder):
    count=0
    for data in list_to_write:
        text_file = open(folder+str(count)+"_1.txt", "w")
        text_file.write(data)
        text_file.close()
        count+=1
    
        


# In[41]:


write_data(train_result_list,train_result_folder)
write_data(train_method_list,train_method_folder)
write_data(train_background_list,train_background_folder)
write_data(test_result_list,test_result_folder)
write_data(test_method_list,test_method_folder)
write_data(test_background_list,test_background_folder)
