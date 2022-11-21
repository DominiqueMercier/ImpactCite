#!/usr/bin/env python
# coding: utf-8

# In[15]:


#count the number of files in each of the test folder

import os, os.path
import pandas as pd
#DIR = '/tmp'
#print len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])


# In[16]:


cwd = os.getcwd()
root='aclImdb'
test_result_folder=cwd + '/'+root+'/test/result/'
test_method_folder=cwd + '/'+root+'/test/method/'
test_background_folder=cwd + '/'+root+'/test/background/'

def get_num_of_files(folder):
    return len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])


# In[17]:


result_file_count=get_num_of_files(test_result_folder)
method_file_count=get_num_of_files(test_method_folder)
background_file_count=get_num_of_files(test_background_folder)

print("result_file_count",result_file_count)
print("method_file_count",method_file_count)
print("background_file_count",background_file_count)


# In[26]:


result_list=['result']*result_file_count
method_list=['method']*method_file_count
background_list=['background']*background_file_count

label_list=background_list+method_list+result_list
label_list.append('background')
print('label_list',len(label_list))


# In[29]:


predictions_df=pd.read_csv('imdb.tsv', sep='\t', header=[0])
predictions_list=predictions_df['prediction'].values.tolist()
print('predictions_list',len(predictions_list))


# In[35]:


from sklearn.metrics import confusion_matrix,classification_report
import numpy as np

labels = ['result', 'method','background']
cm = confusion_matrix(label_list, predictions_list, labels)
print(classification_report(label_list, predictions_list, labels))
print(cm)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("\nAccuracy for class result: {:.3f} %".format((cm.diagonal()[0]) * 100))
print("Accuracy for class method: {:.3f} %".format((cm.diagonal()[1]) * 100))
print("Accuracy for class background: {:.3f} %".format((cm.diagonal()[2]) * 100))


# In[ ]:




