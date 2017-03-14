'''Assign some labels to data set, split into train, validation and test data'''

from itertools import dropwhile
import pandas as pd
import re
import os
import random
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class data:
    
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.review_pattern = "\[t\]*"
        self.aspectAndScore_pattern = "[a-zA-Z0-9]*\[[+-][0-9]\]"
        self.aspect_pattern = "[a-zA-Z0-9]*\["
        self.score_pattern = "[+-][0-9]"
        self.textFile_pattern = "(.*).txt"
        self.othertag_pattern = "\[[upsc]{1,2}\]"
        self.vocab = list()
        self.vocab_size = 500
        self.numReviews = 0;
        
        t = re.compile(self.textFile_pattern)
        dirs = os.listdir(self.dataPath)
        self.products = [t.findall(i) for i in dirs]
        self.products = [''.join(i) for i in self.products]
        
        self.test_prop = 0.05
        self.val_prop = 0.05
    
    '''
    Input: path to the text file containing reviews
    
    Output: Returns a dataframe containing aspect and scores(not unique) and a 
            dictionary with key=review, value=aspects
    '''
    def _summarizeData(self, oneproduct):
        with open("{0}{1}.txt".format(self.dataPath, oneproduct), 'r') as f:
            data = f.read()
            
        #split according to review tags    
        data = re.split(self.review_pattern, data)[1:]
        
        #regex for grabbing aspect and score, after that grab aspect and score individually        
        r1 = re.compile(self.aspectAndScore_pattern)
        r2 = re.compile(self.aspect_pattern)
        r3 = re.compile(self.score_pattern)
        #aspect and score dataframe
        data_aspAndSc_DF = pd.DataFrame(columns=['Aspect', 'Score'])
        #review and aspect
        data_revAndAsp_DF = pd.DataFrame(columns=['Review', 'Aspect'])
        listOfdata_asp = list()
        data_asp = list()
        data_sc = list()
        for rev in data:
            match_aspAndSc = r1.findall(rev)
            
            listOfAsp = list()
            for aspNSc in match_aspAndSc:
                match_asp = r2.findall(aspNSc)
                match_score = r3.findall(aspNSc)
                
                listOfAsp.append(match_asp[0][:-1])
                data_asp.append(match_asp[0][:-1])   #as will return 'a[', remove '['
                data_sc.append(int(''.join(match_score)))
                    
            listOfAsp = set(listOfAsp)
            listOfAsp = list(listOfAsp)
            listOfdata_asp.append(listOfAsp)
        
        data_aspAndSc_DF['Aspect'] = data_asp
        data_aspAndSc_DF['Score'] = data_sc
          
        #filter review, remove labels
        i=0
        for rev in data:
            data[i] = re.sub("##", '', re.sub(self.aspectAndScore_pattern, '', rev))
            data[i] = re.sub(self.othertag_pattern, '', data[i])
            data[i] = data[i].replace("\r\n", " ")
            i = i+1
            
        data_revAndAsp_DF['Review'] = data
        data_revAndAsp_DF['Aspect'] = listOfdata_asp
          
        #return review and aspect for now
        return data_revAndAsp_DF
    
    def _splitData_revAndAsp(self):
        summarizeData = dict()
        for prod in self.products:
            summarizeData[prod] = self._summarizeData(prod)
            
        allData = pd.concat([summarizeData[p] for p in self.products])
        allData_feat, allData_label, self.vocab = self._toNumAndLabels(allData["Review"].tolist(), allData["Aspect"].tolist())

        numTest = int( round(self.test_prop * self.numReviews) )
        test_ind = random.sample(range(self.numReviews), numTest)
        test_feat = allData_feat[test_ind, :]
        test_label = allData_label[test_ind, :]
        
        remaining_index = [i for i in range(self.numReviews) if i not in test_ind]
        num_val = int( round(self.val_prop * self.numReviews) )
        val_index = random.sample(remaining_index, num_val)
        validation_feat = allData_feat[val_index, :]
        validation_label = allData_label[val_index, :]
        training_feat = allData_feat[ [i for i in remaining_index if i not in val_index], : ] 
        training_label = allData_label[ [i for i in remaining_index if i not in val_index], : ] 
      
        return training_feat, training_label, validation_feat, validation_label, test_feat, test_label
    
    def _toNumAndLabels(self, rev, aspect):
        self.numReviews = len(rev)
        vectorizer = CountVectorizer(analyzer="word", stop_words="english", max_features=self.vocab_size)
        features = vectorizer.fit_transform(rev)
        vocab = vectorizer.get_feature_names()
        
        label= []
        for a in aspect:
            temp = np.zeros((len(vocab),1))
            temp[[i for i, item in enumerate(vocab) if item in a]] = 1
            label.append(temp)
            
        label = np.array(label)
        label = np.reshape(label, [self.numReviews, -1])
        features = features.toarray()
        features = np.reshape(features, [self.numReviews, -1])
        return (features, label, vocab)
    
    def getData(self):
        return self._splitData_revAndAsp()
    
    def getVocab(self):
        return self.vocab
        
#canong3 = summarizeData("/home/stanleygan/Documents/Deep_Learning/project/customer_review_data/Canon G3.txt")         

if __name__ == '__main__':
    d = data("/home/stanleygan/Documents/Deep_Learning/project/customer_review_data/")
    train_feat, train_lab, val_feat, val_lab, test_feat, test_lab = d.getData()