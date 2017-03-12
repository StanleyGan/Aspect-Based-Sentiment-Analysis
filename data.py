'''Assign some labels to data set, split into train, validation and test data'''

from itertools import dropwhile
import pandas as pd
import re
import os
import random

class data:
    
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.review_pattern = "\[t\]*"
        self.aspectAndScore_pattern = "[a-zA-Z0-9]*\[[+-][0-9]\]"
        self.aspect_pattern = "[a-zA-Z0-9]*\["
        self.score_pattern = "[+-][0-9]"
        self.textFile_pattern = "(.*).txt"
        self.othertag_pattern = "\[[upsc]{1,2}\]"
        
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
        
        training = pd.DataFrame(columns=["Review", "Aspect"])
        validation = pd.DataFrame(columns=["Review", "Aspect"])
        test = pd.DataFrame(columns=["Review", "Aspect"])
        for prod in summarizeData:
            size = summarizeData[prod].shape[0]
            num_test = int( round(self.test_prop * size) )
            test_ind = random.sample(range(size), num_test)
            test = test.append(summarizeData[prod].iloc[test_ind])
            
            num_val = int( round(self.val_prop * size) )
            remaining_index = [i for i in range(size) if i not in test_ind]
            val_index = random.sample(remaining_index, num_val)
            validation = validation.append(summarizeData[prod].iloc[val_index])
            training = training.append(summarizeData[prod].iloc[ [i for i in remaining_index if i not in val_index] ])
      
        return training, validation, test

    def getData(self):
        return self._splitData_revAndAsp()
    
#canong3 = summarizeData("/home/stanleygan/Documents/Deep_Learning/project/customer_review_data/Canon G3.txt")         

if __name__ == '__main__':
    d = data("/home/stanleygan/Documents/Deep_Learning/project/customer_review_data/")
    training, validation, test = d.getData()