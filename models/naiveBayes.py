import pandas as pd
import math
import numpy as np


class NaiveBayes:

    #Input dataset, target column, instance to classify
    def __init__(self,dataset,target,instance):
        self.dataset=dataset
        self.target=target
        self.classes=dataset[target].unique()
        self.instance=instance
        self.classProbabilities={}
        self.numericalColumnsClassValues={}
        self.categoricalColumnsClassValues={}
        self.classSubsets={}
        self.probabilitiesByClass={}
        self.aprioriProbabilities()
        self.splitByClassValues()
        
        
    #output: dictionary with class values and their probabilities
    def aprioriProbabilities(self):
        for classValue in self.classes:
            self.classProbabilities[classValue]=self.dataset[self.target].value_counts()[classValue]/self.dataset.shape[0]
        return self.classProbabilities
    

    #output: dictionary with class values and their subsets
    def splitByClassValues(self):
        for classValue in self.classes:
            class_subset = self.dataset[self.dataset[self.target] == classValue]
            self.classSubsets[classValue] = class_subset  
            self.numericalColumnsClassValues[classValue] = class_subset.select_dtypes(include=['number'])
            self.categoricalColumnsClassValues[classValue] = class_subset.select_dtypes(include=['object', 'bool']).drop(columns=[self.target])
        
        return self.classSubsets, self.numericalColumnsClassValues, self.categoricalColumnsClassValues

    #input: value
    #output: rounded value
    def custom_round(self,value):
        return math.floor(value + 0.5)
    

    #input: x, mean, std
    #output: normal distribution value
    def normal_distribution(self,x, mean, std):
        result = (1 / (math.sqrt(2 * math.pi * std))) * math.exp(-((x - mean)**2 / (2 * std**2)))
        result=str(result)
        return float(result[0:7])
    

    #output: array with probabilities of each numeric class
    def calculateNumeric(self):

        arr=[]
        for classValue in self.numericalColumnsClassValues:

            for column in self.numericalColumnsClassValues[classValue]:
                mean = self.custom_round(self.numericalColumnsClassValues[classValue][column].mean())
                std = round(self.numericalColumnsClassValues[classValue][column].std(), 1)
                instance_value = self.instance[column]
                arr.append({
                    'column': column,
                    'class': classValue,
                    'prob': self.normal_distribution(instance_value, mean, std)
                })
        return arr
    
    #input: classValue, atributeValue
    #output: laplace smoothing value
    def laplaceSmoothing(self,classValue, atributeValue):
        result = 1 /(classValue + (1*atributeValue))
        return float(result)
    

    #output: array with probabilities of each categorical class
    def calculateCategorical(self):

        arr=[]
        for classValue in self.categoricalColumnsClassValues:
            for column in self.categoricalColumnsClassValues[classValue]:
                value = self.categoricalColumnsClassValues[classValue][column].value_counts(normalize=True).to_dict()
                instance_value = self.instance[column]
                prob=value.get(instance_value, 0)
                if(prob==0):
                    unique = self.categoricalColumnsClassValues[classValue][column].count()
                    prob=self.laplaceSmoothing(self.classSubsets[classValue].shape[0],unique)
                arr.append({
                    'column': column,
                    'class': classValue,
                    'prob': prob
                })
        return arr
    

    #output: array with probabilities of each class
    def calculateProbabilities(self):
        
        arr=[]
        numeric=self.calculateNumeric()
        categorical=self.calculateCategorical()

        for numericProb in numeric:

            for classValue in self.classes:
                
                if(numericProb['class']==classValue):
                    if classValue not in self.probabilitiesByClass:
                        self.probabilitiesByClass[classValue]=self.classProbabilities[classValue]
                    self.probabilitiesByClass[classValue] *= numericProb['prob']
                    
        for categoricalProb in categorical:
            for classValue in self.classes:
                if(categoricalProb['class']==classValue):
                    if classValue not in self.probabilitiesByClass:
                        self.probabilitiesByClass[classValue]=self.classProbabilities[classValue]
                    self.probabilitiesByClass[classValue] *= categoricalProb['prob']
        
        
        totalSum=sum(self.probabilitiesByClass.values())
        

        for classValue in self.probabilitiesByClass:
            self.probabilitiesByClass[classValue]=round(self.probabilitiesByClass[classValue]/totalSum,3)
            arr.append({
                'class': classValue,
                'prob': self.probabilitiesByClass[classValue] * 100
            })
        
        return arr
        
                

    