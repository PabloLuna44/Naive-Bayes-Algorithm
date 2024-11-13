from models.naiveBayes import NaiveBayes
import pandas as pd



print("**************************Naive Bayes***********************")


print("---------------Dataset-----------------")
dataset = pd.read_csv('datasets/dataset.csv')
instance={'Outlook':'overcast','Temp': 65, 'Humidity': 70, 'Windy':True }
print(dataset)
print("\nInstance: ",instance)
nb=NaiveBayes(dataset,'Play',instance)


print("\n---------------Probabilidades-----------------")
results=nb.calculateProbabilities()
for result in results:
    print("Clase: ",result['class'])
    print("Probabilidad: ",result['prob'],"%") 


print("\n---------------TRAIN-----------------")

train = pd.read_csv('datasets/train.csv')
trainInstance={'Outlook':'rainy','Temp': 71, 'Humidity': 85, 'Windy':True }
print(train)
print("\nInstance: ",trainInstance)
naiveTrain=NaiveBayes(train,'Play',trainInstance)

print("\n---------------Probabilidades-----------------")
results=naiveTrain.calculateProbabilities()
for result in results:
    print("Clase: ",result['class'])
    print("Probabilidad: ",result['prob'],"%") 


    
print("\n---------------TEST-----------------")

test = pd.read_csv('datasets/test.csv')
testInstance={'Outlook':'overcast','Temp': 72, 'Humidity': 70, 'Windy':False }
print(test)
print("\nInstance: ",testInstance)

naiveTest=NaiveBayes(test,'Play',testInstance)

print("\n---------------Probabilidades-----------------")
results=naiveTest.calculateProbabilities()
for result in results:
    print("Clase: ",result['class'])
    print("Probabilidad: ",result['prob'],"%") 
