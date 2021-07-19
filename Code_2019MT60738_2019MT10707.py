import numpy as np
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
import math 
import random
style.use('ggplot')

f = open("C:/Users/Navya/Downloads/handwritten_image_data/handwritten_image_data/training_images.txt", "r")
lines=f.readlines()
data=[[float(x) for x in line.strip().split('\t')] for line in  lines]
f2 = open("C:/Users/Navya/Downloads/handwritten_image_data/handwritten_image_data/training_labels.txt", "r", encoding='utf8')
lines=f2.readlines()
y=np.array([[float(x) for x in line.strip().split('\t')] for line in  lines])

def feed_fwd(x, w1, w2):
    x_0 = np.ones((len(x), 1))
    x = np.concatenate((x_0,x),axis = 1)
    z2= np.dot(x, w1)
    a2= sigmoid(z2)
    a2= np.concatenate((x_0,a2),axis = 1)
    z3= np.dot(a2, w2)
    a3= sigmoid(z3)
    return z2, z3, a2, a3, w1, w2

def weights(sl, sl1):
    e = math.sqrt(6/(sl+sl1))
    w = np.random.uniform (-e, e, size=(sl+1,sl1))
    return w

def sigmoid(z):
    a= 1/(1+np.exp(-z))
    return a

def sigmoid_grad(x):
    return sigmoid(x)*(1-sigmoid(x))

def back_prop(X, y, z2, z3, a2, a3, w1, w2):
    x_0 = np.ones((len(X), 1))
    X1 = np.concatenate((x_0,X),axis = 1)
    #del3 = np.multiply((a3-y), sigmoid_grad(z3))
    del3 = a3-y
    dJdw2 = (1/len(X))*np.dot(a2.transpose(), del3)
    del2 = np.multiply(sigmoid_grad(z2), np.dot(del3, w2.transpose())[:, 1:])
    dJdw1 = (1/len(X))*np.dot(X1.transpose(), del2)
    del1 = np.multiply(sigmoid_grad(X), np.dot(del2, w1.transpose())[:, 1:])
    return del1, del2, del3, dJdw1, dJdw2

def reg_back_prop(X, y, z2, z3, a2, a3, w1, w2,lamda1,lamda2):
    x_0 = np.ones((len(X), 1))
    X1 = np.concatenate((x_0,X),axis = 1)
    #del3 = np.multiply((a3-y), sigmoid_grad(z3))
    del3 = a3-y
    dJdw2 = (1/len(X))*(np.dot(a2.transpose(), del3) + lamda1*(w2))
    del2 = np.multiply(sigmoid_grad(z2), np.dot(del3, w2.transpose())[:, 1:])
    dJdw1 = (1/len(X))*(np.dot(X1.transpose(), del2) + lamda2*(w1))
    del1 = np.multiply(sigmoid_grad(X), np.dot(del2, w1.transpose())[:, 1:])
    return del1, del2, del3, dJdw1, dJdw2

def cost(a3, Y):
    m= Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(a3)) + np.multiply(1 - Y, np.log(1 - a3)))
    cost = np.squeeze(cost)
    return cost


def reg_cost(a3, Y,lamda1,lamda2,w1,w2):
    m= Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(a3)) + np.multiply(1 - Y, np.log(1 - a3))) 
    cost = np.squeeze(cost)
    w1_sum = np.sum(np.square(w1)) 
    w2_sum = np.sum(np.square(w2))
    cost = cost + (lamda1/2*m)* w1_sum + (lamda2/2*m)* w2_sum
    return cost

def parameter_update(w1, w2, dJdw1, dJdw2, learning_rate_1,learning_rate_2):
    w1 = w1 - learning_rate_1*dJdw1
    w2 = w2 - learning_rate_2*dJdw2
    return w1, w2

def neural_network(x,y,testing_data,testing_output,num_iter=1000,learning_rate_1=0.1,learning_rate_2=0.1,tol=10**-5,reg = "n",lamda1 = 1,lamda2 = 1,hidden_layer_size = int(50)):
    #hidden_layer_size= int((2/3)*784 +10)
    #hidden_layer_size= int(50)
    w1= weights(len(x[0]), hidden_layer_size)
    w2= weights(hidden_layer_size,10)
    Y= np.zeros((len(y), 10))
    Y_test= np.zeros((len(testing_output), 10))
    x=np.array(x)
    testing_data=np.array(testing_data)
    for i in range(len(y)):
        Y[i][int(y[i])]=1
    for i in range(len(testing_output)):
        Y_test[i][int(testing_output[i])]=1
    train_cost=[]
    test_cost = []
    for i in range(num_iter):
        if (reg == "n"):
            z2, z3, a2, a3, w1, w2= feed_fwd(x, w1, w2)
            c_train = cost(a3, Y)
            train_cost.append(c_train)
            z2_test, z3_test, a2_test, a3_test, w1, w2= feed_fwd(testing_data, w1, w2)
            c_test = cost(a3_test,Y_test)
            test_cost.append(c_test)
            del1, del2, del3, dJdw1, dJdw2= back_prop(x, Y, z2, z3, a2, a3, w1, w2)    
        else:
            z2, z3, a2, a3, w1, w2= feed_fwd(x, w1, w2)
            c_train = reg_cost(a3, Y,lamda1,lamda2,w1,w2)
            train_cost.append(c_train)
            z2_test, z3_test, a2_test, a3_test, w1, w2= feed_fwd(testing_data, w1, w2)
            c_test = reg_cost(a3_test,Y_test,lamda1,lamda2,w1,w2)
            test_cost.append(c_test)
            del1, del2, del3, dJdw1, dJdw2= reg_back_prop(x, Y, z2, z3, a2, a3, w1, w2,lamda1,lamda2)
        w1, w2= parameter_update(w1,w2,dJdw1,dJdw2,learning_rate_1,learning_rate_2)
        if c_train<= tol:
            break
    #predictions = predict_nn(w1, w2, x)
    #print("Train accuracy: {} %", sum(predictions == y) / (float(len(y))) * 100)
    #predictions=predict_nn(parameters,test_x)
    #print("Train accuracy: {} %", sum(predictions == test_y) / (float(len(test_y))) * 100)
    cost_v_iter(num_iter,x,train_cost,testing_data,test_cost)
    return w1, w2

def predict_nn(w1, w2, test_X, test_y,what = "Test"):
    z2, z3, a2, a3, w1, w2 = feed_fwd(test_X, w1, w2)
    predictions = np.argmax(a3, axis=1)
    c=0
    for i in range(len(test_y)):
        if predictions[i] == test_y[i]:
            c=c+1
    Error = 1 - (c/(float(len(test_y))))
    Percent = 100* (1-Error)
    if (what == "Test"):
        print("Test Accuracy:%", Percent)
    elif (what == "Valid"):
        print("Validation Accuracy:%", Percent)
    else: 
        print("Train Accuracy:%", Percent)
    return predictions,Error, Percent

def main(training_points,testing_points,num_iter =1000,learning_rate_1=0.1,learning_rate_2=0.1,tol=10**-5,reg = "n",lamda1=1,lamda2 = 1,hidden_layer_size = int(50)):
    Data = np.concatenate((data,y),1)
    random.shuffle(Data)
    training_data = np.array(Data[:training_points,:784])
    testing_data = np.array(Data[training_points:,:784])
    training_output = np.array(Data[:training_points,784])
    testing_output = np.array(Data[training_points:,784])
    w1,w2 = neural_network(training_data,training_output,testing_data,testing_output, num_iter,learning_rate_1,learning_rate_2,tol,reg,lamda1,lamda2,hidden_layer_size)
    useless,train_error,train_acc = predict_nn(w1, w2, training_data,training_output,"Train")
    predictions,test_error,test_acc = predict_nn(w1, w2, testing_data,testing_output,"Test") 
    display(testing_data,testing_output,predictions)
    return test_error,test_acc,train_error,train_acc

def cross_validation(Training_data,Training_output,testing_data,testing_output,folds = 5,num_iter = 1000,learning_rate_1=0.1,learning_rate_2=0.1,tol=10**-5,reg = "n",lamda1 = 1,lamda2 = 1):
    N = int(len(Training_data)/folds)
    Validation_Error = []
    for i in range(0,folds):
        validation_data = Training_data[N*i:(i+1)*N,:]
        validation_output = Training_output[N*i:(i+1)*N]
        if (i==0):
            training_data = Training_data[N:,:]
            training_output = Training_output[N:]
        elif (i == folds-1):
            training_data = Training_data[:N*i,:]
            training_output = Training_output[:N*i]
        else:
            training_data = np.concatenate((Training_data[:N*i,:],Training_data[(i+1)*N:,:]),axis = 0)
            training_output = np.concatenate((Training_output[:N*i],Training_output[(i+1)*N:]),axis = 0)
        w1,w2 = neural_network(training_data,training_output,testing_data,testing_output,num_iter,learning_rate_1,learning_rate_2, tol,reg ,lamda1,lamda2)
        predictions,validation_error,validation_percent = predict_nn(w1, w2, validation_data,validation_output,"Valid")
        Validation_Error.append(validation_error)
    Validation_Error = np.array(Validation_Error)
    Average_Validation_Error = (1/folds)*np.sum(Validation_Error)
    Average_Validation_Accuracy = 100*(1-Average_Validation_Error)
    print("Cross Validation Error on ",folds," folds is ", Average_Validation_Accuracy)
    return Average_Validation_Error,Average_Validation_Accuracy
    
def cost_v_iter(num_iter,training_data,train_cost,testing_data,test_cost):
    Train_Cost = np.array(train_cost)
    Test_Cost = np.array(test_cost)
    plt.figure(1)
    plt.plot(np.arange(1, num_iter+1), Train_Cost, label = "Training Cost")
    plt.plot(np.arange(1, num_iter+1), Test_Cost, label = "Testing Cost")
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Training and Testing Error vs Iterations')
    plt.legend()
    plt.show()


def display(testing_data,testing_output,predictions):
    # Plotting 30 random samples from the test set with their predictions
    indices = np.random.choice(len(testing_output),30)
    fig = plt.figure()
    for i in range(0,30):
        img = testing_data[indices[i],:].reshape(28,28).transpose()
        fig.add_subplot(5,6,i+1)
        plt.imshow(img,cmap='gray')
        plt.title('y= '+str(predictions[indices[i]]))
        plt.axis("off")
    plt.subplots_adjust(hspace = 0.6)
    plt.show()

def main_valid(training_points, testing_points,folds = 5,num_iter = 1000,learning_rate_1=0.1,learning_rate_2=0.1,tol=10**-5,reg = "n",lamda1 = 1,lamda2 = 1):
    Data = np.concatenate((data,y),1)
    random.shuffle(Data)
    training_data = np.array(Data[:training_points,:784])
    testing_data = np.array(Data[training_points:,:784])
    training_output = np.array(Data[:training_points,784])
    testing_output = np.array(Data[training_points:,784])
    return cross_validation(training_data,training_output,testing_data,testing_output,folds,num_iter,learning_rate_1,learning_rate_2,tol,reg,lamda1,lamda2) 


def Matrix_Maker(LearningRate1_Set,LearningRate2_Set,Validation_Accuracies):
    table = []
    table.append(['Learning Rate for W1','Learning Rate for W2','Cross Validation Accuracies'])
    for i in range(0,len(LearningRate1_Set)):
        table.append([LearningRate1_Set[i],LearningRate2_Set[i],Validation_Accuracies[i]])
    from tabulate import tabulate
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    
    plt.scatter(LearningRate1_Set,Validation_Accuracies,c='b',label = "Learning Rate for W1")
    plt.scatter(LearningRate2_Set,Validation_Accuracies,c='r',label = "Learning Rate for W2")
    plt.xlabel('Learning Rates')
    plt.ylabel('Validation_Accuracies')
    plt.title('Learning rates vs Validation_Accuracies')
    plt.legend()
    plt.show()


#For learning rate
LearningRate1_Set = [0.001,0.01,0.1,1]
LearningRate2_Set = [0.001,0.01,0.1,1]
Validation_Errors = []
Validation_Accuracies = []
for i in LearningRate1_Set:
    for j in LearningRate2_Set:
        print("For learning_rate 1= "+ str(i)+ " and learning_rate 2= "+str(j))
        Average_Validation_Error,Average_Validation_Accuracy= main_valid(4250, 750,3,500,i,j,10**-5,"n",1,1)
        Validation_Errors.append(Average_Validation_Error)
        Validation_Accuracies.append(Average_Validation_Accuracy)
Matrix_Maker(LearningRate1_Set,LearningRate2_Set,Validation_Accuracies)

#For regularisation parameter
Lamda1_Set = [0.1,0.5,1,5]
Lamda2_Set = [0.1,0.5,1,5]
Validation_regErrors = []
Validation_regAccuracies = []
for i in Lamda1_Set:
    for j in Lamda2_Set:
        print("For lambda 1= "+ str(i)+ " and lambda 2= "+str(j))
        Average_Validationreg_Error,Average_Validationreg_Accuracy= main_valid(4250, 750, 3,500, 1, 1,10**-5,"y",i,j)
        Validation_regErrors.append(Average_Validationreg_Error)
        Validation_regAccuracies.append(Average_Validationreg_Accuracy)
Matrix_Maker(Lamda1_Set,Lamda1_Set,Validationreg_Accuracies)

LearningRate1_Set2 = [5,1]
LearningRate2_Set2 = [5,1]
Validation_Errors2 = []
Validation_Accuracies2 = []
for i in LearningRate1_Set2:
    for j in LearningRate2_Set2:
        print("For learning_rate 1= "+ str(i)+ " and learning_rate 2= "+str(j))
        Average_Validation_Error2,Average_Validation_Accuracy2= main_valid(4250,750,3,500,i,j,10**-5,"n",1,1)
        Validation_Errors2.append(Average_Validation_Error2)
        Validation_Accuracies2.append(Average_Validation_Accuracy2)

import seaborn as sn
data = np.matrix('16.31 26.22 53.93 76.93 0; 47.29 61.39 71.82 83.54 0; 66.29 77.37 85.68 90.88 0; 80.29 86.88 91.12 94.32 93.71; 0 0 0 94.20 74.31')
x_axis_labels = [0.001,0.01,0.1,1,5]
y_axis_labels = [0.001,0.01,0.1,1,5]
hm = sn.heatmap(data=data,xticklabels=x_axis_labels, yticklabels=y_axis_labels,annot=True,cmap="bone")
plt.xlabel('Learning Rate for W2')
plt.ylabel('Learning Rate for W1')
plt.title('Cross-Validation Accuracies')
plt.show()

data = np.matrix('94.2 94.82 94.23 93.12; 93.87 94.75 92.93 93.24; 94.18 94.25 93.80 93.47; 93.47 93.59 92.96 92.51')
x_axis_labels = [0.1,0.5,1,5]
y_axis_labels = [0.1,0.5,1,5]
hm = sn.heatmap(data=data,xticklabels=x_axis_labels, yticklabels=y_axis_labels,annot=True,cmap="bone")
plt.xlabel('Lambda 2 ')
plt.ylabel('Lambda 1')
plt.title('Cross-Validation Accuracies')
plt.show()

#nodes of hidden layer
t= int((784)*2/3 +10)
Hidden_Layer_Size = [9,60,250,t,1000]
Train_Accuracies= []
Test_Accuracies = []
for i in Hidden_Layer_Size:
    print("For Hidden layer size= "+ str(i))
    test_error,test_acc,train_error,train_acc= main(4250,750,750,1,1,10**-5,"n",0.1,0.5,i)
    Train_Accuracies.append(train_acc)
    Test_Accuracies.append(test_acc)
table = []
table.append(['Nodes in the Hidden Layer','Training Accuracy','Testing Accuracy'])
for i in range(0,len(Hidden_Layer_Size)):
    table.append([Hidden_Layer_Size[i],Train_Accuracies[i],Test_Accuracies[i]])
from tabulate import tabulate
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    
plt.scatter(Hidden_Layer_Size,Train_Accuracies,c='b',label = "Training Accuracy")
plt.scatter(Hidden_Layer_Size,Test_Accuracies,c='r',label = "Testing Accuracy")
plt.xlabel('Nodes in the Hidden Layer')
plt.ylabel('Accuracies')
plt.title('Nodes in the Hidden Layer vs Accuracies')
plt.legend()
plt.show()

#nodes of hidden layer
t= int((784)*2/3 +10)
Hidden_Layer_Size2 = [5,2000]
Train_Accuracies2= []
Test_Accuracies2 = []
for i in Hidden_Layer_Size2:
    print("For Hidden layer size= "+ str(i))
    test_error,test_acc,train_error,train_acc= main(4250,750,750,1,1,10**-5,"n",0.1,0.5,i)
    Train_Accuracies2.append(train_acc)
    Test_Accuracies2.append(test_acc)
table2 = []
table2.append(['Nodes in the Hidden Layer','Training Accuracy','Testing Accuracy'])
for i in range(0,len(Hidden_Layer_Size2)):
    table2.append([Hidden_Layer_Size2[i],Train_Accuracies2[i],Test_Accuracies2[i]])
from tabulate import tabulate
print(tabulate(table2, headers='firstrow', tablefmt='fancy_grid'))
    
plt.scatter(Hidden_Layer_Size2,Train_Accuracies2,c='b',label = "Training Accuracy")
plt.scatter(Hidden_Layer_Size2,Test_Accuracies2,c='r',label = "Testing Accuracy")
plt.xlabel('Nodes in the Hidden Layer')
plt.ylabel('Accuracies')
plt.title('Nodes in the Hidden Layer vs Accuracies')
plt.legend()
plt.show()

#check your own image!!!\
from PIL import Image, ImageChops
import cv2
import os

ima=Image.open('C:/Users/Navya/Downloads/twooo.jpeg')
imag= ImageChops.invert(ima)
grayimg= cv2.cvtColor(np.float32(imag), cv2.COLOR_RGB2GRAY)
img = cv2.resize(grayimg, (28, 28))
img_flatten = img.reshape(-1, 1)
plt.imshow(img,cmap='gray')
zb2, zb3, ba2, ans, wb1, wb2= feed_fwd(img_flatten.transpose(), w1, w2)  
predictn= np.argmax(ans, axis=1)
plt.title('y= '+str(predictn))
plt.show()