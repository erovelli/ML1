'''
Evan Rovelli
ECE 241
Project 3 - part 2
12/17/21
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class HouseLibrary:
    def __init__(self):
        self.houseList = []  # houses list
        self.size = None # size of house list
        self.minPrice = None  # minimum price
        self.maxPrice = None  # maximum price
        self.stdDv = None  # standard deviation
        self.avgPrice = None  # average house price
        self.X = [] # feature data
        self.W = None # final training weight
        self.Wsave1 = None
        self.Wsave2 = None

    def __str__(self): # object print
        s = "Min: %s, Max %s, Mean: %s, Standard Deviation: %s" % (
            self.minPrice, self.maxPrice, self.avgPrice, self.stdDv)
        return s

    def loadData(self, filename: str):
        data = pd.read_csv(filename) # read csv
        self.houseList = pd.DataFrame(data) # creates pandas dataframe
        self.size = len(self.houseList) # stores house list size
        self.pList = self.houseList["Price"]
        self.minPrice = self.pList.min() # minimum house price
        self.maxPrice = self.pList.max() # maximum house price
        self.avgPrice = self.pList.mean() # mean house price
        self.stdDv = self.pList.std() # standard deviation of house price

        ### format csv data to just feature list
        self.X = self.houseList.loc[:, self.houseList.columns != "Price"]
        self.X = self.X.loc[:, self.X.columns != "Id"]

    def plotHistogram(self, N):
        plt.hist(self.pList, N, facecolor='red', alpha=0.5)

        ### graph details
        plt.xlabel('sale price')
        plt.ylabel('number of houses')
        plt.title('Histogram of sales price \n(%s bars)'%N)
        plt.grid(True)
        plt.show()

    def plotPairScatter(self):
        #plt.figure()
        sns.pairplot(self.houseList[["GrLivArea", "BedroomAbvGr", "TotalBsmtSF", "FullBath"]])
        plt.show()

    def Pred(self, W):
        return np.dot(self.X, W)

    def loss(self, W):
        return sum((self.Pred(W) - self.pList) ** 2) / self.size

    def gradient(self, W):
        return (2 / self.size) * np.dot(np.transpose(self.X), self.Pred(W) - self.pList)

    def update(self, W, a, N):
        MSE = [self.loss(W)]
        for i in range(N):
            W = W - a * self.gradient(W)
            MSE.append(self.loss(W))
        self.W = W

        return MSE


    def plot1MSE(self, W, a, N):
        MSE = self.update(W, a, N)

        x = np.arange(1, N+2)
        plt.plot(x, MSE)
        plt.legend(['MSE w/ α = %s' % a])
        plt.yscale('log')
        plt.xlabel('Iterations')
        plt.ylabel('MSE')
        plt.title('plot of MSE per number of iterations')
        plt.show()


    def plot2MSE(self, W, a1, a2, N):
        MSE1 = self.update(W, a1, N)
        self.Wsave1 = self.W
        MSE2 = self.update(W, a2, N)
        self.Wsave2 = self.W

        x = np.arange(1, N+2)
        plt.plot(x, MSE1)
        plt.plot(x, MSE2)
        plt.legend(['MSE w/ α = %s'%a1, 'MSE w/ α = %s'%a2])
        plt.yscale('log')
        plt.xlabel('Iterations')
        plt.ylabel('MSE')
        plt.title('plot of MSE per number of iterations')
        plt.show()


if __name__ == '__main__':
    ### initialized values
    file1 = "train.csv"
    file2 = "test.csv"
    a = 0.2
    a1 = 10**(-11) # test alpha 1
    a2 = 10**(-12) # test alpha 2
    N = 50000 # number of iterations
    np.random.seed(1)

    ### initialize data and class
    A = HouseLibrary() # initialized object
    A.loadData(file1) # load data
    W = np.random.rand(len(A.houseList.columns) - 2) #initialize weight vector

    # Q2: data statistics
    # print(A)

    # Q3: Histogram of sales price
    #  A.plotHistogram(50) # plots 50 bins of sale prices histogram

    # Q4: pair-wise scatter plot
    # A.plotPairScatter()

    # Q5: pred function
    # print("pred:\n",A.Pred(W))

    # Q6: loss function
    # print("loss:", A.loss(W))

    # Q7: gradient function
    # print("gradient:\n", A.gradient(W))

    # Q8-9: update function
    # print(A.update(W, a, N))

    # Q10: running 500 update iterations with alpha = 0.2 (also visualizing for fun)
    # A.plot1MSE(W, a, N)

    # Q11:
    # A.plot2MSE(W, a1, a2, N)

    # Q13: compare model to test csv
    # B = HouseLibrary()  # initialized object
    # B.loadData(file2)  # load data
    # MSEtrain = A.loss(A.Wsave1)
    # MSEtest = B.loss(A.Wsave1)
    # print("Training MSE: %s \nTest MSE: %s"%(MSEtrain, MSEtest))




