import math
import numpy as np

from copy import deepcopy


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)



class Neuro ():
    def __init__(self,xx, activation):
        self.a = np.random.rand( xx )*2
        self.vixod = 0
        self.delta = []
        self.activation = activation
    
    def summa (self, x):
        self.vixod = self.a.dot(x)
        if self.activation == "None":
            return self.vixod
        if self.activation == "Sigmoid":
            return sigmoid (self.vixod)
        
    
    def back (self,grad,vxod):
        #print("ВХОД", vxod)
        #print("ГРАД", grad )
        
        if self.activation == "None":
            self.delta.append (  grad * vxod )
            return self.a * grad
        
        if self.activation == "Sigmoid":
            self.delta.append (  grad * deriv_sigmoid(self.vixod)* vxod )
            return self.a * deriv_sigmoid(self.vixod)* grad
        
        
    def pri(self, lr):
        self.delta = np.array(self.delta)

        self.a = self.a - self.delta.mean(axis=0) * lr
        self.delta = []
        
class Layer ():
    def __init__(self, kvo_neurons, kvo_vxod, activation):

        self.neurons = []
        
        
        for i in range(kvo_neurons):
            self.neurons.append(Neuro(kvo_vxod, activation ))
            
            
            
    def forward_lay (self, vxod):
        self.vixods = np.array([])
        for neuron in self.neurons:
            self.vixods = np.append(self.vixods,  neuron.summa(vxod)  )


        return self.vixods
    

class Model ():
    def __init__ (self):
        self.layers = []
        
        
    def add_input_layer(self, kvo_neurons, kvo_vxod, activation="None"):
        self.layers.append(  Layer( kvo_neurons, kvo_vxod, activation) )
    
    def add_layer(self, kvo_neurons, activation="None"):
        self.layers.append(  Layer(  kvo_neurons, len(self.layers[-1].neurons) , activation)  ) 
    
    def forward (self, x):
        self.stroki = deepcopy(range(x.shape[0]))
        if len(x.shape) == 1:
            
           
            self.vixods = []
            self.vixods.append(x)
            
            for layer in self.layers:
                self.vixods.append(layer.forward_lay(x))
                x = layer.forward_lay(x)
            
            return self.vixods[-1]
            
        if len(x.shape) == 2:
            otvets = []
            for stroka in self.stroki :
                
                    
                x_n = x[stroka]

                   

                self.vixods = []
                self.vixods.append(x_n)


                for layer in self.layers:
                    self.vixods.append(layer.forward_lay(x_n))
                    x_n = layer.forward_lay(x_n)


                    
                otvets.append(self.vixods[-1])
                
            return otvets
    
    
    
    def fit (self, x, y, lr, epox=1):
        self.stroki = deepcopy(range(x.shape[0]))

        
        if len(x.shape) == 1:

            for epo in range (epox):
        
                self.vixods = []
                self.vixods.append(x)
                
                
                for layer in self.layers:
                    self.vixods.append(layer.forward_lay(x))
                    x = layer.forward_lay(x)
                
                x = self.vixods[0]

                loss = ((self.vixods[-1]-y)**2).mean() 
                #print("ЛОС", loss )

                loss_dif =  2*(self.vixods[-1]-y)
                self.grad = []

                self.grad.append(loss_dif)
                # print("ГРАД", grad)


                ur = 0

                for layer in self.layers[::-1]:
                    grad_num = 0
                    num = []

                    for neuron in layer.neurons:


                        if len(layer.neurons) == 1: 
                            self.grad.append( neuron.back(self.grad[-1][grad_num] , self.vixods[-2-ur]) )
                            neuron.pri(lr)
                        else:
                            #print(grad)
                            num.append( neuron.back(self.grad[-1][grad_num], self.vixods[-2-ur]) )
                            neuron.pri(lr)


                        grad_num +=1


                    if len(num)!=0:

                        num = np.array(num)
                        self.grad.append(num.sum(axis=0))

                    ur += 1

                
        
        
        if len(x.shape) == 2:
            
            loss_history = np.array([])

            self.lr_primary = deepcopy(lr)

            for epo in range (epox):
                
                loses = []
                
                for stroka in self.stroki :

                    
                    x_n = x[stroka]

                    
                    y_n = y[stroka]

        
                    self.vixods = []
                    self.vixods.append(x_n)


                    for layer in self.layers:
                        self.vixods.append(layer.forward_lay(x_n))
                        x_n = layer.forward_lay(x_n)

                    x_n = self.vixods[0]

                    loses.append ( (self.vixods[-1]-y_n)**2  )
                    
                    if stroka+1 == x.shape[0]:
                        loss = np.mean(loses)
                        print("ЛОС", loss )
                        loss_history = np.append(loss_history, loss)
                        
                        
                    

                    loss_dif =  2*(self.vixods[-1]-y_n)
                    self.grad = []

                    self.grad.append(loss_dif)
                    # print("ГРАД", grad)


                    ur = 0

                    for layer in self.layers[::-1]:
                        grad_num = 0
                        num = []

                        for neuron in layer.neurons:


                            if len(layer.neurons) == 1: 
                                grad.append( neuron.back(self.grad[-1][grad_num] , self.vixods[-2-ur]) )
                                if stroka+1 == x.shape[0]:
                                    neuron.pri(lr)
                            else:
                                #print(grad)
                                num.append( neuron.back(self.grad[-1][grad_num], self.vixods[-2-ur]) )
                                if stroka+1 == x.shape[0]:
                                    neuron.pri(lr)


                            grad_num +=1



                        if len(num)!=0:

                            num = np.array(num)
                            self.grad.append(num.sum(axis=0))


                        ur += 1

                    #print("ГРАД", grad)

                    #for layer in self.layers[::-1]:

            
                 # небольшая оптимизация Learning Rate
                if len(loss_history)>=2:
                    if loss_history[-1]> loss_history[-2]:
                        lr=lr/10
                        print("LEARNING RATE = ",  lr) 

                    elif loss_history[-2]*0.95<loss_history[-1]: # здесь я выбрал 95%
                        lr = lr *1.1
                        print("LEARNING RATE = ", lr)
    
                        
        
        
        

    
        