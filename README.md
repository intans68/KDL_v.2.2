# KDL_v.2.2
Library for building and training fully connected neural networks. With backpropagation implementation. Supports linear and sigmoid activation functions. Automatic linear optimizer added. Currently, only the quadratic loss function is supported!!!

#pip install numpy

m = Model()
m.add_input_layer(25,12, activation = "Sigmoid")
m.add_layer(8,activation = "Sigmoid")
m.add_layer(1)

m.fit(x,y, lr=0.0001,epox =1000)

Y = m.forward(X)
