#Este algoritmo lee los potenciales generados del GeneradorPotencial.py y hace una red neuronal
#con dos capas ocultas, ademas visualiza y guarda los resultados
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as mp

bins=128
seedmax=3 #Abre la cantidad de semillas que nosotros colocamos
#Definimos los vecotres
trainx = []
trainy = []
validx = []
validy = []
#Ahora procedemos a leer los archivos, el contador va hasta la cantidad de archivos que queramos leer
for i in range(seedmax):
    with open('test_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainx.append([float(num) for num in row])
    with open('test_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainy.append([float(num) for num in row])
    with open('valid_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validx.append([float(num) for num in row])
    with open('valid_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validy.append([float(num) for num in row])
#Inicializamos una semilla aleatoria en numpy y tensorflow
seed=31
np.random.seed(seed)
tf.set_random_seed(seed)
#Definimos las siguientes cantidades
startrate=0.125
gs=0
gslist=[1,1,2,3,10,20,40,100,200,10000]
ic=0
learnrate=tf.Variable(startrate, trainable=False)
updatelearnrate=tf.assign(learnrate,tf.multiply(learnrate,0.75))
#Ahora definimos las redes neuronales
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
#Primera capa oculta
W1=tf.Variable(tf.random_uniform([bins-1, bins-1], -1/bins, 1/bins))
B1=tf.Variable(tf.random_uniform([bins-1], -1, 1))
L1=tf.nn.softplus(tf.matmul(X, W1) + B1)
#Segunda capa oculta
W2=tf.Variable(tf.random_uniform([bins-1, bins-1], -1./bins, 1./bins))
B2=tf.Variable(tf.random_uniform([bins-1], -1., 1.))
L2=tf.nn.softplus(tf.matmul(L1, W2) + B2)
#Capa de salida
W3=tf.Variable(tf.random_uniform([bins-1, bins-1], -1./bins, 1./bins))
B3=tf.Variable(tf.random_uniform([bins-1], -1., 1.))
L3=tf.nn.softplus(tf.matmul(L2, W3) + B3)
#Función costo
costfunc = tf.reduce_mean(tf.square(tf.subtract(L3,Y)))
optimizer = tf.train.GradientDescentOptimizer(learnrate)
trainstep = optimizer.minimize(costfunc)
#Inicializamos en tensorflow
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#Aquí inicializa el proceso
for step in range(100000):
    if step % 150 == 0:
        if ic == gslist[gs]:
            gs = gs + 1
            ic = 1
            sess.run(updatelearnrate)
        else:
            ic = ic + 1
    if step %100 == 0:
        print(step, 'Train loss: ',sess.run(costfunc,feed_dict={X: trainx, Y: trainy}), 'Valid loss: ',sess.run(costfunc,feed_dict={X: validx, Y: validy}))
    sess.run(trainstep, feed_dict={X: trainx, Y: trainy})
#Por último, este guarda los resultados obtenidos
with open('W1.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(sess.run(W1).tolist())
with open('W2.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(sess.run(W2).tolist())
with open('W3.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(sess.run(W3).tolist())
with open('B1.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows([sess.run(B1).tolist()])
with open('B2.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows([sess.run(B2).tolist()])
with open('B3.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows([sess.run(B3).tolist()])