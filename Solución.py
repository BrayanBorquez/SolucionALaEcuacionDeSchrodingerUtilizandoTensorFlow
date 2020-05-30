import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#Utilizaremos las dos funciones utilizadas para generar potenciales de el primer algoritmo
def subexp(expon):
    return np.power(abs(np.log(np.random.uniform())),expon)

def GenerarPot(estilo,param):
    mu = 1 + bins*param 
    forxp=2.5-2*param
    escala=2.5*np.pi**2
    if estilo<2:
        dx=bins/mu
        xlist=[-dx/2]
        while xlist[-1]<bins:
            xlist.append(xlist[-1] + dx*subexp(1))
        vlist=[escala*subexp(2) for k in range(len(xlist))]
        k=0
        poten=[]
        for l in range(1,bins):
            while xlist[k+1]<l:
                k+=1
            if estilo==0:
                poten.append(vlist[k])
            else:
                poten.append(vlist[k]+(vlist[k+1]-vlist[k])*(l-xlist[k])/(xlist[k+1]-xlist[k]))
    else:
        sincoef=[(2*np.random.randint(2)-1)*escala*subexp(2)/np.power(k,forxp) for k in range(1,bins//2)]
        coscoef=[(2*np.random.randint(2)-1)*escala*subexp(2)/np.power(k,forxp) for k in range(1,bins//2)]
        zercoef=escala*subexp(2)
        poten=np.maximum(np.add(np.add(np.matmul(sincoef,sinval),np.matmul(coscoef,cosval)),zercoef),0).tolist()
    return poten
#Creamos los arregos de las ordenadas al origen y los pesos
B1=[]
B2=[]
B3=[]
W1=[]
W2=[]
W3=[]
#Abrimos las ordeandas al origen y los pesos los cuales serán guardados en los arreglos anteriores
with open('B1.csv', 'r') as csvfile:
    flurg=csv.reader(csvfile)
    for row in flurg:
        B1.append([float(num) for num in row])
with open('B2.csv', 'r') as csvfile:
    flurg=csv.reader(csvfile)
    for row in flurg:
        B2.append([float(num) for num in row])
with open('B3.csv', 'r') as csvfile:
    flurg=csv.reader(csvfile)
    for row in flurg:
        B3.append([float(num) for num in row])
with open('W1.csv', 'r') as csvfile:
    flurg=csv.reader(csvfile)
    for row in flurg:
        W1.append([float(num) for num in row])
with open('W2.csv', 'r') as csvfile:
    flurg=csv.reader(csvfile)
    for row in flurg:
        W2.append([float(num) for num in row])
with open('W3.csv', 'r') as csvfile:
    flurg=csv.reader(csvfile)
    for row in flurg:
        W3.append([float(num) for num in row])
seedmax = 20 #Número de semillas que se desean abrir, en nuestro caso son un total de 20 las cuales
#van de 0-19
seed=77#Semilla aleatoria
np.random.seed(seed)
tf.set_random_seed(seed)
#Creamos dos arreglos los cuales serán para introducir los vectores
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
#Función softplus que se activan una a una utilizando primero los potenciales y luego las capas sucesivas
L1 = tf.nn.softplus(tf.matmul(X, W1) + B1)
L2 = tf.nn.softplus(tf.matmul(L1, W2) + B2)
L3 = tf.nn.softplus(tf.matmul(L2, W3) + B3)
np.random.seed(seed)#Inicializamos la semilla aleatoria
bins=128 #El número de columnas guardadas son bins-1 ya que la primera y la última son 0
npots=1000 #Número de potenciales
#Las siguientes líneas corresponden a constantes y arreglos para potenciales
sinval = np.sin([[np.pi*i*j/bins for i in range(1,bins)] for j in range(1,bins//2)]) #Estas dos lineas corresponden a los arreglos que contienen 
cosval = np.cos([[np.pi*i*j/bins for i in range(1,bins)] for j in range(1,bins//2)]) #los valores de los senos y cosenos en series de fourier
sqrt2=np.sqrt(2)
defgrdstate=tf.constant([sqrt2*np.sin(i*np.pi/bins) for i in range(1,bins)])
psi=tf.Variable(defgrdstate)
zerotens=tf.zeros([1])
psil=tf.concat([psi[1:],zerotens],0)
psir=tf.concat([zerotens,psi[:-1]],0)
renorm = tf.assign(psi,tf.divide(psi,tf.sqrt(tf.reduce_mean(tf.square(psi)))))
optimzi = tf.train.GradientDescentOptimizer(0.0625/bins)
reinit = tf.assign(psi,defgrdstate)
init = tf.global_variables_initializer()
#Creamos los arreglos que contendrán los datos
potentials = []
wavefuncs = []
#Inicializamos sesión en TensorFlow
sess = tf.Session()
sess.run(init)
#Aquí se resuelven los potenciales, como se puede ver en el contador solo
#3 potenciales se resuelven los cuales queremos obtener su solución
#analitíca con metodos convencionales para luego comparar con la solución
#dada por TensorFlow
for i in range(npots-3,npots):
    for j in range(3):
        vofx = GenerarPot(j,(1.*i)/npots)
        energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),tf.add(vofx,1.*bins*bins)),
                                            tf.multiply(tf.multiply(tf.add(psil,psir),psi),0.5*bins*bins)))
        training = optimzi.minimize(energy)
        sess.run(reinit)
        for t in range(20000):
            sess.run(training)
            sess.run(renorm)
        potentials.append(vofx)
        wavefuncs.append(sess.run(psi).tolist())
p=2#Número de potencial que queremos analizar, este va del 0-2 
predicted=sess.run(L3,feed_dict={X: [potentials[p]]})[0]#Aquí se predice la solución
#En las siguientes líneas hacemos la gráfica de los potenciales 
plt.plot(predicted)
plt.plot([potentials[p][i]/max(potentials[p]) for i in range(bins - 1)])
plt.plot(wavefuncs[p])
plt.legend(["Predicted solution","Potential","Real Solution"])
plt.xlabel("x")
plt.ylabel("Energy")
plt.show()
#Calculo del error promedio portentual 
error=0
for i in range(127):
    error+=abs((predicted[i]-wavefuncs[p][i])/wavefuncs[p][i])*100
error=error/128
print(error)