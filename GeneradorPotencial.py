import csv
import numpy as np
import tensorflow as tf

def subexp(expon):
    return np.power(abs(np.log(np.random.uniform())),expon)

def GenerarPot(estilo,param):#0 es  por pasos, 1 es lineal, 2 es fourier, y de 1-2 es en picos
    mu = 1 + bins*param #Número de puntos de saltos para los pasos 1 y 2
    forxp=2.5-2*param
    escala=2.5*np.pi**2#Escala de energía
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

seed=19 #El número de semilla
np.random.seed(seed)#Inicializamos la semilla aleatoria
bins=128 #El número de columnas guardadas son bins-1 ya que la primera y la última son 0
npots=20 #Número de potenciales
validnth=5 #Cada número de estos las funciones son guardadas como validación
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
validpots = []
wavefuncs = []
validfuncs = []

sess = tf.Session()
sess.run(init)
#Aquí se obtiene la solución al potencial
for i in range(npots):#Inicializamos el conteo
    if i%2 == 0:
        print(str((100.*i)/npots) + "%","complete")#Esta parte es para ver el avance :D
    for j in range(3):
        vofx = GenerarPot(j,(1.*i)/npots)
        energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),tf.add(vofx,1.*bins*bins)),
                                            tf.multiply(tf.multiply(tf.add(psil,psir),psi),0.5*bins*bins)))
        training = optimzi.minimize(energy)
        sess.run(reinit)
        for t in range(20000):
            sess.run(training)
            sess.run(renorm)
        if i%validnth == 0:
            validpots.append(vofx)
            validfuncs.append(sess.run(psi).tolist())
        else:
            potentials.append(vofx)
            wavefuncs.append(sess.run(psi).tolist())

with open('test_pots'+str(seed)+'.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(potentials)
with open('valid_pots'+str(seed)+'.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(validpots)
with open('test_out'+str(seed)+'.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(wavefuncs)
with open('valid_out'+str(seed)+'.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(validfuncs)
print('Output complete')