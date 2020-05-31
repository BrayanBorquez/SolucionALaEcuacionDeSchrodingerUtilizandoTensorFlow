# SolucionALaEcuacionDeSchrodingerUtilizandoTensorFlow
Los siguientes códigos utlizan TensorFlow para predecir solución a la ecuación de Schrödinger unidimensional para dado potencial. El archivo <code>GeneradorPotencial</code> genera potenciales aleatorios, así como obtiene la solución o función de onda para estos, los potenciales generados son guardados utilizando el formato CSV. Por su parte, el archivo <code>Schrödinger</code> crea a partir de potenciales y soluciones generados anteriormente 2 capas ocultas y una de salida, así como sus respectivas ordenadas al origen las cuales nos servirán para predecir soluciones, estos de igual forma son guardados en formato CSV. El archivo <code>Solución</code> crea de igual forma 3 potenciales aleatorios, obtiene la solución de estos y por último obtiene la solución predicha por TensorFlow aplicando de manera sucesiva las capas ocultas y la de salida.
# Repaso
La idea central es a partir de ciertos potenciales y sus soluciones entrenar un modelo de Machine Learning en TensorFlow, el cual, a partir de dado potencial obtiene su solución numérica. Los potenciales generados son potenciales básicos, el primero de ellos son funciones por pasos, dichas funciones representan el efecto túnel cuántico en el cual cierta partícula puede pasar de un estado de energía a otro, el segundo son líneas rectas que cambian de pendiente y el tercero son series de Fourier aleatorias.
Se puede decir que se sabe todo acerca de la ecuación de Schrödinger y que hay muchas formas de resolverla, pero en esta se implementará una herramienta relativamente nueva la cual es TensorFlow, gracias a que podemos obtener soluciones de maneras relativamente fáciles y rápidas es posible obtener una buena cantidad de datos para entrenamiento sin utilizar tanto tiempo y gasto de cómputo.
# GeneradorPotencial
En el archivo <code>GeneradorPotencial</code> se definen dos funciones, una es para generar números aleatorios positivos y la otra para generar potenciales, esta función genera 3 tipos de potenciales distintos, por pasos, lineales con zigzag y series de Fourier, dependiendo del estilo que se introduce es el tipo de potencial que se genera. 
Dentro del cuerpo del algoritmo se definen constantes y arreglos los cuales sirven para los potenciales, se define una semilla aleatoria empezando con el 0, se define también el numero de puntos, en este caso utilizamos 128, y también el numero de potenciales, el cual es 200, mediante un ciclo generamos los 200 potenciales y con herramientas de TensorFlow obtenemos las soluciones, estas herramientas son los métodos convencionales, cada 5 potenciales sirven como validación, el programa guarda un total de 4 archivos en formato CSV, 2 corresponden a todos los potenciales y su solución y los otros 2 a los de validación así como su solución.
El algoritmo anterior se repite cambiando la semilla aleatoria dependiendo de la cantidad de potenciales que se quieran utilizar para entrenar al programa.
# Schrödinger 
El algoritmo <code>Schrödinger</code> lee los potenciales del total de semillas utilizadas, se define una constante igual a la cantidad de semillas que se leerán, por cuestiones de pruebas el algoritmo muestra 3, pero esta cantidad es muy pequeña y para obtener un buen programa es recomendable tomar más, una vez hecho esto se generan las constantes de TensorFlow como la constante de aprendizaje y la que actualizará esta constante de aprendizaje, se define una semilla aleatoria la cual puede darse cualquier valor, se define ahora dos capas ocultas y una de salida, para estas capas se definen los pesos y las ordenadas al origen. 
<pre><code>X=tf.placeholder(tf.float32)
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
</code></pre>
Se define la función costo y el optimizador el cual funciona utilizando Gradiente Descendiente. Se utiliza un ciclo de 100,000 pasos dentro del cual cada cierto paso y bajo ciertas condiciones se actualiza el radio de aprendizaje, reduciendo así la función costo, por último, se guardan los pesos y las ordenadas al origen en formato CSV.
# Solución
En este archivo se llaman los pesos y las ordenadas al origen, luego, con las funciones definidas en el algoritmo <code>GeneradorPotencial</code>  generamos 3 potenciales para poner a prueba el programa, activamos las 2 capas ocultas y las de salida utilizando como entrada uno de los 3 potenciales que se generan, se obtiene una solución predicha por TensorFlow y utilizando métodos convencionales se obtienen de igual forma la solución, con el módulo matplotlib se grafica tanto los potenciales como la solución por métodos convencionales y la solución utilizando TensorFlow y por ultimo se calcula el error porcentual promedio entre los puntos de la solución dada por TensorFlow.
<pre><code>p=2#Número de potencial que queremos analizar, este va del 0-2 
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
</code></pre>
