# Sobre la clasificación

- Tecnica utilizada en minería de datos
- Viene del área del Machine Learning
- Metodo de **Aprendizaje Supervisado"

# Objetivo

Asignar objetos no vistos anteriormente a una clase dentro de un conjunto determinado de clases con la mayor precisión posible.

En otras palabras, mapear un set x a una clase y, conjunto de atributos (x) -> Modelo de Clasificación -> etiquetas (y)
# Enfoque

Dada una colección de registros (conjunto de entrenamiento):
    - Cada registro tiene un conjunto de atributos
    - Uno de los atributos es la clase (etiqueta) que debe predecirse

Aprender un modelo para el atributo de clase como función de los otros atributos

# Variantes

- Clasificación binaria (fraude/no fraude o verdadero/falso)
- Clasificación multi-clase (bajo, medio, alto)
- clasificación multi-etiqueta (más de una clase por registrom por ejemplo: intereses del usuario)

# Como?

Para lograr el objetivo se entrena un algoritmo de clasificación mostrándole datos etiquetados para que aprenda. 
Así se obtiene un modelo entrenado capaz de asignar etiquetas a atributos.

En machine learning, a la clasificación se le considera como un enfoque de aprendizaje supervisado, pues requiere de datos etiquetados.

# Ejemplos de clasificación

- Evaluación de reisgo crediticio
- Marketing (recomendaciones y deteción de clientes objetivo)
- Detección de SPAM
- Detección de sentimiento
- Identificación de células tumorales

# Componentes principales

Conjunto de entrenamiento -> Input para entrenar
Algoritmo de clasificación -> Para obtener el modelo de clasificación
Conjunto de validación -> Para probar el modelo de clasificación

# En resumen

Dada una colección de objetos (set de entrenamiento), cada record contiene un set de atributos, uno de los cuales es su clase.

Encontrar un modelo para el atributo de clase, en base a los otros atributos

Meta: records nuevos deben ser asignados correctamente a su clase, un set de evaluación se utiliza para medir la exactitud del modelo.

# Notas!

Es mejor para datos binarios y nominales

No es tan bueno para datos ordinales ya los algoritmos no consideran relaciones de orden entre clase

# Técnicas de clasificación

- Basados en Árboles de Decisión
- Métodos basados en Reglas
- Razonamiento en base a memoria
- Redes neuronales
- Naive Bayes y Redes de Soporte Bayesianas
- Support Vector Machines

# Clasificación en ML vs DM

Cuando se hace clasificación en machine learning queremos automatizar una tarea

Cuando hacemos clasificación en data mining queremos encontrar un patrón en los datos, queremos entender cómo se relaciona x con y por medio de un modelo predictivo.

# Cómo saber si un modelo es bueno o no

Lo más importante es la capacidad predictiva del modelo

Pero hacer predicciones correctas sobre los datos de entrenamiento no es suficiente para determinar la capacidad predictiva

El modelo construido debe generalizar, es decir, debe ser capaz de realizar predicciónes correctas en datos distintos a los datos de entrenamiento

Otros factores importantes son la interpretabilidad y la eficiencia

Resumimos la capacidad predictiva de un modelo mediante métricas de desempeño

Las metricas se calculan contrastando los valores predichos versus los valores reales de la variable objetivo

Este se hace con datos no usados durante el entrenamiento

Diseñamos experimentos en que comparamos las métricas de desempeño para varios modelos distintos y nos quedamos con el mejor.

# Matriz de Confusión

|            |          Clase Predicha     |
|------------|-----------------------------|
| Clase Real |         |clase =+ |clase =- | 
|            |clase =+ |  a = TP |  b = FN |
|            |clase =- |  c = FP |  d = TN |


[!NOTE]
TN: True Positive y TN: True Negative es cuando el modelo clasifica correctamente un positivo o negativo como corresponda.
FN: False negative, el modelo predijo negativa una clase pero en realidad era positiva, ej: el modelo dice que NO tiene covid pero SÍ tiene
FP: False positive, el modelo predijo positiva una clase pero en realidad era negativa, ej: el modelo dice que SÍ tiene covid pero NO tiene

# Accuracy (Exactitud)

Qué proporción de las predicciónes son correctas?

ACCURACY = (TP + TN)/(all predictions)
Error rate = 1 - Accuracy

## Limitaciónes

- Accuracy no es buen amétrica cuando tenemos clases desbalanceadas.

ej: tenemos un conjunto grande de clase 1, pequeño de clase 2, el modelo marca todos clase 1, la acc es alta y el modelo es malo.

# Precision

¿Qué proporción de predicciónes positivas son correctas?

PRECISION = TP/(TP + FP)

Se enfoca en los errores false positive, ej de uso: detección fina de una enfermedad mortal: no queremos dar ningún tratamiento pesado sin necesidad.


# Recall

¿Qué proporción de los positivos reales se clasificarion como positivos?

RECALL = TP(TP+FN)

Se enfoca en los errores false negative, ej de uso: detección masiva de una enfermedad contagiosa (covid): no queremos perder a ningun paciente

# F-measure

Combina Precision y Recall mediante un promedio armónico ponderado

Generalmente ocupamos la F1 measure

F1 = 2PR/(P+R)

F1 ignora TN

# Ejercicio

Considere 286 mujeres: 201 no tienen reincidencia de cancer después de 5 años, 85 sí tienen. Compare los modelos:

M1: "Todas reinciden"
M2: "Ninguna reincide"


Matriz de M1: TP=85, FP=201, FN=0, TN=0
Matriz de M2: TP=0, FP=0, FN=85, TN=201

AccM1: 85/286=29%, Pecision: 85/286: 29%, Recall=1, F1=0.6/(0.6+1)=0.37
AccM2: 201/286=70%, Precision: undef, Recall=0%, F1=undef

# Micro-Averaging vs Macro-Averaging

Si tenemos más de una clase, ¿Cómo combinamos múltiples métricas de desempeño en un solo valor>

- Macro-Averaging: Computar métrica para cada clase y luego promediar
- Micro-Averaging: Crear matriz de confusión binaria para cada clase, combinar las matrices y luego evaluar

- micro-averaging son dominados por las clases más frecuentes
- macro-averaging pueden sobre-representar a clases minoritarias

# Evaluación de desempeño del modelo

El desempeño de un modelo puede depender de factores diferentes al algoritmo de aprendizaje

- Distribución de las clases
- Costo de clasificaciónes erróneas
- Tamaño de los datos de entrenamiento y test

## Métodos para evaluar el desempeño de un modelo

La idea es estimar la capacidad de generalización del modelo, evaluándolo en datos distintos a los de entrenamiento.

- Holdout
- Random subsampling
- Cross validation

## Holdout

Particionamos los datos etiquetados en una partición de training y otra de testing, usualmente es trainting(66%)/testing(33%)

Limitaciones:

- La evaluación puede variar mucho según las particiones escogidas
- Training muy pequeño -> modelo sesgado
- Testing muy pequeño -> acc poco confiable

## Random subsampling

Se itera sobre holdout varias veces, permite obtener una distribución de los errores o medidas de desempeño

Limitaciones: Puede que algunos datos nunca se usen para entrenar, puede que otros nunca se usen para evaluar.

## Cross validation

De particiona el dataset en k conjuntos disjuntos o folds (manteniendo distribución de clases en cada fold)

Para cada partición i:

- Juntar todas las k-1 particiones restantes y entrenar el modelo sobre esos datos
- Evaluar el modelo en la partición i

El error total se calcula sumando los errores hechos en cada fold de testing

Estamos entrenando el modelo k veces

Variante: leave-one-out (k=n)

# Problemas prácticos en la clasificación

Errores de entrenamiento (malos resultados sobre los datos de entrenamiento): esto ocurre cuando el clasificador no tiene la capacidad de aprender el patrón

Errores de generalización (malos resultados sobre los datos nuevos): Esto ocurre cuando el modelo se hace demasiado específico a los datos de entrenamiento

Overfitting: error de generalización
Underfitting: error de entrenamiento

Overfitting refleja un modelo más complejo de lo necesario, el error de entrenamiento no es un indicador confiable de cómo se desempeñaría el modelo sobre datos nuevos.

# Curva ROC (Reciever Operating Characteristic Curve)

De manera similar al trade-off entre precision y recall, tambien existe un trade-off entre la tasa de verdaderos positivos y la tasa de falsos positivos:

TP Rate: TP/(TP+FN)
FP Rate: FP/(FP+TN)

La curva ROC se construye graficando TP Rate vs FP Rate para varios umbrales de clasificación de un clasificador probabilístico (ej: regresión logística, naive Bayes)

Entre mayor sea el area bajo la curva mejor es el modelo.

El area bajo la curva ROC se conoce como AUC y es una métrica ampliamente usada


