
# GEMA
Genetic Electronic Music Assistant (GEMA)

GEMA es un programa en Python (3.8) que mediante el uso de inteligencia artificial (algoritmos genéticos), genera melodías musicales aleatorias dado los parámetros (o generados aleatoriamente); escala musical, duración, tempo y desviación máxima de la raíz. 

GEMA sugiere distintas melodías basadas en la escala musical, y esta muta entre generaciones dependiendo del feedback (calificación del 1-10), el proceso se repite sucesivamente.

# Sobre esta rama
Esta rama contiene la implementación de progresiones ármonicas, donde una progresión de acordes sigue las melodias generadas.

## Directorios

|Directorio| Description |
|--|--|
| algorithms | Contiene los algoritmos usados en el programa. |
| classes | Contiene la definicion de las clases (con sus respectivos metodos).
| out | El folder objetivo donde se guardara el output (archivos midi).
| utility | Contiene funciones utiles para solicitar inputs del usuario.