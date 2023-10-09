Machine Learning Operations (MLOps)

Proyecto MLOps: Sistema de Recomendación de Películas
En este proyecto el objetivo consiste en implementar un sistema de recomendación de películas utilizando técnicas de Machine Learning, para llevar el modelo al mundo real. Trabajé en el contexto de una start-up que provee servicios de agregación de plataformas de streaming.

Descripción del Problema:
En el rol de un MLOps Engineer, nos encontramos con un modelo de recomendación de películas que ha demostrado buenas métricas en el entorno de desarrollo. Sin embargo, necesitamos llevar este modelo al mundo real y enfrentamos desafíos en la etapa de ingeniería de datos y despliegue de la API.

El conjunto de datos con el que trabajé tiene poca madurez, con datos anidados, falta de transformaciones y procesos automatizados para la actualización de nuevas películas o series. mi objetivo es realizar las transformaciones necesarias en los datos, desarrollar una API con el framework FastAPI y desplegarla para que pueda ser consumida desde la web. Además, implementaremos un sistema de recomendación basado en la similitud entre películas.

Requerimientos:
Python: 3
Bibliotecas: FastAPI, pandas,numpy, scikit-learn, etc.
Entorno virtual: (opcional)

Transformación de Datos
En esta etapa, realizamos las siguientes transformaciones en los datos:

-Desanidamos los campos anidados 
-Rellenamos los valores nulos de los campos revenue y budget con el número 0.
-Eliminamos los valores nulos
-Creamos columnas nuevas, return para calcular el retorno de inversión por ejemplo
-Eliminamos columnas no utilizadas.

Análisis Exploratorio de Datos (EDA)

Durante el análisis exploratorio de los datos, realizamos las siguientes tareas:

visión general y detallada de los datos, revelando patrones, tendencias y posibles problemas de calidad. Esto nos permite comprender mejor la estructura de los datos, identificar variables relevantes y realizar decisiones informadas en futuros análisis.

Desarrollo de la API:

Para desarrollar la API, hemos utilizado el framework FastAPI. Este framework nos permite crear rápidamente endpoints para exponer los datos  y realizar consultas a través de ellos.
Para desplegar la API, hemos utilizado el servicio Render.

Uso de la API:
La API está disponible para realizar consultas! Puedes acceder a ella a través de este enlace: " ". Utiliza los diferentes endpoints para obtener información sobre películas, actores, directores y más.
