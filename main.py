from fastapi import FastAPI
import pandas as pd
import uvicorn

df = pd.read_csv('pelis.csv')

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

app = FastAPI()

meses = df['release_date'].dt.month

@app.get("/cantidad_filmaciones_mes/")
def cantidad_filmaciones_mes(mes: str):
    data = df

    meses = {
        'enero': 1,
        'febrero': 2,
        'marzo': 3,
        'abril': 4,
        'mayo': 5,
        'junio': 6,
        'julio': 7,
        'agosto': 8,
        'septiembre': 9,
        'octubre': 10,
        'noviembre': 11,
        'diciembre': 12
    }

    num_mes = meses.get(mes.lower())

    if num_mes is None:
        raise ValueError('Nombre de mes inválido')

    filmaciones_mes = data[data['release_date'].dt.month == num_mes]
    cantidad_filmaciones = len(filmaciones_mes)

    return {"mensaje": f"Se estrenaron {cantidad_filmaciones} filmaciones el mes {mes.capitalize()}."}

#2

dias_semana = {
    'lunes': 'Monday',
    'martes': 'Tuesday',
    'miércoles': 'Wednesday',
    'jueves': 'Thursday',
    'viernes': 'Friday',
    'sábado': 'Saturday',
    'domingo': 'Sunday' 
}

@app.get('/cantidad_filmaciones_dia/')
def cantidad_filmaciones_dia(Dia: str):
    # Obtener el nombre del día de la semana en inglés
    dia_en = dias_semana.get(Dia.lower())
    
    if not dia_en:
        raise ValueError('Nombre de día inválido')
    # Filtrar las películas que fueron estrenadas en el día consultado
    filmaciones_dia = df[df['release_date'].dt.day_name() == dia_en]
    # Obtener la cantidad de películas encontradas
    cantidad_filmaciones = len(filmaciones_dia)
    # Retornar el mensaje con la cantidad de películas encontradas en el día consultado en español
    return {"mensaje": f"Se estrenaron {cantidad_filmaciones} películas en el día {Dia}."}

#3

@app.get('/score_titulo/')
def score_titulo(titulo_de_la_filmacion: str):
    # Filtrar la película por título
    pelicula = df[df['title'] == titulo_de_la_filmacion]
    
    if len(pelicula) == 0:
        raise ValueError('Película no encontrada')
    # Obtener los datos de título, año de estreno y score
    titulo = pelicula['title'].iloc[0]
    año_estreno = pelicula['release_year'].iloc[0]
    score = pelicula['vote_average'].iloc[0]
    # Retornar el mensaje con los datos de la película
    return {"mensaje": f"La película {titulo} fue estrenada en el año {año_estreno} con una puntuación de {score}."}

#4

@app.get('/votos_titulo/')
def votos_titulo(titulo: str):
    # Filtrar la película por título
    pelicula = df[df['title'] == titulo]
    
    if pelicula.empty:
        return {"mensaje": f"No se encontró la película '{titulo}' en la base de datos."}
    # Obtener la cantidad de votos y el valor promedio de las votaciones
    votos = pelicula['vote_count'].iloc[0]
    promedio_votos = pelicula['vote_average'].iloc[0]
    
    if votos < 2000:
        return {"mensaje": f"La película '{titulo}' no cumple con la condición de tener al menos 2000 valoraciones."}
    # Retornar el mensaje con la información de la película
    return {"mensaje": f"La película '{titulo}' fue estrenada en el año {pelicula['release_year'].iloc[0]}. "
                       f"La misma cuenta con un total de {votos} valoraciones, con un promedio de {promedio_votos}."}

#5

@app.get('/get_actor/')
def get_actor(nombre_actor: str):
    # Filtrar las películas en las que el actor es parte del elenco
    peliculas_actor = df[df['cast'].str.contains(nombre_actor, na=False)]
    
    if len(peliculas_actor) == 0:
        raise ValueError('Actor no encontrado')
    # Obtener la cantidad de películas en las que ha participado el actor
    cantidad_peliculas = len(peliculas_actor)
    # Calcular el retorno total del actor sumando los retornos de todas las películas
    retorno_total = peliculas_actor['return'].sum()
    # Calcular el promedio de retorno
    promedio_retorno = round(retorno_total / cantidad_peliculas,2)
    # Retornar el mensaje con la información del actor
    mensaje = f"El actor {nombre_actor} ha participado en {cantidad_peliculas} filmaciones. Ha obtenido un retorno total de {retorno_total} con un promedio de {promedio_retorno} por filmación."
    
    return {"mensaje": mensaje}

#6
import re

def extract_director(crew):
    matches = re.findall(r"Director:\s*([^\n\r,]+)", crew)
    if matches:
        return matches[0]
    else:
        return None


df['crew'] = df['crew'].astype(str)
df['directors'] = df['crew'].apply(extract_director)

@app.get("/get_director")
def get_director_endpoint(nombre_director: str):
    director = df[df['directors'] == nombre_director]
    
    if director.empty:
        return {"mensaje": "El director no se encuentra en el dataset."}
    
    exito_director = director['return'].sum()
    peliculas_director = director[['title', 'release_date', 'return', 'budget', 'revenue']].to_dict(orient='records')
    
    return {"exito_director": exito_director, "peliculas_director": peliculas_director}



# MODELO ML

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.extmath import randomized_svd
import numpy as np

# El Vectorizador TfidfVectorizer con parámetros de reduccion procesamiento
df['genres'].fillna('', inplace=True)

vectorizar = TfidfVectorizer(min_df=10, max_df=0.5, ngram_range=(1,2))

# Vectorizamos, ajustamos y transformamos el texto de la columna "title" del DataFrame
X = vectorizar.fit_transform(df['genres'])

# Calcular la matriz de similitud de coseno con una matriz reducida de 5000
similarity_matrix = cosine_similarity(X[:2500,:])

# Obtenemos la descomposición en valores singulares aleatoria de la matriz de similitud de coseno con 10 componentes
n_components = 10
U, Sigma, VT = randomized_svd(similarity_matrix, n_components=n_components)

# Construir la matriz reducida de similitud de coseno
reduced_similarity_matrix = U.dot(np.diag(Sigma)).dot(VT)
@app.get('/recomendacion/')
def recomendacion(titulo:str):
    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''
    titulo = titulo.title()
    #Ubicamos el indice del titulo pasado como parametro en la columna 'title' del dts user_item
    indice = np.where(df['title'] == titulo)[0][0]
    #Vemos los indices de aquellas puntuaciones y caracteristicas similares hacia el titulo 
    puntuaciones_similitud = reduced_similarity_matrix[indice,:]
    # Se ordena los indicies de mayor a menor
    puntuacion_ordenada = np.argsort(puntuaciones_similitud)[::-1]
    # Que solo 5 nos indique 
    top_indices = puntuacion_ordenada[:5]
    
    return df.loc[top_indices, 'title'].tolist()



def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()