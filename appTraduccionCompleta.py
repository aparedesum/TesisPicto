import json
import warnings
import re
import os

warnings.filterwarnings('ignore')

from pprint import pprint

from ahocorasick import Automaton

import spacy

# Load the Spanish language model
nlp = spacy.load("es_dep_news_trf")

class OracionTraducida:
    def __init__(self, id, oracion, oracion_lematizada, palabras_traducidas, oracion_traducida_basica):
        self.id = id
        self.oracion = oracion
        self.oracion_lematizada = oracion_lematizada
        self.palabras_traducidas = palabras_traducidas
        self.oracion_traducida_basica = oracion_traducida_basica

class PalabraTraducida: 
    def __init__(self, key, palabraInfo, pictogramas,usaLema):
        self.key = key
        self.palabraInfo = palabraInfo
        self.usaLema = usaLema
        pictogramas = list(pictogramas)
        if (pictogramas != None and len(pictogramas) > 0):
            self.pictogramas = pictogramas
        else:
            self.pictogramas = []
        
class PictogramaInfo:
    def __init__(self, palabra, pos, tag, pictograma, isSimple):
        self.palabra = palabra
        self.pos = pos
        self.tag = tag
        self.keyword = pictograma.keyword
        self.id = pictograma.id
        self.synsets = pictograma.synsets
        self.plural = pictograma.plural
        self.isSimple = isSimple

class Pictograma:
    def __init__(self, keyword, id, plural, synsets):
        self.keyword = keyword.lower() if keyword is not None else None
        self.id = id
        self.synsets = synsets
        self.plural = plural.lower() if plural is not None else None
        

def cargar_pictogramas():
    nombre_archivo = 'pictogramasArasaac.json'
    try:
        with open(nombre_archivo, 'r', encoding='utf-8') as archivo_json:
            # Carga el contenido del archivo JSON en un diccionario
            datos_dict = json.load(archivo_json)
            lista_pictogramas = []
            for element in datos_dict:
                keywords = element.get('keywords', [])
                for keyword in keywords:
                    lista_pictogramas.append(Pictograma(keyword.get('keyword', None) , element.get("_id", None), keyword.get('plural', None), element.get("synsets", None)))

            diccionario_pictogramas_simples = {}
            diccionario_pictogramas_compuestos = {}

            for pictograma in lista_pictogramas:
                isSimple = pictograma.keyword.strip().count(" ") == 0
                if isSimple:
                    if pictograma.keyword not in diccionario_pictogramas_simples:
                        diccionario_pictogramas_simples[pictograma.keyword] = []
                    diccionario_pictogramas_simples[pictograma.keyword].append(pictograma)
                else:
                    if pictograma.keyword not in diccionario_pictogramas_compuestos:
                        diccionario_pictogramas_compuestos[pictograma.keyword] = []
                    diccionario_pictogramas_compuestos[pictograma.keyword].append(pictograma)
            print('Diccionario cargado')
            return (diccionario_pictogramas_simples, diccionario_pictogramas_compuestos)
    except FileNotFoundError:
        print(f'El archivo {nombre_archivo} no se encuentra.')
    except json.JSONDecodeError as e:
        print(f'Error al decodificar el archivo JSON: {e}')
    except Exception as e:
        print(f'Error durante la carga del archivo JSON: {e}')

def normalize_translation(translation):
    # Separar la cadena de entrada en segmentos con y sin pictogramas
    segments = re.split(r"(#\d+#[^#]*#)", translation)
    result = []
    
    for segment in segments:
        # Si el segmento corresponde a un pictograma, procesarlo
        if re.match(r"^#\d+#[^#]*#$", segment):
            match = re.match(r"#(\d+)#([^#]*)#", segment)
            if match:
                pictogram_id, text = match.groups()
                # Crear un token normalizado
                normalized_token = f"pict_{pictogram_id}_{text.strip().replace(' ', '_')}"
                result.append(normalized_token)
        else:
            if segment.strip() != "":
                # Añadir los segmentos no pictograma directamente al resultado
                result.append(segment.strip())

    # Unir todos los segmentos procesados
    return " ".join(result)

def tokenizar_lematizar_oraciones():
    try:
        with open("D:\\Tesis\\Tesis\\ElaboracionDataset\\repositorio\\oraciones_lectura_simples_no_duplicados.txt", 'r', encoding='utf-8') as archivo_texto:
            lineas = archivo_texto.readlines()
            lineas = [linea.strip() for linea in lineas]
            oraciones_lematizadas = []
            i = 0
            for linea in nlp.pipe(lineas):
                tokens_por_oracion = []
                lemmas = []
                for token in linea:
                    lema_values = token.lemma_.split()
                    lema = lema_values[0]

                    properties = { "texto" : token.text, "lema": lema, "pos": token.pos_, "tag": token.tag_}
                    if(token.pos_ == "DET"):
                        lemmas.append(token.text)
                    else:
                        lemmas.append(lema)
                    tokens_por_oracion.append(properties)
                oraciones_lematizadas.append({"oracion": lineas[i].strip(), "oracion_lematizada": " ".join(lemmas),"tokens": tokens_por_oracion})
                i = i + 1
            return oraciones_lematizadas
    except FileNotFoundError:
        print(f'El archivo no se encuentra.')
    except Exception as e:
        print(f'Error durante la tokenización: {e}')

def traducir_oraciones(oraciones_tokenizadas, dictionario_pictogramas_simples, dictionario_pictogramas_compuestos):
    print(f'Traduciendo Oraciones.')
    oraciones_traducidas = []
    dataset_basico = []
    existe_oracion = set()
    i = 1
    j = 1
    for oracion_tokenizada in oraciones_tokenizadas:
        palabras_traducidas = []
        key_simple = 0
        frase_agregada = set()
        oracion_traducida_basica = ""
        buscar_pictogramas_simples(dictionario_pictogramas_simples, oracion_tokenizada, palabras_traducidas, key_simple)
        buscar_pictogramas_compuestos(dictionario_pictogramas_compuestos, oracion_tokenizada, palabras_traducidas, frase_agregada)

        for palabra_traducida in palabras_traducidas:
            palabra_info = palabra_traducida.palabraInfo["texto"]
            if palabra_traducida.pictogramas:
                oracion_traducida_basica += str(f"#{palabra_traducida.pictogramas[0].id}#{palabra_traducida.pictogramas[0].keyword}# ") 
            else:
                oracion_traducida_basica += f"{palabra_info} "
        
        oracion_traducida = OracionTraducida(id = i, oracion = oracion_tokenizada["oracion"], oracion_lematizada = oracion_tokenizada["oracion_lematizada"],palabras_traducidas = palabras_traducidas, oracion_traducida_basica=oracion_traducida_basica.strip())
        oraciones_traducidas.append(oracion_traducida)

        if oracion_tokenizada["oracion"] not in existe_oracion:
            existe_oracion.add(oracion_tokenizada["oracion"])
            dataset_basico.append({"id": j, "oracion" : oracion_tokenizada["oracion"] , "traduccion": normalize_translation(oracion_traducida_basica.strip())})
            j = j + 1
        else:
            print(f"existe: {oracion_tokenizada['oracion']}")
        i = i + 1
    print(f'Fin traducción.')
    guardar_resultados(dataset_basico)
    return oraciones_traducidas

def guardar_resultados(dataset_basico):
    dataset_basico_json = json.dumps(dataset_basico,ensure_ascii=False,indent=2)

    with open('oraciones_lectura_simples.txt', 'w', encoding='utf-8') as file:
        for ora in dataset_basico:
            file.write(ora["oracion"] + '\n')

    with open('oraciones_lectura_simples_traducidas.json', 'w', encoding='utf-8') as file:
        file.write(dataset_basico_json)

def buscar_pictogramas_compuestos(dictionario_pictogramas_compuestos, oracion_tokenizada, palabras_traducidas, frase_agregada):
    for end_index, (pattern_idx, found) in automaton.iter(oracion_tokenizada["oracion"]):
        frase_agregada.add(found)

        numero_palabras = len(found.split())
        indice = oracion_tokenizada["oracion"].find(found)
        segmento_previo = oracion_tokenizada["oracion"][:indice]

        palabras_y_puntuacion = re.split(r'([\s,.:;"\'()¿?¡!=-]+)', segmento_previo)

        palabras_y_puntuacion = [p for p in palabras_y_puntuacion if p.strip()]

        numero_palabras_antes = len(palabras_y_puntuacion)
        newKey = palabras_traducidas[numero_palabras_antes].key

        del palabras_traducidas[numero_palabras_antes:numero_palabras_antes+numero_palabras]

        pictograma_compuesta = dictionario_pictogramas_compuestos[found]

        pictogramasInfo = map(lambda picto: PictogramaInfo(found, None, None, picto, False), pictograma_compuesta)
        properties = { "texto" : found, "lema": found, "pos": None, "tag": None}
        palabras_traducidas.insert(numero_palabras_antes, PalabraTraducida(newKey, properties, pictogramasInfo, True))

    for end_index, (pattern_idx, found) in automaton.iter(oracion_tokenizada["oracion_lematizada"]):
        if found not in frase_agregada:
            cantidad_inicial = re.split(r'([\s,.:;"\'()¿?¡!=-]+)', oracion_tokenizada["oracion_lematizada"])
            cantidad_inicial = [p for p in cantidad_inicial if p.strip()]
            if len(cantidad_inicial) != len(palabras_traducidas):
                break

            pictograma_compuesta = dictionario_pictogramas_compuestos[found]

            numero_palabras = len(found.split())
            indice = oracion_tokenizada["oracion_lematizada"].find(found)
            segmento_previo = oracion_tokenizada["oracion_lematizada"][:indice]

            palabras_y_puntuacion = re.split(r'([\s,.:;"\'()¿?¡!=-]+)', segmento_previo)
            palabras_y_puntuacion = [p for p in palabras_y_puntuacion if p.strip()]

            numero_palabras_antes = len(palabras_y_puntuacion)
            newKey = palabras_traducidas[numero_palabras_antes].key

            del palabras_traducidas[numero_palabras_antes:numero_palabras_antes+numero_palabras]

            pictogramasInfo = map(lambda picto: PictogramaInfo(found, None, None, picto, False), pictograma_compuesta)
            properties = { "texto" : found, "lema": found, "pos": None, "tag": None}

            palabras_traducidas.insert(numero_palabras_antes, PalabraTraducida(newKey, properties, pictogramasInfo, True))

def buscar_pictogramas_simples(dictionario_pictogramas_simples, oracion_tokenizada, palabras_traducidas, key_simple):
    for token in oracion_tokenizada["tokens"]:
        pictogramas = dictionario_pictogramas_simples.get(token["texto"].lower(), {})
        if(pictogramas):
            pictogramasInfo = map(lambda picto: PictogramaInfo(token["texto"], token["pos"], token["tag"], picto, True), pictogramas)
            palabras_traducidas.append(PalabraTraducida(key_simple, token, pictogramasInfo, False))
        else:
                # Tratamos de buscar en el dictionary la palabra lematizada
            pictogramas = dictionario_pictogramas_simples.get(token["lema"].lower(), {})
            if(pictogramas):
                pictogramasInfo = map(lambda picto: PictogramaInfo(token["texto"], token["pos"], token["tag"], picto, True), pictogramas)
                palabras_traducidas.append(PalabraTraducida(key_simple, token, pictogramasInfo, True))
            else:
                palabras_traducidas.append(PalabraTraducida(key_simple, token, [], False))
        key_simple = key_simple + 1

# Función de serialización personalizada
def custom_serializer(obj):
    if isinstance(obj, (OracionTraducida, PalabraTraducida, PictogramaInfo)):
        return obj.__dict__
    return obj

def save_to_file(oraciones_traducidas, directorio, archivo):
    oraciones_traducidas_json = json.dumps(oraciones_traducidas, default=custom_serializer, ensure_ascii=False,indent=2)

    if not os.path.exists(directorio):
        # Crear el directorio si no existe
        os.makedirs(directorio)

    with open(archivo, 'w', encoding='utf-8') as file:
        file.write(oraciones_traducidas_json)

def sharding(oraciones_guardar):
    total_oraciones = len(oraciones_guardar)
    shards = 50
    print(f"total oraciones traducidas : {total_oraciones}")

    for i in range(450):
        comienza_en = shards*i
        termina_en = shards*(i+1)
        if(termina_en > total_oraciones):
            termina_en = total_oraciones

        save_to_file(oraciones_guardar[comienza_en:termina_en], f"resultados/validacion/", f"resultados/validacion/resultado_traduccion_simple_{i}.json")

def preprocesar_compuestos(dictionario_pictogramas_compuestas):
    #Usamos tries para realizar el preprocesamiento de los pictogramas compuestos
    automaton = Automaton()

    for idx, phrase in enumerate(list(dictionario_pictogramas_compuestas.keys())):
        automaton.add_word(phrase, (idx, phrase))

    automaton.make_automaton()
    return automaton

def eliminar_duplicados():
        try:
            with open("D:\\Tesis\\Tesis\\ElaboracionDataset\\repositorio\\validados_psicologia\\train\\oraciones lectura simple_v1.txt", 'r', encoding='utf-8') as archivo_texto:
                lineas = archivo_texto.readlines()
                lineas = [linea.strip() for linea in lineas]
                oraciones_no_dups = []
                existe_oracion = set()
                for oracion in lineas:
                    if oracion not in existe_oracion:
                        existe_oracion.add(oracion)
                        oraciones_no_dups.append(oracion)
                    else:
                        print(f"existe {oracion}")
                with open('oraciones_lectura_simples_no_duplicados.txt', 'w', encoding='utf-8') as file:
                    for sentence in oraciones_no_dups:
                        file.write(sentence + '\n')
        except FileNotFoundError:
            print(f'El archivo no se encuentra.')

#eliminar_duplicados()
dictionario_pictogramas_simples, dictionario_pictogramas_compuestas = cargar_pictogramas()
automaton = preprocesar_compuestos(dictionario_pictogramas_compuestas)
oraciones_tokenizadas = tokenizar_lematizar_oraciones()
oraciones_traducidas = traducir_oraciones(oraciones_tokenizadas, dictionario_pictogramas_simples, dictionario_pictogramas_compuestas)

print("Guardando ... ")
#sharding(oraciones_traducidas)
print("Finalizado ... ")

