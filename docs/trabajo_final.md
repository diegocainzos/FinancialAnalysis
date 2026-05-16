# Diseño de una tarea de evaluación en PLN: sentimiento financiero en redes sociales

## 1. Propósito de la tarea

### 1.1 Identificación del problema

En economía y finanzas, el sentimiento del mercado puede aportar información complementaria a los datos financieros tradicionales de una empresa. Las noticias, comentarios y publicaciones en redes sociales pueden reflejar expectativas, preocupaciones o entusiasmo de los participantes del mercado. Por este motivo existen modelos específicos para el dominio financiero, como FinBERT, entrenados o ajustados para clasificar textos económicos en sentimiento positivo, neutral o negativo.

El problema abordado consiste en diseñar una tarea de evaluación para medir hasta qué punto un sistema de PLN puede clasificar correctamente el sentimiento financiero expresado en publicaciones recientes sobre empresas cotizadas. La tarea se centra en textos procedentes de redes sociales porque el mercado es un sistema dinámico y se generan datos nuevos constantemente. Un conjunto de datos fijo puede ser útil para investigación o entrenamiento, pero una aplicación práctica requiere un flujo de datos renovable.

El reto principal no es solo clasificar el sentimiento, sino separar previamente los textos realmente relacionados con economía o mercado. En las pruebas de ingesta se observó que muchos posts que mencionan empresas no hablan de su rendimiento financiero. Por ejemplo, publicaciones que contienen Amazon o AMD pueden ser anuncios de productos, ofertas o comentarios técnicos, no información económica sobre la empresa.

### 1.2 Definición clara de la tarea

La tarea propuesta es una tarea de clasificación en dos etapas sobre publicaciones de redes sociales asociadas a empresas:

1. Determinar si el texto está relacionado con economía, bolsa, resultados corporativos, ingresos, acciones, valoración o mercado de la empresa mencionada.
2. Si el texto es relevante para el dominio económico, clasificar su sentimiento hacia la empresa como `positive`, `neutral` o `negative`.

Las categorías de relevancia económica son:

- `yes`: el texto trata sobre bolsa, mercado, resultados trimestrales, ingresos, valoración, cotización, expectativas financieras o actividad inversora relacionada con la empresa.
- `no`: el texto menciona la empresa, ticker o productos, pero no habla del rendimiento económico o financiero. Incluye usos de producto, soporte técnico, anuncios de compra, empleo, música, coches o ingresos personales.

Las categorías de sentimiento son:

- `positive`: el texto expresa una señal favorable para la empresa o su acción, como subidas, récords, buenos resultados, crecimiento o perspectivas positivas.
- `neutral`: el texto menciona información financiera sin una polaridad clara, o contiene datos mixtos sin una conclusión positiva o negativa evidente.
- `negative`: el texto expresa una señal desfavorable, como caídas, riesgos, advertencias, ventas, presión competitiva o amenazas a ingresos.

La unidad de evaluación es una pareja documento-empresa. Un mismo post puede mencionar varias empresas, por lo que puede generar varias filas de evaluación si el sistema detecta varias menciones.

## 2. Colección de datos

### 2.1 Fuentes de datos

Se valoraron varias fuentes para obtener datos actualizados:

- X: se consideró inicialmente por su volumen de datos y por disponer de API y documentación, pero se descartó para este prototipo por no disponer de un nivel gratuito adecuado.
- Reddit: se consideró como fuente de debates financieros, pero no se utilizó en la implementación final porque el acceso mediante API no estaba disponible para el prototipo.
- Bluesky: se eligió como fuente de datos porque su API pública permite buscar posts sin clave de API y se pudo integrar en el pipeline.

La implementación utiliza Bluesky mediante su cliente API en Python. La ingesta se realiza por compañía usando consultas basadas en el ticker, el cashtag y la expresión `Company stock`. Los resultados se almacenan en SQLite para mantener trazabilidad entre el texto original, la empresa, la mención detectada, la decisión de relevancia y el sentimiento calculado.

Se han incluido empresas técnologicas como Apple, Tesla, Microsoft o Amazon entre otras.

### 2.2 Preparación de los datos

El pipeline implementado realiza las siguientes fases:

1. Carga de empresas desde `config/companies.json`.
2. Ingesta de posts desde Bluesky.
3. Almacenamiento de documentos en `raw_documents`.
4. Detección de menciones a empresas mediante reglas sobre ticker, cashtag, nombre y alias.
5. Clasificación de relevancia económica mediante un LLM local servido con `llama.cpp`.
6. Clasificación de sentimiento con FinBERT.
7. Persistencia de resultados en SQLite.

Inicialmente se consideró un filtrado por palabras clave para eliminar posts no económicos. Ese enfoque se descartó para el modo final con LLM porque era demasiado estricto y podía eliminar textos relevantes escritos con lenguaje variado. En la versión final, cuando se usa el filtro LLM, todos los documentos pendientes se envían al clasificador de relevancia salvo los que ya habían sido marcados previamente como irrelevantes.

El LLM escogido ha sido Qwen3.5:0.8B, un modelo suficientemente valido para hacer clasificación few shot y correr en mi maquina local.

Para evitar reprocesamiento, las decisiones de relevancia se guardan en la base de datos. Si el LLM marca una pareja documento-empresa como irrelevante, esa decisión se conserva y no se vuelve a enviar a clasificación de sentimiento en ejecuciones posteriores. Los posts irrelevantes no se borran. Se mantienen para trazabilidad y posible revisión futura.

Para la evaluación final se seleccionaron 100 filas de la base de datos que ya tenían sentimiento FinBERT calculado. Sobre esas 100 filas se realizó etiquetado manual de:

- Relevancia económica, es decir, que el post realmente pertenece al ambito economico.
- Sentimiento manual 

La distribución del etiquetado manual fue:

- Relevancia económica manual: 76 `si`, 24 `no`.
- Sentimiento manual: 44 `positive`, 46 `neutral`, 10 `negative`.

La distribución de FinBERT en la misma muestra fue:

- FinBERT: 18 `positive`, 75 `neutral`, 7 `negative`.

Para la evaluación, se hizo una división en conjuntos de entrenamiento, validación y prueba (0.8,0.1,0.1). La muestra anotada manualmente se usó como conjunto de prueba para comparar la salida de FinBERT frente al etiquetado humano.

## 3. Evaluación

### 3.1 Estructura de la tarea

La tarea es de clasificación supervisada para evaluación. Cada participante recibiría un archivo con una fila por pareja documento-empresa. Como mínimo, cada fila contendría:

- Identificador del resultado.
- Identificador del documento.
- Ticker de la empresa.
- Proveedor del texto.
- Texto original.
- Etiqueta esperada de sentimiento en el conjunto de referencia.

El sistema participante debe devolver una etiqueta de sentimiento por fila con uno de estos valores exactos:

- `positive`
- `neutral`
- `negative`

Para la parte de relevancia económica, la salida esperada sería:

- `yes`
- `no`

### 3.2 Protocolo de evaluación

La evaluación automatizada compara `manual_label` frente a `finbert_label`. Se implementa un script de evaluación que calcula:

- Accuracy.
- Recall macro y weighted.
- F1-score macro y weighted.
- Precision, recall y F1 por clase.
- Matriz de confusión manual -> FinBERT.

Los resultados obtenidos en la muestra de 100 ejemplos fueron:

- Accuracy: 0.6600.
- Recall macro: 0.5879.
- Recall weighted: 0.6600.
- F1-score macro: 0.5823.
- F1-score weighted: 0.6239.

Métricas por clase:

| Clase | Precision | Recall | F1 | Soporte |
| --- | ---: | ---: | ---: | ---: |
| negative | 0.5714 | 0.4000 | 0.4706 | 10 |
| neutral | 0.6133 | 1.0000 | 0.7603 | 46 |
| positive | 0.8889 | 0.3636 | 0.5161 | 44 |

Matriz de confusión, donde las filas son las etiquetas manuales y las columnas son las etiquetas FinBERT:

| Manual / FinBERT | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 4 | 4 | 2 |
| neutral | 0 | 46 | 0 |
| positive | 3 | 25 | 16 |

También se realizó evaluación manual de relevancia económica. En la muestra final, el LLM había marcado como económicas las 100 filas, pero la revisión manual marcó 24 como no relacionadas realmente con economía o mercado. Esto muestra que el filtro de relevancia es funcional para reducir ruido, pero todavía es demasiado permisivo.

## 4. Reflexión final

El trabajo muestra que una tarea de sentimiento financiero en redes sociales tiene dos dificultades principales, la obtención de datos actualizados y la separación entre menciones reales de mercado y menciones superficiales de productos o marcas.

Bluesky permitió construir un flujo de datos sin clave de API, pero la ingesta no fue completamente estable porque aparecieron errores  de `rate limit`. Además, la fuente presenta sesgos, no representa necesariamente a todos los inversores, está condicionada por los usuarios activos en esa red y puede contener mucho ruido promocional o técnico.

El uso de FinBERT es coherente con el dominio financiero, pero los resultados de la prueba piloto indican que tiende a clasificar muchos textos como `neutral`. En la muestra manual, 44 ejemplos fueron etiquetados como positivos, mientras que FinBERT solo marcó 18 como positivos. Esto afecta especialmente al recall de la clase positiva.

La evaluación también muestra que el filtrado de relevancia es una parte crítica. En los 100 ejemplos revisados, 24 fueron considerados no económicos manualmente aunque el LLM los había dejado pasar. Esto confirma la necesidad de definir con precisión qué se considera texto económico y de mejorar el prompt o incorporar una segunda fase de revisión.

Como mejoras futuras se proponen:

- Ampliar el número de ejemplos anotados manualmente.
- Mejorar el modelo usado en el filtro de LLM, o usar un modelo BERT para clasificación como ya hemos visto en otras prácticas de la asignatura.
- Equilibrar la muestra por clases de sentimiento.
- Incorporar fuentes adicionales fuera de redes sociales para reducir el sesgo de Bluesky.


