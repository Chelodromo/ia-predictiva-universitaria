# Borrador de Trabajo Final

## 1. Introduccion

Las instituciones de educacion superior enfrentan, de forma recurrente, el desafio de planificar recursos academicos, administrativos y financieros bajo condiciones de incertidumbre. Entre las variables criticas para dicha planificacion se encuentra la matricula estudiantil, cuya variacion entre periodos impacta directamente en la asignacion docente, la apertura de comisiones, la gestion presupuestaria y la toma de decisiones estrategicas.

En este contexto, la inteligencia artificial predictiva ofrece un enfoque metodologico para estimar comportamientos futuros a partir de datos historicos. Este trabajo propone el desarrollo de un sistema de pronostico de matricula universitaria utilizando informacion historica de aspirantes y alumnos, con foco en predecir, para el proximo cuatrimestre, tanto la cantidad total de nuevos matriculados como su distribucion aproximada por carrera.

El proyecto se implementa sobre una arquitectura de servicios contenerizados orientada a practicas de MLOps (Airflow, MLflow, MinIO, FastAPI y bases de datos relacionales), con el objetivo de garantizar reproducibilidad, trazabilidad experimental y posibilidad de automatizacion del ciclo de vida del modelo.

## 2. Objetivos

### 2.1 Objetivo general

Desarrollar e implementar una solucion de inteligencia artificial predictiva que permita estimar la cantidad de estudiantes que se matricularan en la universidad en el proximo cuatrimestre, tanto a nivel agregado institucional como por carrera.

### 2.2 Objetivos especificos

1. Consolidar un dataset analitico a partir de las tablas institucionales de aspirantes, alumnos y personas, aplicando reglas de negocio consistentes para identificar eventos de matriculacion.
2. Definir una estrategia de depuracion y transformacion de datos que contemple calidad historica, tratamiento de duplicados y consistencia temporal por periodo y cuatrimestre.
3. Entrenar y comparar modelos de pronostico para series temporales, incorporando lineas base y metricas de evaluacion adecuadas.
4. Seleccionar el modelo con mejor desempeno predictivo bajo validacion temporal y registrar experimentos en una plataforma de seguimiento.
5. Integrar el proceso de entrenamiento y generacion de pronosticos en un pipeline automatizado sobre la infraestructura contenerizada del proyecto.
6. Exponer resultados de prediccion en un servicio API para su consumo por otras capas de analitica o gestion.

## 3. Alcance inicial

El alcance inicial contempla la construccion de predicciones para el horizonte de un cuatrimestre, utilizando como unidad temporal el par (anio, cuatrimestre). En una primera etapa, el analisis se concentra en periodos historicos con mayor densidad de registros para reducir sesgos por falta de datos en periodos tempranos.

## 4. Nota de trabajo

Este documento es un borrador vivo y se actualizara de forma iterativa a medida que avance la construccion del dataset, la validacion de modelos y la implementacion del pipeline MLOps.

## 5. Sintesis de avance tecnico (23-02-2026)

Durante la etapa actual se definio el objetivo predictivo en dos niveles: pronostico de matriculados totales para el proximo cuatrimestre y pronostico de matriculados por carrera. La regla de negocio adoptada para identificar conversion de aspirante a alumno fue `id_alumno IS NOT NULL` en la tabla de aspirantes.

### 5.1 Decisiones de datos

1. Se priorizo la variable temporal oficial del sistema (`periodo` + `id_cuatrimestre`) por sobre fechas historicamente inconsistentes.
2. Se establecio una ventana de trabajo inicial desde 2007 por mayor densidad de registros historicos.
3. Se acordo que los registros con `id_cuatrimestre = 0` se expanden a ambos cuatrimestres (1 y 2) para no perder informacion.
4. Se definio evitar doble conteo de matriculacion por persona, anio y cuatrimestre.

### 5.2 Activos construidos en base transaccional

Se crearon vistas analiticas para preparar la capa de datos orientada a modelado:

1. `vw_aspirante_base`: normaliza llaves minimas para el analisis (`id_persona`, `id_alumno`, `anio`, `cuatrimestre`).
2. `vw_dm_matricula_total`: agrega por anio-cuatrimestre la demanda total y el objetivo de matriculados.
3. `vw_dm_matricula_carrera`: agrega por anio-cuatrimestre-carrera para el objetivo segmentado.

### 5.3 Integracion MLOps sobre infraestructura docker

Se implemento un DAG inicial en Airflow para extraccion automatizada desde MySQL remoto (acceso por VPN) hacia MinIO en formato CSV.

1. DAG: `extract_mysql_views_to_minio`.
2. Archivo: `airflow/dags/extract_mysql_views_to_minio.py`.
3. Fuente: vistas `vw_dm_matricula_total` y `vw_dm_matricula_carrera`.
4. Destino: bucket `data` en MinIO, bajo prefijo configurable (`mysql_exports/unsta`).
5. Dependencias incorporadas en Airflow: `apache-airflow-providers-mysql` y `pymysql`.
6. Conexion declarada en secretos de Airflow: `mysql_unsta` en `airflow/secrets/connections.yaml`.

### 5.4 Resultado de la etapa

Con esta etapa finalizada, el proyecto cuenta con una base reproducible para iniciar el entrenamiento de modelos de series temporales y registrar experimentos en MLflow sobre datos extraidos de manera automatizada.

### 5.5 Proximo hito

Implementar el pipeline de entrenamiento inicial (baseline + modelos candidatos), con validacion temporal y seleccion de modelo para prediccion de proximo cuatrimestre total y por carrera.
