# CONTROL Y DETECCIÓN DE FLUJO PEATONAL Y VEHICULAR

> [!IMPORTANT]
> - Clonar el repositorio en la rama que corresponde 'daniel'
> - git clone --single-branch --branch daniel https://github.com/diegoalrv/yolo-flow.git


> [!TIP]
> - Para mantener un orden con los videos con los que se trabajará se puede crear una carpeta y guardarlos en ella.
> - Hacer los cambios de nombres correspondientes a los videos y/o al .csv para los gráficos. 
> - Asegurarse de tener las librerias instaladas en su entorno.
> - Se puede utilizar tanto VisualCode como GoogleCollab.

> [!WARNING]
> - Los modelos entrenados están cargados en el repositorio
> - Si se modifica el nombre del 'frame_inicial.jpg', se debe corregir en los gráficos.


---


### Pasos a seguir para la detección del video - 'main.py'
- Ya con el repositorio clonado, cambiar la ruta local de los archivos que se utilizarán.
- Seleccionar el modelo a trabajar y las clases que se identificarán con VisDrone.
- Esta es una demostración de las secciones para modificar:
1.
```python
video_path = "Ruta_del_video.mp4"
extraer_frame_inicial(video_path)  
```
2.
```python 
  model = YOLO('yolov8_50epochs.pt')
#{0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}
clases_a_detectar = [0, 3, 8, 9]
```
- Al comenzar con el análisis del video, se genera una imagen con el primer frime del video usado que se utilizará para dar un contexto al gráfico correspondiente.
- Al finalizar el análisis, se genera un archivo .csv que extraerá información necesaria para generar los gráficos.

---

### Pasos para los gráficos - 'graficos.ipynb' 
- Ejecutar correctamente las casillas y corregir el nombre usado en el .csv guardado.
- Si se hicieron los ajustes de nombre, se ejecutará y mostrará en base a los datos recopilados.
```python 
  file_path = "detecciones_1.csv"
df = pd.read_csv(file_path, delimiter=';')
```
  
---

<p align="center">
  <img src="logo-citylab-sticky.png" alt="CityLab" />
</p>