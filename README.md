# 🧠 RNA-BackPropagation

Este proyecto implementa una **Red Neuronal Artificial (RNA)** utilizando el algoritmo de **retropropagación (Backpropagation)** en Python. Ideal para quienes están aprendiendo cómo funcionan las redes neuronales desde cero.


## 🗂️ Estructura del Proyecto

- 📄 **`main.py`**: Archivo principal para ejecutar la red neuronal.
- 🔁 **`back_propagation.py`**: Implementación del algoritmo de retropropagación.
- 🧮 **`normalize_patterns.py`**: Funciones para normalizar los datos de entrada.
- 📊 **`propagation_values.py`**: Manejo de los valores propagados en la red.
- ⚖️ **`weights.py`**: Gestión de los pesos sinápticos.
- 💬 **`messages.py`**: Mensajes de estado y configuración.
- 📁 **`data/`**: Directorio con conjuntos de datos de entrenamiento y prueba.
- 📦 **`requirements.txt`**: Lista de dependencias necesarias.

## ⚙️ Requisitos

- Python 3.7 o superior
- Instalar dependencias:

```bash
pip install -r requirements.txt
```
## 🚀 Uso

1. 📌 **Prepara los datos**  
   Asegúrate de colocar los archivos de datos dentro del directorio `data/`.

   El directorio `data/` debe contener archivos `.csv` con los patrones de entrada y salida esperados para entrenar la red. Por ejemplo:

   - `training_patterns.csv`: contiene los patrones de entrada (por ejemplo: valores binarios o reales normalizados).
   - `target.csv`: contiene las salidas esperadas para cada patrón (por ejemplo: `[0]`, `[1]` o valores entre `0` y `1`).
   - Los datos deben estar organizados en filas, separados por comas o espacios, y deben coincidir en cantidad.
