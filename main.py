from utils import Utils
from model2 import Models
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load
import csv
if __name__ == "__main__":
    utils = Utils()
    models = Models()
    dataFrame1 = utils.load_from_csv('./in/urls2.csv', delimiter=',', quotechar='"', escapechar='\\')



    # unlabeled_rows = dataFrame1[dataFrame1['label'].isna() | (dataFrame1['label'] == '')]

    # print(f"Número total de filas: {len(dataFrame1)}")
    # print(f"Número de filas sin etiquetar: {len(unlabeled_rows)}")

    # if len(unlabeled_rows) > 0:
    #     print("\nPrimeras 10 filas sin etiquetar:")
    #     print(unlabeled_rows.head(10))
        
    #     # Guardar las filas sin etiquetar en un nuevo archivo CSV
    #     unlabeled_rows.to_csv('unlabeled_urls.csv', index=False)
    #     print("\nLas filas sin etiquetar se han guardado en 'unlabeled_urls.csv'")


    frames = [dataFrame1]
    alldata = pd.concat(frames)
    alldata_data = alldata.values[:, 0]
    alldata_target = alldata.values[:, 1]
    
# Code para mejorar el data set y tener mas informacion.
    # Aquí agregamos el conteo de ejemplos por clase
    y = pd.Series(alldata_target)  # Convertimos las etiquetas en una Serie de pandas
    class_counts = y.value_counts()  # Contamos cuántas instancias tiene cada clase
    print(f"Conteo de ejemplos por clase:\n{class_counts}")  # Imprimimos el conteo

    # Identificamos la clase con menos ejemplos
    min_class = class_counts.idxmin()
    min_class_count = class_counts.min()
    print(f"La clase con menos ejemplos es '{min_class}' con {min_class_count} ejemplos.")



    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(alldata_data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, alldata_target, test_size=0.33, random_state=42)

    best_model, X_test_transformed,name_model = models.grid_training(X_train, y_train, X_test)
    print("")
    print("")
    print("")
    # Evaluar el modelo en el conjunto de prueba
    y_pred = best_model.predict(X_test_transformed)

    # Asegurar que tanto y_test como y_pred sean cadenas de texto
    y_test_str = [str(label) for label in y_test]
    y_pred_str = [str(label) for label in y_pred]

    # Imprimir el informe de clasificación y la precisión
    print(classification_report(y_test_str, y_pred_str, zero_division=1))
    print(f"Accuracy: {accuracy_score(y_test_str, y_pred_str)}")

    # Guardar el modelo
    dump(best_model, f'./models/best_model.joblib')
    
    # Guardar el vectorizador
    dump(vectorizer, f'./models/vectorizer.joblib')

    print(f"Modelo {name_model} y vectorizador guardados exitosamente.")

    print(alldata)

# # Ejemplo de cómo cargar y usar el modelo guardado
# def use_saved_model(new_data):
#     # Cargar el modelo
#     loaded_model = load('./models/best_model.joblib')
    
#     # Cargar el vectorizador
#     loaded_vectorizer = load('./models/vectorizer.joblib')
    
#     # Transformar los nuevos datos
#     X_new = loaded_vectorizer.transform(new_data)
    
#     # Hacer predicciones
#     predictions = loaded_model.predict(X_new)
    
#     return predictions

# # Ejemplo de uso
# new_data = ["http://example.com", "http://malicious-example.com"]
# predictions = use_saved_model(new_data)
# print(predictions)