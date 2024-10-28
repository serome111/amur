import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.sparse import hstack

# Clase de modelos ajustada
class Models:

    def __init__(self):
        self.reg = {
            'NB': MultinomialNB(),
            'KNN': KNeighborsClassifier(),
            'Nn': MLPClassifier(),
            'TreeCl': tree.DecisionTreeClassifier()
        }

        self.params = {
            'NB': {
                'alpha': [.01, .04, .005]
            },
            'KNN': {
                'n_neighbors': [3, 5, 10]
            },
            'Nn': {
                'activation': ['identity', 'logistic', 'tanh', 'relu'],
                'solver': ['adam'],
                'hidden_layer_sizes': [(8,)],
                'max_iter': [1500],
                'learning_rate_init': [0.01, 0.001],
                'tol': [0.01],
                'early_stopping': [True],
                'n_iter_no_change': [10],
                'verbose': [True]
            },
            'TreeCl': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [4, 5, 6, 7, 8, 9, 10, 15, 20]
            }
        }

    def preprocess_data(self, df):
        """
        Preprocesar el texto (URLs) convirtiéndolas a una representación numérica.
        Incluir otros campos si es necesario.
        """
        vectorizer = CountVectorizer()
        X_urls = vectorizer.fit_transform(df['urls'])

        # Si el campo extra es de texto o numérico, procesarlo
        if 'extra_field' in df.columns:
            X_extra = df[['extra_field']].values
            X_extra = StandardScaler().fit_transform(X_extra)  # Escalar si es numérico
            X = hstack((X_urls, X_extra))  # Usar hstack para concatenar matrices dispersas y densas
        else:
            X = X_urls

        return X, df['label']

    def grid_training(self, X_train, y_train, X_test):
        """
        Entrena varios modelos usando GridSearchCV para encontrar el mejor modelo.
        Aplica PCA solo en los modelos necesarios y transforma tanto X_train como X_test.
        """
        best_score = 0
        best_model = None
        pca = None

        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for name, reg in self.reg.items():
            X_train_prepared, X_test_prepared = X_train_scaled, X_test_scaled

            if name in ['KNN', 'Nn']:
                # Ajusta el número de componentes de PCA según la dimensionalidad
                n_components = min(10, X_train_scaled.shape[1])
                pca = PCA(n_components=n_components)
                X_train_prepared = pca.fit_transform(X_train_scaled)
                X_test_prepared = pca.transform(X_test_scaled)

            grid_reg = GridSearchCV(
                reg, self.params[name], cv=5, scoring='accuracy').fit(X_train_prepared, y_train)
            score = grid_reg.best_score_
            print(f"Score para {name}: {score}")
            print(f"Mejor modelo: {grid_reg.best_estimator_}")

            if score > best_score:
                best_score = score
                best_model = grid_reg.best_estimator_

        print(f"Mejor Score: {best_score}")
        print(f"Mejor Modelo: {best_model}")
        return best_model, X_test_prepared


# Cargar datos desde CSV
data = pd.read_csv('./in/urls.csv')

# Instancia de la clase Models
model = Models()

# Preprocesar los datos
X, y = model.preprocess_data(data)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar y obtener el mejor modelo
best_model, X_test_transformed = model.grid_training(X_train, y_train, X_test)

# Evaluar el modelo en el conjunto de prueba
y_pred = best_model.predict(X_test_transformed)

# Asegurar que tanto y_test como y_pred sean cadenas de texto
y_test_str = [str(label) for label in y_test]
y_pred_str = [str(label) for label in y_pred]

# Evitar error de precisión indefinida y asegurar tipos de datos consistentes
print(classification_report(y_test_str, y_pred_str, zero_division=1))
print(f"Accuracy: {accuracy_score(y_test_str, y_pred_str)}")
