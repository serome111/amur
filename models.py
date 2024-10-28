import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import csv

# Clase de modelos ajustada
class Models:

    def __init__(self):
        # Agregamos todos los modelos recomendados
        self.reg = {
            'NB': MultinomialNB(alpha=0.005),
            'KNN': KNeighborsClassifier(n_neighbors=3),
            'Nn': MLPClassifier(),
            'TreeCl': tree.DecisionTreeClassifier(max_depth=15),
            'RF': RandomForestClassifier(criterion='entropy', max_depth=30, n_estimators=200),
            # 'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            # 'LGBM': LGBMClassifier(),
            # 'SVM': SVC()
            'LR': LogisticRegression(),
            'AdaBoost': AdaBoostClassifier(algorithm='SAMME', learning_rate=1, n_estimators=100),
            'Bagging': BaggingClassifier(estimator=tree.DecisionTreeClassifier())
        }

        # Agregamos los parámetros para cada modelo
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
                'verbose': [False]
            },
            'TreeCl': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [4, 5, 6, 7, 8, 9, 10, 15, 20]
            },
            'RF': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'criterion': ['gini', 'entropy']
            },
            # 'XGB': {
            #     'n_estimators': [100, 200],
            #     'max_depth': [6, 10, 15],
            #     'learning_rate': [0.01, 0.1, 0.3]
            # },
            # 'LGBM': {
            #     'n_estimators': [100, 200],
            #     'max_depth': [6, 10, 15],
            #     'learning_rate': [0.01, 0.1, 0.3]
            # },
            # 'SVM': {
            #     'C': [0.1, 1, 10],
            #     'kernel': ['linear', 'rbf']
            # }
            'LR': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2']
            },
            'AdaBoost': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 1]
            },
            'Bagging': {
                'n_estimators': [10, 50, 100]
            }
        }

    def preprocess_data(self, df):
        """
        Preprocesar URLs: Convertir todo a minúsculas, pero sin eliminar ningún carácter.
        Vectorización usando TfidfVectorizer.
        """
        # Convertir las URLs a minúsculas sin eliminar caracteres
        df['urls'] = df['urls'].str.lower()

        # Utilizamos TfidfVectorizer para capturar los patrones en las URLs
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))  # N-gramas de 3 a 5 caracteres
        X_urls = vectorizer.fit_transform(df['urls'])

        # Si hay campos adicionales, los procesamos
        if 'extra_field' in df.columns:
            X_extra = df[['extra_field']].values
            X_extra = StandardScaler().fit_transform(X_extra)  # Escalado si es numérico
            X = hstack((X_urls, X_extra))  # Usar hstack para combinar
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
        name_model = ""

        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for name, reg in self.reg.items():
            X_train_prepared, X_test_prepared = X_train_scaled, X_test_scaled

            if name in ['KNN', 'Nn']:
                n_components = min(10, X_train_scaled.shape[1])
                pca = PCA(n_components=n_components)
                X_train_prepared = pca.fit_transform(X_train_scaled)
                X_test_prepared = pca.transform(X_test_scaled)

            # print(f"Entrenando modelo {name}...")
            grid_reg = GridSearchCV(
                reg, self.params[name], cv=4, scoring='accuracy').fit(X_train_prepared, y_train)
            score = grid_reg.best_score_
            # print("")
            # print(f"Score para {name}: {score}")
            # print(f"Mejor modelo: {grid_reg.best_estimator_}")

            if score > best_score:
                best_score = score
                best_model = grid_reg.best_estimator_
                name_model = name

        print("")
        print(f"Mejor Score: {best_score}")
        print(f"Mejor Modelo: {best_model}")
        return best_model, X_test_prepared,name_model
