import pandas as pd
class Utils:

    def __init__(self):
        self.key = "none"
        self.keys = ['48e4d31402d5a0bb72a3943002160fbb','01c18064ba2caa5c8162dd48e42e086f','3c7ef109b4b29235aa8132f969495234']

    def load_from_csv(self, path, delimiter=',', quotechar='"', escapechar=None):
        return pd.read_csv(path, delimiter=delimiter, quotechar=quotechar, escapechar=escapechar)

    def features_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y]
        return X, y

    def model_export(self, clf, score):
        print('name: ' + str(clf))
        score = np.abs(score)
        joblib.dump(clf, './models/' + str(score))

    def vectorized_fiting(self, path, wordToVectorized):
        dataFrame1 = self.load_from_csv('./in/urls2.csv')
        alldata = pd.concat([dataFrame1])
        alldata_data = alldata.values[:, 0]
        vectorizer = TfidfVectorizer()
        vectorizer.fit_transform(alldata_data)
        x_test = vectorizer.transform([wordToVectorized])
        return x_test

    def textData_cleaning(self, text):
        # Inicializa los procesadores de texto
        porter = PorterStemmer()
        lancaster = LancasterStemmer()
        wordnet = WordNetLemmatizer()

        # Tokeniza el texto
        text = TweetTokenizer().tokenize(text)
        # print(f'Texto tokenizado: {text}')  # Verifica la tokenización

        # Elimina stopwords
        text = self.stopwords_cleaner(text)
        # print(f'Texto sin stopwords: {text}')  # Verifica el texto después de eliminar stopwords

        # Construye el texto final
        if not text:
            # print("No se encontraron palabras después de eliminar las stopwords.")
            return ''  # Retorna cadena vacía si no hay texto

        thematik = ' '.join(text)
        # print(f'Texto final limpiado: {thematik}')  # Verifica el texto final limpio
        return thematik
    
    def stopwords_cleaner(self, text):
        stoped = stopwords.words('spanish')
        content = [w for w in text if w.lower() not in stoped]
        return content

    def confirmKey(self, value):
        self.key = value
        return self.key in self.keys
