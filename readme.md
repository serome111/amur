# URL analyzer to determine if it is malicious or not.

Base project with which I won the hackathon of the DragonJar conf computer security event

# install 

create a virtual env
```uv venv```
```
source .venv/bin/activate
```


# Run model 

```python models.py```

# Config for run microservice

In server.py put the path where your model has been exported from the models.py file

```model_path = './models/best_model.joblib'
vectorizer_path = './models/vectorizer.joblib'
loaded_model = load(model_path)
```

# web sources
https://medium.com/bitgrit-data-science-publication/forget-pip-install-use-this-instead-754863c58f1e