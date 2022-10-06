How to use embeddings for feature extraction?
==============================

In real world datasets, we can have numerical features like price and we also have categorical features like gender.
Usually, categorical features can be handled through encoders like one-hot encoder which creates a sparse binary matrix, label encoder which assigns a label to each category or, even more complex methods, like the catboost encoder which can be seen as a conditional probability in case of a classification or a moving average in case of a regression problem.
Those techniques, can be extremely powerful and useful when we have a small set of categories, but what if we have a text describing a product with on average 85 words?


**Dataset and Problem definition:**

In this case, we have a dataset with 27 columns and the goal is to predict the sales that each recipe will generate in the upcoming weeks.
We have several features like:
Numerical: calories, carbs, proteins, fat, etc.
Categorical: difficulty, cooking_time, etc.
Textual: description and recipe_name

sample of the dataset:

![image](https://user-images.githubusercontent.com/104824314/194349314-765659ec-3e72-473e-8c5a-63e6f266d559.png)


**How to use embeddings to extract information from description and recipe_name?**

We are going to take advantage of the incredible hugging face 🤗 framework to extract information from those features.

**First:** We need to import the model and the tokenizer:
 - There are different models that we can try, and you check them here: https://huggingface.co/models?pipeline_tag=feature-extraction
 - It is important to use the model's tokenizer so that it receives the data in a proper format and they are also useful since they already clean up the data for you.
 - Each tokenizer will have different ways of dealing with the data, therefore it is important to read about them.
```
from transformers import AutoModel, AutoTokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
```

**Second:** We extract the hidden state associated to the token CLS which represents an entire sequence of text and rather than dealing with a 768 array for each token in a string, we just need to deal with one (the 768 dimension varies from model to model).
 - The CLS token in case of DistilBert is the first one, therefore we can use the following code to access the hidden state:

```
df_train_clean['recipe_name'] = df_train_clean['recipe_name'].apply(lambda x: model(**tokenizer(x, return_tensors="pt")).last_hidden_state[:,0,:].detach().numpy()[0])
df_train_clean['description'] = df_train_clean['description'].apply(lambda x: model(**tokenizer(x, return_tensors="pt")).last_hidden_state[:,0,:].detach().numpy()[0])
```

**Plotting Embeddings**

Now, we can use some techniques of dimensionality reduction like PCA or UMAP to plot these embeddings and try to understand if these features have some predictive power.

Let's pick up the recipe_name as an example.

We start by creating a data frame from the embeddings:
```
recipe_name_df = helper.get_embeddings_df(df_train_clean, 'recipe_name')
```

Then we apply PCA on this new dataset:
```
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
recipe_name_2_components = pd.DataFrame(pca.fit_transform(recipe_name_df), columns = ['X', 'Y'])
```

In our case, since we have a continuous target, we should discretise it in bins so that we can easily plot it.
```
from sklearn.preprocessing import KBinsDiscretizer
bin = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
df_train_clean['bin_sales'] = bin.fit_transform(df_train_clean['sales'].values.reshape(-1, 1))
sns.countplot(df_train_clean.bin_sales)
plt.title("Number of samples in each bin level")
plt.show()
```

We can now create a scatter plot with the 2 components and a different color based on the bin.

```
recipe_name_2_components['bin_sales'] = df_train_clean['bin_sales']
sns.scatterplot(data=recipe_name_2_components, x='X', y='Y', hue='bin_sales', palette="tab10")
plt.show()
```
![image](https://user-images.githubusercontent.com/104824314/194350255-53489e49-fe83-4b13-82bf-4026d86c506e.png)

From the chart above is not clear the predictive power of these 2 components, maybe because this feature is not important for this problem!
Nevertheless, the goal here was to show how to deal with categorical features with high cardinality, so let's use these features in our model!!


**Model**
```
# get a 768 column data frame for recipe_name and description
recipe_name_df = helper.get_embeddings_df(df_train_clean, 'recipe_name')
description_df = helper.get_embeddings_df(df_train_clean, 'description')
recipe_name_df_test = helper.get_embeddings_df(df_test_clean, 'recipe_name')
description_df_test = helper.get_embeddings_df(df_test_clean, 'description')
# identify the number of components needed based on variance explained
recipe_name_components, recipe_name_components_test = helper.apply_PCA(recipe_name_df, recipe_name_df_test, 'recipe_name', variance_explained=0.8)
description_components, description_components_test = helper.apply_PCA(description_df, description_df_test, 'description', variance_explained=0.8)
# merge with data frame
df_train_clean = pd.merge(df_train_clean, recipe_name_components, left_index=True, right_index=True, how='left')
df_train_clean = pd.merge(df_train_clean, description_components, left_index=True, right_index=True, how='left')
df_test_clean = pd.merge(df_test_clean, recipe_name_components_test, left_index=True, right_index=True, how='left')
df_test_clean = pd.merge(df_test_clean, description_components_test, left_index=True, right_index=True, how='left')
```

Now that we have all the features in our train and test sets, we can select a model to predict sales! 😁
You can check the entire code here: https://github.com/rjguedes8/feature_embedding/blob/main/notebooks/feature_embedding.ipynb

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
