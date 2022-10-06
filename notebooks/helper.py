import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import datetime
import argparse
import sys
sys.argv = ['']

TAGS_FEATURES = [
    "meta_tags",
    "carbs_content",
    "dish_types",
]

CAT_FEATURES_FILL_NAN = [
    "meta_tags",
    "carbs_content",
    "dish_types",
    "dish_type",
    "description",
]

CATEGORICAL_FEATURES = [
    "product_type",
    "dish_type",
    "protein_cuts",
    "heat_level",
    "preferences",
]

NUMERICAL_FEATURES = [
    "proteins",
    "number_of_ingredients_per_recipe",
    "fat",
    "carbs",
    "calories",
    "n_products",
]

TEXT_FEATURES = [
    "recipe_name",
    "description",
]

FEATURES = ['recipe_name',
            'product_type',
            'calories',
            'carbs',
            'cooking_time',
            'cuisine',
            'dish_type',
            'heat_level',
            'fat',
            'number_of_ingredients_per_recipe',
            'preferences',
            'carbs_content',
            'dish_types',
            'seasons',
            'protein_types',
            'proteins',
            'meta_tags',
            'protein_cuts',
            'n_products']


def rmse(y_true, y_pred):
    '''Calculation of root mean sqaured error'''
    return np.sqrt(mean_squared_error(y_true, y_pred))


def tag_tokenizer(tags):
    '''Tokenization through replacing "," by " " and respective split'''
    return tags.replace(",", " ").split(" ")


def estimate_sample_weights(df, weight=4):
    '''Note 1: sample_weight helps to add more importance to the most recent obsevations
       Note 2: Rasing weight to the positive power helps to increase importance of the most recent data'''
    return ((df.week_day - df.week_day.min()).dt.days + 1) ** weight

def create_week_day(df):
    '''Note: additional -1 indicates that we always pick Monday as the starting day of
       the week, otherwise logic doesn't work'''
    
    df["week_day"] = df.year_week.apply(
    lambda year_week: datetime.datetime.strptime(str(year_week) + "-1", "%G%V-%u"))
    return df

def create_n_recipes_feature(df):
    '''create number of recipes available per week'''
    
    df = df.merge(
    (df
        .groupby("year_week")
        .agg({"recipe_id": "count"})
        .rename(columns={"recipe_id": "n_products"})
    ),
    how="left",
    left_on="year_week",
    right_index=True,)
    return df


def get_embeddings_df(dataframe, feature):
    """ Creates a dataframe with the emebeddings for a specific feature
    Args: 
        dataframe (pd.DataFrame): dataframe
        feature (str): feature name
    Returns:
        emb_df (pd.DataFrame): dataframe withh embeddings
    """
    emb_df = pd.DataFrame()
    for i in range(len(dataframe)):
        emb_df = emb_df.append(pd.DataFrame(dataframe.iloc[i][feature].reshape(1,-1)))
    
    return emb_df


def apply_PCA(train_df_emb, test_df_emb, feature_name, variance_explained=0.8):
    """ Creates a dataframe with the PCA components for a specific feature
    Args: 
        train_df_emb (pd.DataFrame): embeddings from train df
        test_df_emb (pd.DataFrame): embeddings from test df
        feature_name (str): feature name
        variance_explained (float): to decide the number of components
    Returns:
        components (pd.DataFrame): dataframe with components for train
        components_test (pd.DataFrame): dataframe with components for test
    """
    pca = PCA(n_components=variance_explained)

    components = pd.DataFrame(pca.fit_transform(train_df_emb))
    components_test = pd.DataFrame(pca.transform(test_df_emb))
    for col in components.columns:
        components.rename(columns={col:str(col)+'_'+feature_name}, inplace=True)
        components_test.rename(columns={col:str(col)+'_'+feature_name}, inplace=True)
    
    return components, components_test

    
def parse_arguments():
    ''' Function to define parameters'''
    parser = argparse.ArgumentParser()
    # train params
    parser.add_argument('--n_estimators', type=int, default=3000, help='n_estimators for LGBM')
    parser.add_argument('--objective', type=str, default='regression', help='model objective') 
    parser.add_argument('--num_leaves', type=int, default=4, help='num_leaves')
    parser.add_argument('--max_depth', type=int, default=10, help='max_depth')
    parser.add_argument('--min_child_samples', type=int, default=60, help='min_child_samples')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='learning_rate')
    parser.add_argument('--colsample_bytree', type=float, default=0.6, help='colsample_bytree')
    parser.add_argument('--verbosity', type=int, default=-1, help='verbosity')
    parser.add_argument('--extra_trees', default=True, help='extra_trees')
    parser.add_argument('--random_state', type=int, default=42, help='random_state')
    #TFIDF
    parser.add_argument('--min_df', type=int, default=50, help='min_df')
    parser.add_argument('--stop_words', type=str, default='english', help='stop_words')
    #Numerical nans
    parser.add_argument('--strategy', type=str, default='mean', help='strategy')
    #Categorical nans
    parser.add_argument('--handle_missing', type=str, default='return_nan', help='handle_missing')
    #Weight previous samples
    parser.add_argument('--weight', default=4, help='weight') 
    #cross validation set up
    parser.add_argument('--folds', type=int, default=5, help='k-folds')
    parser.add_argument('--weeks', type=int, default=5, help='weeks_to_test')
    
    
    return parser.parse_args()