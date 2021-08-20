import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re
import string
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

st.set_page_config(page_title='Recipe Recommender', page_icon="🍔", layout='wide', initial_sidebar_state='auto')

st.title('🧑‍🍳 Ultimate Recipe Recommender!')

@st.cache
def load_recipes():
    df = pd.read_pickle('dataset.pickle')
    return df.iloc[:100000]

@st.cache
def load_doc_topic():
    with open('doc_topic.pickle', 'rb') as f:
        doc_topic = pickle.load(f)
    return doc_topic[:100000]

@st.cache
def load_model():
    with open('svd_model.pickle', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache(allow_output_mutation=True)
def load_tfidf():
    with open('tfidf.pickle', 'rb') as f:
        tfidf = pickle.load(f)
    return tfidf

loading_text = st.text('🛒 Gathering Ingredients... ')
data = load_recipes()
doc_topic = load_doc_topic()
model = load_model()
tfidf = load_tfidf()
indexes = data.index
loading_text.text('Ingredients Gathered! ')

col1, col2 = st.columns(2)
col1.subheader("🍅 Input ")
col2.subheader("🥫 Recommendations ")

def preprocessor(text):
    
    text = re.sub('(\((?:\(??[^\(]*?\)))','',text)
        
    text = text.split(',')[0]
    text = text.split(' or ')[0]
    text = text.split(' for ')[0]
    text = re.sub('[^a-zA-Z]\sto\s[^a-zA-Z]','',text)
    text = text.split(' to ')[0]

    text = text.split(' and ')[0]


    text = text.lower()
    
    # Removes quotation marks.
    text = text.replace('"', "")
    
    # Remove numeric characters.
    text = re.sub('\w*\d\w*', ' ', text)
    
    # Remove puncuation.
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    
    text = re.sub('\s+', ' ', text).strip()
          
    text = " ".join([stemmer.stem(x) for x in text.split(' ') if (x not in unit_words)])

    text = text.title()
    
    combined = text.replace(' ','')
    
    text = " ".join([combined,text])
    
    
    return text

unit_words = {
    'all',
     'bunch','bunched'
     'bushel','bushels',
     'c',
     'chopped',
     'clove','cloves',
     'coarse','coarsely',
     'crumbled',
     'cup','cups',
     'dash','dashes',
     'diced',
     'drop','drops',
     'extra',
     'finely',
     'fl. oz',
     'fresh','freshly',
     'g',
     'gallon','gallons',
     'glass','glasses',
     'gram','grams',
     'grated',
     'italian',
     'kg','kgs',
     'lb','lbs',
     'liter','liters',
     'lrg','large',
     'mashed',
     'med','medium',
     'minced',
     'ml','mls',
     'ounce','ounces','oz',
     'package','packaged','pkg',
     'pasta','pastas',
     'pinch','pinches',
     'pint','pints',
     'pound','pounds',
     'purpose',
     'qt',
     'quart','quarts',
     'scoop','scoops',
     'shot','shots',
     'shredded',
     'sifted',
     'sliced','slivered'
     'sm','small',
     'stick','sticks',
     'tablespoon','tablespoons','tbs','tbsp','tbsps',
     'teaspoon','teaspoons','tsp','tsps',
     'thin','thinly',
     'to',
     'virgin',
     'Virgin',
     'white','whites',
     'whole',
     'x',
     'yolk','yolks'
}


def process_custom_ingredients(text):
    global custom_ingredients
    ingredients = text.split(',')
    ingredients = ' '.join([preprocessor(x) for x in ingredients])
    custom_ingredients = ingredients = ' '.join(list(set(ingredients.split(' '))))

with col1:
    custom_ingredients = 'test'
    recipe = data['title'].iloc[0]
    index = 0 


    input_type = st.selectbox(
        'Your ingredients or exisiting recipe?',
        ['My Ingredients','Existing Recipe'])

    def input_recipe():
        global index, recipe
        recipe = st.selectbox(
            'Which recipe do you already love?',
            data['title'].iloc[:10000])

        index = indexes[data['title']==recipe][0]
        
        st.markdown('[Source]('+data.iloc[index]['url']+')')

        st.markdown('   \n'.join(data.iloc[index]['ingredients']))
    def input_ingredients():
        ingredients_text = st.text_area(
            'Enter your ingredients (comma-seperated):',
            '''ingredient 1, ingredient 2, ...'''
        )

        st.button(
            'Get Recommendations',
            on_click=process_custom_ingredients(ingredients_text)
        )

    if input_type == 'Existing Recipe':
        input_recipe()
    else:
        input_ingredients()
with col2:
    

    def get_recommendations(input):
        rec_idx = pairwise_distances(input,doc_topic,metric='cosine').argsort()[0][1:11]
        recipe_url = list(zip(data.iloc[rec_idx]['title'],data.iloc[rec_idx]['url'],list(range(1,11))))
        
        st.markdown('  \n'.join([' '+str(x[2])+'. ['+x[0]+']('+x[1]+')' for x in recipe_url]))
        


    # st.button('Get Recommendations!', key=None, help=None, on_click=get_recommendations, args=None, kwargs=None)
    if input_type == 'Existing Recipe':
        get_recommendations(doc_topic[index].reshape(1,-1))
    else:
        get_recommendations(model.transform(tfidf.transform([custom_ingredients])))
    