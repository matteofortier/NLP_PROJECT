# Ultimate Recipe Recommender

*Matteo Fortier*

## Abstract



## Design

Meal kit delivery companies such as Blue Apron, Hello Fresh and plated have all been experiencing poor customer retention. Analysts estimate a churn rate of 70% after 12 months. [Source](https://www.saasquatch.com/blog/blue-apron-addicted-to-acquisition/)

<img src="https://www.saasquatch.com/wp-content/uploads/2019/10/BlueApronCustomerRetention.png" style="zoom:33%;" />

According to a survey of people who have stopped using meal kits, the biggest reason for churn is value for money. However, 32% of respondents claimed reasons related to the food and recipes themselves, such as 'flavour of the finished recipe', 'ability to choose meals on diet', or 'difficulty level of recipe'. Hence, having a recipe recommender system based on recipes that users have previously liked, could significantly impact the retention rate of customers.

## Data

The Recipes1M+ dataset provided by [MIT](http://pic2recipe.csail.mit.edu) was used for this project. The dataset includes information on recipes including their title, instructions, ingredients and url. Instructions and ingredients were provided in a list of strings format, neither in a standardised format as all recipes were scrapped from various recipe websites. Hence, a significant amount of preprocessing and nlp had to be done on the dataset.

The dataset included 1 million recipes. The large dataset meant cloud computing tools such as google cloud platform had to be used to more easily process the data. Additionally parallel processing tools such as spacy's nlp.pipe and swifter (dask pandas) had to be used to more quickly process data.

## Algorithms

***Natural Language Processing***

1. Per-ingredient processing (get a single ingredient token per string in list, i.e red bell peppers -> RedBellPeppers)
2. Per-word processing (get multiple tokens per string in list, i.e parmesan cheese -> Parmsan Cheese)
3. Spacy Processing to extract nouns and adjectives only from text
4. Spacy Processing for lemmatization
5. Alt. SnowballStemmer for stemming
6. General preprocessing steps such as removing items inside brackets, removing punctuation, etc.

***Unsupervised Learning Models***

Multiple models were tried. Count vectorizer and TFIDF vectorizer were both tried with TFIDF being the selected vectorizer. TFIDF was preferred as it led to better recommendations for the test set of recipes. This is probably due to TFIDF accounting for the document frequency of certain ingredients.

NMF and SVD were both tried as topic modellers, and compared on the test set. It seemed SVD performed better than NMF. This could be due to the fact that SVD may hold more information with regards to negative coefficients compared to NMF. It makes sense within the context of food as certain ingredients may contradict topics. 

***Model Evaluation and Selection***

Models were evaluated against a test set of recipes. The test set includes a range of different recipes in terms of number of ingredients and types of cuisine. Based on intuition and exploring the recommendation outputs, models were evaluated on how well they performed on the test set.

## Tools

- Python, pandas, NumPy, SciPy, scikit-learn
- Google Cloud Platform
- Swifter (Dask Pandas)
- SpaCy
- SVD, NMF, TFIDF Vectorizer
- Streamlit
- WordCloud

## Communication

The project used powerpoint for the presentation and the python visualisation libraries for the visuals. 

WordCloud was used to generate a word cloud

Streamlit was used to deploy a prototype application for the recommender.
