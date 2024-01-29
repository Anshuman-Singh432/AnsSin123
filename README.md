#  Use the Gdrive link to download the whole solution zipped file-  The file name is assigment.zip

https://drive.google.com/file/d/1IabM4srlPa8Vh2oye7fFsJCM9j0a_lkd/view?usp=sharing


Visable.ipynb- is the Jupyter Notebook file name of the data modelling code file.
main.py- is the file used for integrating Fast API and its using custom_tranformer.py fo data preprocessing.
custom_tranformer.py- is the file used for preprocessing the data recieved from application
Dockerfile-  have been created containg enviroment dependencies.


# AnsSin123
German search queries classification
Data Science Coding Challenge – Visable
## Task- 
German search queries classification project - training a machine learning model, deploying the trained model by creating a web API with FAST API, and containerizing the application using Docker.
## Approach-
## Data Modelling
1 - Reading the provided dataset with pandas it contains text and label as column name.

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/e0053305-b169-4d79-b8b6-23d45ff5d675)

2 - Identifying and dropping the rows having any null values in dataset. Final shape of dataset is-

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/65fdb8e4-e0c6-4bbd-bace-f88309409583)

3 - Creating a map of unique values to unique numbers for the label column as machine learning works on numbers.

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/7a256e3b-3a8b-44bb-a22b-5160f48038e2)

4 - Using Spacy Library- It provides pre-trained models for various languages, including English, German, Spanish, French, and others. These models are trained on large corpora and include components for part-of-speech tagging, named entity recognition, dependency parsing, and more.

5 - Downloading and using the German language model for the given task 

    !python -m spacy download de_core_news_sm
    
6 - Processing German text of dataset, using the loaded language model.

Rendering- It is used for visualizing linguistic annotations in a Jupyter notebook or a web browser. It generates a graphical representation of the linguistic analysis performed by spaCy, making it easier to understand the relationships and structures within a given text.
The most common use of displacy.render() is to visualize the dependency tree of a parsed document. A dependency tree illustrates the syntactic relationships between words in a sentence, showing how each word depends on or relates to other words. The arrows in the tree represent the grammatical relationships, such as subject, object, or modifier.

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/6d443359-66c0-43f8-ae3f-d088fade76e9)

Dependency Parsing- It is a technique that involves analyzing the grammatical structure of a sentence by identifying the syntactic dependencies between words.

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/05aebf1c-e31e-40a9-b596-a65fdde1f1f0)

Lemmatization- It is a linguistic process that involves reducing words to their base or root form, known as the lemma. In the context of natural language processing (NLP), lemmatization is often performed to standardize and simplify words, making it easier to analyze and compare them.

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/422a6e9b-ec7c-45a6-9d8f-885f66048eb6)

Sentence Boundary Detection- It is a NLP task that involves identifying and marking the boundaries between sentences in a given text. The primary goal of SBD is to determine where one sentence ends and the next one begins within a larger body of text.

Named Entity Recognition (NER)- It is a NLP task that involves identifying and classifying named entities (such as persons, organizations, locations, dates, and more) in a given text. The goal of (NER) is to extract structured information about entities mentioned in the text and assign them to predefined categories.

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/dab28191-ab9e-4963-b800-3270dcf0db0b)

Similarity- The similarity() method calculates a similarity score between two tokens based on their word vectors.

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/1c587eb6-6727-4e64-a646-7a21e8933862)

Tokenization- The user defined tokenize function serves as a preprocessing step for text data. It tokenizes the input text, lemmatizes the tokens, removes stopwords and punctuation, and returns a cleaned and processed string. This preprocessing is done before feeding text data into machine learning models or other natural language processing tasks. Using German stop words which can be used through the code-

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/52f7fbe4-68f4-4d58-bcb1-d789c0a8d0d0)

Custom Transformer- It is designed to clean and preprocess text data. When used in a machine learning pipeline, it can be applied to the input data during the transformation step. The cleaning function is applied to each element in the input data, resulting in a list of cleaned text data. This kind of text preprocessing is common
in NLP tasks to ensure consistency and improve the performance of machine learning models on text data.

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/df9471d0-fbaa-4383-9dea-b8aa247661e2)


Term Frequency-Inverse Document Frequency (TF-IDF):

Suppose we have a collection of documents, and we want to figure out which words are important in each document. TF-IDF helps us do that by highlighting words that are both frequently used within a document and not overly common across all documents. It helps identify the unique and significant words that can be used to characterize the content of each document.


Term Frequency (TF):

What it does: Measures how often a word appears in a document.

Why it matters: Words that appear more frequently in a document are often more important or relevant to the document's content.



Inverse Document Frequency (IDF):

What it does: Measures how unique or rare a word is across all documents.

Why it matters: Words that are common across all documents (like "the" or "and") may not provide much useful information. IDF gives less weight to these common words.



Combining TF and IDF:

What it does: TF-IDF is the product of TF and IDF.

Why it matters: TF-IDF gives high weight to words that are frequent in a specific document but not common across all documents. This helps identify words that are both important within a document and distinctive across the entire collection.


7- Splitting the dataset into train and test

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/9456ea0c-cb8e-494d-b5d6-e6ca7edbd11c)


8- Using the Model- Support Vector Classifier As the dataset has more than two labels which is 5 therefore, Naive Bayes or Logistic Regression will not be a good approach so SVC is used.

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/a9401730-3977-45e2-9b2f-47529db73cf7)


9- Creating NLP Pipeline

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/53984f4d-f320-4e7b-952a-0fc2082a2737)

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/f1fb3bed-077b-4f61-843b-b796f6646903)



10- Evaluating NLP pipeline

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/b13364cf-c2f4-4ccb-9535-184123f7147e)

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/a7c45d98-551d-49ae-b506-447a81af406e)



11- Plotting the Confusion matrix for each label

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/8b4edd72-095a-4064-aa64-ee159c1adc22)



12- Saving and downloading the created model through Joblib

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/741b8e2a-2a6f-424a-9608-45adceb3379a)



13- Saving and downloading the created tfidf_vectorizer through Joblib

![image](https://github.com/Anshuman-Singh432/AnsSin123/assets/47605733/e3b4c075-46cf-47e3-b7ab-31a14a4b0190)



## Integrating with Fast API-

1- A python script main.py is created integrating FAST API with the built model and using another python file custom_tranformer.py to preprocess the input given through FAST API and to predict the label class of input text.

## Dockerizing the built application-

1- Finally, Dockerizing the FastAPI application to simplify deployment and ensure consistent behaviour across different environments. A file named as “Dockerfile” is thus created specifying the dependencies, environment settings, and commands needed to run the application.
