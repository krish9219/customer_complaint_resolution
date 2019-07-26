# Customer_Complaint_Resolution

Customer resolutions is very important subject for any organisation. Customer satisfaction has impact on organisation's profit as well as reputation. In this project i will approach 2 methods for sloving this problem by using robust Machine Learning Models like Random Forest Model (RF) and Xtra Gradient Boosting Model (XGB). <br />

In the first method i will choose only 2 columns and convert custome complaint resolution problem in to Auto tagging problem. As we did in StackOverflow tag prediction project. But, here we will use Machine Learning Algorithms instead of Deep Learning. 

In the second method i will do the problem in hard approach (by using tfidi vectorizer, making more features from text etc.,) to predict the future customer resolutions based on there issues, products etc.

### Libraries Used

import pandas as pd <br />
import numpy as np <br />
from textblob import TextBlob, Word <br />
from nltk.stem import SnowballStemmer, WordNetLemmatizer <br />
from nltk.corpus import stopwords <br />
from sklearn.feature_extraction.text import TfidfVectorizer <br />
import matplotlib.pyplot as plt <br />
from sklearn.model_selection import train_test_split <br />
from sklearn.metrics import accuracy_score, confusion_matrix <br />
from sklearn.linear_model import LogisticRegression <br />
from sklearn.naive_bayes import MultinomialNB <br />
from sklearn.ensemble import RandomForestClassifier <br />
from sklearn.svm import SVC <br />
import xgboost <br />
from xgboost import XGBClassifier

***DATA***

__**Column names**__ = ['date_received', 'product', 'sub_product', 'issue', 'sub_issue',
       'consumer_complaint_narrative', 'company_public_response',
       'company', 'state', 'zipcode', 'tags', 'consumer_consent_provided',
       'submitted_via', 'date_sent_to_company',
       'company_response_to_consumer', 'timely_response',
       'consumer_disputed?', 'complaint_id']
       
data shape = (555957, 7)


