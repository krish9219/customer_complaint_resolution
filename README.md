# Customer_Complaint_Resolution



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
