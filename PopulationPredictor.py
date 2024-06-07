# Storing online table
import pandas as pd
import ssl 
# Linear Regression Model imports
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

#Retrive the data from the website
print("loading population data...")
url = 'https://www.census.gov/data/tables/time-series/dec/popchange-data-text.html'
ssl._create_default_https_context = ssl._create_unverified_context
scraper = pd.read_html(url)
df = scraper[0]


    
