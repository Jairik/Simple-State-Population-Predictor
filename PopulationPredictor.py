# Storing online table
import pandas as pd
import numpy as np
import ssl 
# Linear Regression Model imports
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
# Visualizing data
import matplotlib.pyplot as plt

' --- Helper Functions --- '

def convert_percentage_to_float(x):
    if isinstance(x, str) and x.endswith('%'):
        return float(x.rstrip('%'))
    return x


#Clean the dataframe into a format readible by the model
def Clean_Data(df):
    #Transposing the df to flip rows and columns, making the data readible by the model
    m_df = df.set_index('State_or_Region').T.copy()
    #Adding 'previous population' and 'previous percent change' fields
    m_df['PREVIOUS_POPULATION'] = m_df['RESIDENT_POPULATION'].shift(-1)
    m_df['PREVIOUS_PERCENT_CHANGE'] = m_df['PERCENT_CHANGE'].shift(-1)
    m_df['RESIDENT_POPULATION'] = m_df['RESIDENT_POPULATION'].astype(int) 
    m_df = m_df.dropna()
    
    return m_df #Returning the modified dataframe


#Create a bar chart using matplotlib
def Visualize(df, new_pop):
    #Making a new dataframe for easy plotting
    df = df.reset_index()
    pop_data = df[['index', 'RESIDENT_POPULATION']]
    #Adding the new population
    new_elements = pd.DataFrame({
        'index': ['New Census'],
        'RESIDENT_POPULATION': [new_pop]
    })
    pop_data = pd.concat([new_elements, pop_data], ignore_index=True)
    print(pop_data)
    #Plotting the dataframe with a bar-chart
    pop_data.plot(x='index', y='RESIDENT_POPULATION', kind='bar', title='Population Over Years')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.show()
    #Plotting the dataframe with a line-chart
    pop_data.plot(x='index', y='RESIDENT_POPULATION', kind='line', title='Population Over Years')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.xticks(rotation=90) #Making the x-labels vertical
    plt.show()


   #Take in the dataframe of the state's population and population change as arguments
def Model_and_Visualize(orignal_df):    
    df = Clean_Data(orignal_df)
    #Extracting x and y
    x = df[['PREVIOUS_POPULATION', 'PREVIOUS_PERCENT_CHANGE']].values
    y = df['RESIDENT_POPULATION'].values
    #Fitting the model and predicting next year's population
    model = LinearRegression()
    model.fit(x, y)
    cur_year = df.index[0] #Getting the most recent year 
    cur_pop = df.at[cur_year, 'RESIDENT_POPULATION']
    cur_percent = df.at[cur_year, 'PERCENT_CHANGE']
    y_pred = model.predict([[cur_pop, cur_percent]])
    new_pop = y_pred[0]
    print("Next Predicted Population: ", new_pop)
    Visualize(df, new_pop)
    


#Retrive the data from the website
url = 'https://www.census.gov/data/tables/time-series/dec/popchange-data-text.html'
ssl._create_default_https_context = ssl._create_unverified_context
scraper = pd.read_html(url)
#Saving the scraper as a pandas dataframe
df = scraper[0]
#Cleaning the data
df = df.applymap(lambda x: x.replace(' ', '_') if isinstance(x, str) else x) #replace spaces with underscores
df.columns = df.columns.str.replace(' ', '_') #replacing the columns
df = df.applymap(convert_percentage_to_float) 
df['State_or_Region'] = df['State_or_Region'].str.upper()
print("---Population Data Loaded---")

state_to_find = 'S' #Some value other than Q
while(state_to_find != 'Q'):
    state_to_find = input("Enter the state/region to calculate (Q to quit): ")
    state_to_find = state_to_find.upper()

    row_indicies, col_indicies = np.where(df.values == state_to_find) #Finding the element
    
    if len(row_indicies) > 0: #If the element was found
        row_index = row_indicies[0]
        state_df = df.iloc[row_index + 1: row_index + 3].copy() #Save the data in a df as a copy
        #Run the Linear Regression Model
        Model_and_Visualize(state_df)
    elif state_to_find != 'Q':
        print("Invalid Input")
       
print("Thank you for using JJ's State Population Predictor!")