#!/usr/bin/env python
# coding: utf-8

# # PREDICTIVE MODELS WITH SKTIME 

# In[1]:


#pip install sktime[all_extras]
#pip install pmdarima


# In[13]:


#calling packages and dependencies that I go to use in this project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
import panel as pn
import pickle
import datetime as dt
import sktime as sk
from sktime.forecasting.arima import AutoARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.model_selection import train_test_split
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
import warnings
warnings.filterwarnings('ignore')


# In[14]:


#Loading data

data = pd.read_csv('train.csv', names = ['Date', 'y'], skiprows =1)

data


# In[15]:


#Descriptive statistics of the dependant variable

data['y'].describe()


# In[16]:


#Review the columns' types

data.info()


# In[17]:


#With this plot I can see the data distribution in the time serie

data.plot("Date", "y");


# In[18]:


#Changing the date format from object to datetime

data['Date'] = pd.to_datetime(data['Date'], format = '%d.%m.%y')

data


# In[19]:


#Using strfrime to format a date according to a specified format string.

data["Date"] = data["Date"].dt.strftime("%d.%m.%y")

data


# In[20]:


#Building the different subsets, train dataset and test dataset

train = data.iloc[:-12]

test = data.iloc[-12:]


# In[21]:


#Fit the first model

autoarima_model = AutoARIMA(suppress_warnings=True,
                           stepwise = True,
                           trace = True)

autoarima_model.fit(train["y"]) #Fit the model

forecast_steps = 12 #I select 12 steps because I want to predict the next year in twelve periods (months)

forecast = autoarima_model.predict(fh=pd.RangeIndex(start=1, 
                                                    stop=forecast_steps + 1)) #Generating the forecast model with AUTOARIMA 


# In[22]:


#Descriptive statistics of the predicted variable

forecast.describe()


# In[23]:


(autoarima_model.summary())


# # This model is a SARIMA model (0, 1, 2)
# 
# ### AR (0) = indicates that there is no autoregressive term in the model, meaning that the current value of the time series is not directly influenced by its own past values.
# 
# ### I (1) = indicates that the time series is differenced once to remove any non-stationarity or trend. Differencing involves subtracting the previous value from the current value, which helps to make the data more stationary.
# 
# ### MA (2) =  suggests that there are two rolling average terms in the model. Rolling average models use past forecast errors to predict future values. In this case, the model considers the previous two forecast errors when making predictions.

# In[24]:


# I've created a date range from the last train report data until the last data I wanted to predict

start_date = "2021-03-01"
end_date = "2022-02-01"
date_range = pd.date_range(start=start_date, end=end_date, freq="MS")


# In[25]:


# Joining the date range and predicted data to create the test document.

test["Date"] = date_range
test["Date"] = test["Date"].dt.strftime("%d.%m.%y")
test["y"] = forecast


# In[26]:


#I select 12 steps because I want to predict the next year in twelve periods (months)

forecast_steps = 12


# ##### The initial y's mean is relatively equal to the mean of data predict 2.47659 ~ 2.47404

# ##### I've decided choose AutoARIMA model because use a stepwise approach to search multiple combinations of p,d,q parameters and chooses the best model that has the least AIC.

# In[27]:


#Joining the train dataset with test dataset to obtain the last result

test = pd.concat([train, test], axis = 0)
test


# In[33]:


test.plot("Date", "y" ,title = 'Sales by Month (M€)', xlabel = 'Date', ylabel = 'y', figsize = (15, 7));


# In[30]:


# Write the results of test data in csv file 

test.to_csv("test.csv")


# # Predictive Models with PMDARIMA

# ### I've chosen the previous model because is the best selected model through AUTOARIMA PROCESS, but I've done different trials with different librarys. For the next steps I´ve used PMDARIMA library.

# In[31]:


data = test
data


# In[34]:


# Changing the date format from object to datetime

data['Date'] = pd.to_datetime(data['Date'].astype(str), format = '%d.%m.%y')
data


# In[35]:


# Transforming Date variable to index.

data = data.set_index(['Date'])
data


# In[38]:


data['rolling_avg'] = data['y'].rolling(window =12).mean() # Calculate Rolling Mean
data['rolling_std'] = data['y'].rolling(window =12).std() # Calculate Rolling Std


# Plot Rolling variables vs Original variable.

plt.figure(figsize = (15, 7))
plt.plot(data['y'], color = '#379BDB', label = 'Original')
plt.plot(data['rolling_avg'], color = '#D22A0D', label = 'Rolling Mean')
plt.plot(data['rolling_std'], color = '#142039', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Standard Deviation')
plt.show()


# In[37]:


ARIMA_model = pm.auto_arima(data['y'],
                        test = 'adf',
                        m = 1,
                        d = None,
                        trace = True,
                        suppress_warnings = True,
                        stepwise = True)


# In[62]:


(ARIMA_model.summary())


# # This model is a SARIMA model (0, 1, 2) (0, 0, 0)
# 
# ### AR (0) = indicates that there is no autoregressive term in the model, meaning that the current value of the time series is not directly influenced by its own past values.
# 
# ### I (1) = indicates that the time series is differenced once to remove any non-stationarity or trend. Differencing involves subtracting the previous value from the current value, which helps to make the data more stationary.
# 
# ### MA (2) =  suggests that there are two rolling average terms in the model. Rolling average models use past forecast errors to predict future values. In this case, the model considers the previous two forecast errors when making predictions.
# 
# ### (0, 0, 0) = indicates that there are no seasonal autoregressive, seasonal integrated, or seasonal moving average terms in the model.
# 
# #####  This model is suitable for time series data that exhibits a seasonal pattern and requires differencing to achieve stationarity. It uses past values and errors to forecast future values in the time series.

# In[41]:


#Diagnostics graphs of SARIMA (0 1 2) (0 0 0).

(ARIMA_model.plot_diagnostics(figsize =(15, 12)))
plt.show()


# In[44]:


SARIMA_model = pm.auto_arima(data['y'],
                            test = 'adf',
                            start_p = 1, start_q = 1,
                            max_p = 3, max_q = 3,
                            m = 12,
                            start_P = 0,
                            seasonal = True,
                            d = None,
                            D = 1,
                            trace = True,
                            error_action = 'ignore',
                            suppress_warnings = True,
                            stepwise = True)


# In[45]:


(SARIMA_model.summary())


# # This model is a SARIMA model (0, 1, 2) (0, 1, 1, [12])
# 
# ### AR (0) = indicates that there is no autoregressive term in the model, meaning that the current value of the time series is not directly influenced by its own past values.
# 
# ### I (1) = indicates that the time series is differenced once to remove any non-stationarity or trend. Differencing involves subtracting the previous value from the current value, which helps to make the data more stationary.
# 
# ### MA (2) =  suggests that there are two rolling average terms in the model. Rolling average models use past forecast errors to predict future values. In this case, the model considers the previous two forecast errors when making predictions.
# 
# ### (0, 1, 1) = indicates that there are no seasonal autoregressive, seasonal integrated, or seasonal moving average terms in the model.
# 
# ### Seasonal AR order (P) = 0, indicating no seasonal autoregressive terms.
# 
# ### Seasonal I order (D) = 1, suggesting that seasonal differencing is applied once.
# 
# ### Seasonal MA order (Q) = 1, signifying that one past seasonal forecast error is considered.
# 
# ### Seasonality (s) = 12, representing a seasonal period of 12 units.

# In[83]:


(SARIMA_model.plot_diagnostics(figsize =(15, 12)))
plt.show()


# ### To conclude the analysis, we would like to comment on the following: The reason why I have selected the first model (the one performed with SKTIME) is because once reviewed the diagnostic returned by the summary the AIC ratio was the lowest among all the models analyzed.
# 
# ### This ratio is Akaike Information Criterion, which is a measure of the relative quality of statistical models for a given set of data. "Minor AIC" refers to the goal of finding a statistical model that best fits the data

# In[ ]:




