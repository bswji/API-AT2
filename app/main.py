from fastapi import FastAPI,Path
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
from datetime import datetime
import numpy as np
import traceback
from datetime import datetime, timedelta

app = FastAPI()

@app.get("/")
def read_root():
    return {"Project objectives" : "This project aimed to create predictive models that are able to predict the sales revenue for a given item in a specific store at a given date and also a model that will forecast the total sales revenue across all stores and items for the next 7 days", 
            "List of endpoints": "/health will return a welcome message,/sales/national/<date> will forecast the total sales revenue over all stores. Simply replace <date> with a date in YYYY-mm-dd format. /sales/stores/items/<item_id>/<date>/<store_id> will return the sales revenue for a given item in a specific store. Replace the <item_id>, <date> and <store_id> with custom inputs.",
            "Outputs" : "/sales/stores/items/<item_id>/<date>/<store_id> will return a statement containing the predicted revenue. /sales/national/<date> will return a list containing each date and their respective forecasted total revenue. ",
            "Links to repositories": "API:https://github.com/bswji/API-AT2 and Modelling:https://github.com/bswji/Assignment2"
            }

@app.get("/health/")
def return_success():
    return {"message":"You have successfully connected to the API!"}, 200

def format_features(
    item_id: str,
    date: str,
    store_id: str,
    ):
    try:
        #Split date into year,month,day
        date = pd.to_datetime(date)
        year = date.year
        month = date.month
        day = date.day
        #Create dept_id, state_id, cat_id variables
        parts = item_id.split("_",2)
        dept_id = parts[0] + "_" + parts[1]
        cat_id = parts[0]
        store_parts = store_id.split("_")
        state_id = store_parts[0]
        #Create event type and event name counts
        calendar_events = load('../models/calendar_events.joblib')
        if date in calendar_events['date'].values:
            event_info = calendar_events[calendar_events['date'] == date]
            event_name = event_info['event_name'].values[0]
            event_type = event_info['event_type'].values[0]
        else:
            event_name = 'normalday'
            event_type = 'normalday'
        #Store in dict
        data =  {
            'item_id': [item_id],
            'dept_id': [dept_id],
            'cat_id': [cat_id],
            'store_id': [store_id],
            'state_id' : [state_id],
            'event_name': [event_name],
            'event_type': [event_type],
            'year': [year], 
            'month': [month],
            'day': [day]
        }
        data_df = pd.DataFrame(data)
        #Label encode cat cols
        label_encoder = load('../models/label_encoders.joblib')
        for column, label_encoder in label_encoder.items():
            if column in data_df.columns:
                data_df[column] = label_encoder.transform(data_df[column])
        return data_df

    except Exception as e:
        return {"error": str(e)}


@app.get('/sales/stores/items/{item_id}/{date}/{store_id}')
def predict(
    item_id: str = Path(...,description="Item ID"),
    date: str = Path(...,description="Date in YYYY-MM-DD"),
    store_id: str = Path(...,description="Store ID")
):
    try:
        #Add column names to dataframe contianing features
        features = format_features(item_id, date, store_id)
        obs = pd.DataFrame(features)
        # Make predictions
        model = load('../models/histmodel.joblib')
        preds = model.predict(obs)
        value = preds[0]
        return{"Output": f"Total revenue for {item_id} on {date} at {store_id} is {value}"}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

@app.get('/sales/national/{date}')
def national_sales(
    date: str = Path(...,description="Date in YYYY-MM-DD")
):
    try:
        date = pd.to_datetime(date)
        calendar_events = load('../models/calendar_events.joblib')
        #Store list of dates to forecast
        date_list = [date + timedelta(days=i) for i in range(7)]
        dates = pd.DataFrame(date_list, columns = ['date'])
        #Create date cols
        dates['date'] = pd.to_datetime(dates['date'])
        dates['month'] = dates['date'].dt.month
        dates['day'] = dates['date'].dt.day
        dates['quarter'] = dates['date'].dt.quarter
        dates['year'] = dates['date'].dt.year
        #Check if date has events
        combined_df = pd.merge(dates, calendar_events, on='date', how = 'left')
        combined_df['event_type'] = combined_df['event_type'].fillna("normalday")
        combined_df['event_name'] = combined_df['event_name'].fillna("normalday")
        #Label encode cat cols
        label_encoder = load('../models/label_encoders.joblib')
        for column, label_encoder in label_encoder.items():
            if column in combined_df.columns:
                combined_df[column] = label_encoder.transform(combined_df[column])
        combined_df = combined_df.drop('date', axis =1)
        #Create lag cols
        combined_df=combined_df.assign(lag=np.nan,lag2=np.nan)
        combined_df = combined_df[['year', 'month', 'day', 'quarter', 'event_name', 'event_type', 'lag', 'lag2']]
        numpy_array = combined_df.to_numpy()
        #Forecast revenue
        xgb_forecast = load('../models/xgbforecast.joblib')
        forecast1 = xgb_forecast.predict(numpy_array)
        df = pd.DataFrame(forecast1)
        df['date'] = date_list
        df.columns = ['forecast', 'date']
        df = df[['date','forecast']]
        df['forecast'] = df['forecast'].apply(lambda x: round(x, 2))
        results = df.to_dict(orient='records')
        return results
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}




