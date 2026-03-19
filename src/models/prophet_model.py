from prophet import Prophet
import pandas as pd


class ProphetModel:

    def __init__(self):
        self.model = Prophet()

    def train(self, df):

        prophet_df = pd.DataFrame()

        prophet_df["ds"] = df.index
        prophet_df["y"] = df["Adj Close"].values

        self.model.fit(prophet_df)

    def predict(self, periods=1):

        future = self.model.make_future_dataframe(periods=periods)

        forecast = self.model.predict(future)

        return forecast

        #market regime markov chain