from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:

    def __init__(self, order=(1,0,1)):
        self.order = order
        self.model = None

    def train(self, series):

        self.model = ARIMA(series, order=self.order).fit()

    def predict(self, steps=1):

        forecast = self.model.forecast(steps=steps)

        return forecast