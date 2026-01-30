from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm

sarima_model = SARIMAX(train,
                       order=(0,1,0),
                       seasonal_order=(0,1,1,0),
                       trend='n',
                       enforce_stationarity=False,
                       enforce_invertibility=False)


sarima_fit = sarima_model.fit(disp=False)

sarima_forecast = sarima_fit.predict()

##########
# ploting:
##########
plt.figure(figsize = (19, 9))
plt.plot(test, color='yellow')
plt.plot(sarima_forecast)