# WAPE metric

WAPE stands for Weighted Absolute Percentage Error. It is a way to measure how far a set of forecasts is from what actually happened, expressed as a single percentage. It is popular in demand and sales forecasting.

To calculate it, take the absolute difference between the forecast and the actual value for each item or time period, and add those differences up. Then divide by the sum of the actual values. If total actual sales were 1,000 units and the forecasts were off by 120 units in total, the WAPE is 12%.

WAPE is often preferred over the simpler MAPE (mean absolute percentage error). MAPE averages the percentage error of each item separately, so a tiny item that sold 2 units but was forecast at 4 counts as a 100% error and can distort the result. WAPE weights every error by volume, so large items count more, and it does not break when some actual values are zero.

The trade-off is that WAPE can hide poor accuracy on small items, because they barely move the total. It also treats over-forecasting and under-forecasting the same, even though one may cost a business more than the other. It works best alongside other measures, such as a bias check that shows which direction the errors lean.
