import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# -----------------------------
# STEP 1: Raw Data
# -----------------------------

# Cumulative registrations
days = np.array([1, 2, 3, 4]).reshape(-1, 1)
cumulative = np.array([20, 140, 170, 220])

capacity = 500

# -----------------------------
# STEP 2: Polynomial Regression (Degree 2)
# -----------------------------

poly = PolynomialFeatures(degree=2)
days_poly = poly.fit_transform(days)

model = LinearRegression()
model.fit(days_poly, cumulative)

# -----------------------------
# STEP 3: Predict Smooth Curve (For Visualization Only)
# -----------------------------

future_days = np.arange(1, 15).reshape(-1, 1)
future_poly = poly.transform(future_days)
predicted_curve = model.predict(future_poly)

# Apply capacity cap to curve
predicted_curve = np.minimum(predicted_curve, capacity)

# -----------------------------
# STEP 4: Realistic Sellout Estimation (Rate-Based)
# -----------------------------

# Daily sales after early bird
daily_sales = np.diff(cumulative)  # [120, 30, 50]

# Post price-change daily sales (Day 3 & 4)
post_price_sales = daily_sales[1:]  # [30, 50]

average_daily_rate = np.mean(post_price_sales)

current_total = cumulative[-1]
remaining_tickets = capacity - current_total

days_needed = remaining_tickets / average_daily_rate
estimated_sellout_day = 4 + days_needed

# -----------------------------
# STEP 5: Revenue Calculation
# -----------------------------

early_bird_tickets = 140
early_bird_price = 399
regular_price = 499

regular_tickets = capacity - early_bird_tickets

total_revenue = (early_bird_tickets * early_bird_price) + \
                (regular_tickets * regular_price)

# -----------------------------
# STEP 6: Print Results
# -----------------------------

print("Average Daily Sales (Post Price Change):", round(average_daily_rate, 2))
print("Estimated Additional Days to Sell Out:", round(days_needed, 2))
print("Estimated Sellout Around Day:", round(estimated_sellout_day))
print("Predicted Total Revenue:", total_revenue)

# -----------------------------
# STEP 7: Plot
# -----------------------------

plt.scatter(days, cumulative, label="Actual Data")
plt.plot(future_days, predicted_curve, label="Polynomial Fit")
plt.axhline(y=capacity, linestyle='--', label="Capacity (500)")
plt.xlabel("Day")
plt.ylabel("Cumulative Registrations")
plt.legend()
plt.show()
