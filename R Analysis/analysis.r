# Load necessary libraries
library(ggplot2)
library(readr)

# Load the data
data <- read_csv("Crop_recommendation.csv", show_col_type = FALSE)

# Scatter plot of temperature vs humidity colored by crop type
ggplot(data, aes(x = temperature, y = humidity, color = label)) +
  geom_point() +
  labs(title = "Temperature vs Humidity for different crops",
       x = "Temperature",
       y = "Humidity")

# Box plot for pH values across different crops
ggplot(data, aes(x = label, y = ph, fill = label)) +
  geom_boxplot() +
  labs(title = "pH distribution for different crops",
       x = "Crop",
       y = "pH")

# Histogram of rainfall
ggplot(data, aes(x = rainfall)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  labs(title = "Distribution of Rainfall",
       x = "Rainfall",
       y = "Frequency")