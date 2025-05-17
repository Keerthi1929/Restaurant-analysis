from pandas import *
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns


df = read_csv('dataset.csv')  

# Task 1: Top Cuisines
# Count the frequency of each cuisine
top_cuisines = df['Cuisines'].value_counts().nlargest(3)

# Calculate the percentage of restaurants serving each top cuisine
total_restaurants = len(df)
cuisine_percentages = (top_cuisines / total_restaurants) * 100

# Print results
print(f"Top 3 Cuisines:{top_cuisines}")
print(f"\nPercentage of Restaurants Serving Each Cuisine:{cuisine_percentages}")



# Task 2: City Analysis
# City with the highest number of restaurants
city_counts = df['City'].value_counts()
city_with_most_restaurants = city_counts.idxmax()

# Average rating for restaurants in each city
average_ratings_by_city = df.groupby('City')['Aggregate rating'].mean()

# City with the highest average rating
city_with_highest_rating = average_ratings_by_city.idxmax()

# Print results
print(f"City with the most restaurants: {city_with_most_restaurants}")
print(f"\nAverage Ratings by City:{average_ratings_by_city}".encode('utf-8'))
print(f"\nCity with the highest average rating: {city_with_highest_rating}")



# Task 3: Price Range Distribution
# Create a histogram for price ranges
plt.figure(figsize=(10, 6))
plt.hist(df['Price range'], edgecolor='black', align='left')
plt.xlabel('Price Range')
plt.ylabel('Number of Restaurants')
plt.title('Distribution of Price Ranges')
plt.show()

# Calculate percentage of restaurants in each price range
price_range_percentages = df['Price range'].value_counts(normalize=True) * 100
print(f"\nPercentage of Restaurants in Each Price Range:{price_range_percentages}")

# Task 5: Restaurant Ratings Analysis
# Analyze the distribution of aggregate ratings
plt.figure(figsize=(8, 6))
sns.histplot(df['Aggregate rating'], bins=10, kde=True)
plt.title("Distribution of Aggregate Ratings")
plt.xlabel("Aggregate Rating")
plt.ylabel("Number of Restaurants")
plt.show()

# Calculate the average number of votes received by restaurants
avg_votes = df['Votes'].mean()
print(f"\nAverage number of votes received by restaurants: {avg_votes:.2f}")

# Calculate the most common rating range
most_common_rating = df['Aggregate rating'].mode()[0]
print(f"\nMost common rating range: {most_common_rating}")

# Task 5: Restaurant Ratings
# Analyze distribution of aggregate ratings
rating_distribution = df['Aggregate rating'].value_counts().sort_index()

# Most common rating range
most_common_rating_range = rating_distribution.idxmax()

# Average number of votes
average_votes = df['Votes'].mean()

# Print results
print(f"\nDistribution of Aggregate Ratings:{rating_distribution}")
print(f"\nMost Common Rating Range: {most_common_rating_range}")
print(f"Average Number of Votes: {average_votes:.2f}")



# Task 6: Cuisine Combination
# Identify the most common combinations of cuisines
cuisine_combinations = df['Cuisines'].value_counts().head(5)

# Analyze if certain cuisine combinations have higher ratings
cuisine_rating = df.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False)
print(f"\nTop cuisine combinations by average rating:{cuisine_rating.head(5)}")
print(f"\nMost common cuisine combinations:{cuisine_combinations}")



# Task 7: Geographic Analysis
# Plot restaurant locations on a map
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Longitude', y='Latitude', data=df, hue='City', alpha=0.6)
plt.title("Geographic Distribution of Restaurants")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Task 8: Restaurant Chains
# Identify restaurant chains
restaurant_chains = df['Restaurant Name'].value_counts()[lambda x: x > 1]

# Analyze ratings and popularity of chains
chain_ratings = df[df['Restaurant Name'].isin(restaurant_chains.index)]
average_chain_ratings = chain_ratings.groupby('Restaurant Name')['Aggregate rating'].mean()

# Print results
print(f"\nRestaurant Chains:{restaurant_chains}".encode("utf-8"))
print(f"\nAverage Ratings of Chains:{average_chain_ratings}".encode("utf-8"))




# Task 9: Restaurant Reviews
# Analyze text reviews
df['Review Length'] = df['Rating text'].apply(lambda x: len(str(x).split()))
average_review_length = df['Aggregate rating'].mean()

# Sentiment analysis
df['Sentiment'] = df['Rating text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Print results
print(f"\nAverage Review Length: {average_review_length:.2f} words")
print(f"\nSample Sentiment Scores:{df['Sentiment'].head()}")

# Task 10: Votes Analysis
# Identify restaurants with the highest and lowest number of votes
restaurant_max_votes = df.loc[df['Votes'].idxmax()]
restaurant_min_votes = df.loc[df['Votes'].idxmin()]

print(f"\nRestaurant with the highest votes:{restaurant_max_votes[['Restaurant Name', 'Votes']]}")
print(f"\nRestaurant with the lowest votes:{restaurant_min_votes[['Restaurant Name', 'Votes']]}")


# Analyze correlation between votes and rating
correlation = df['Votes'].corr(df['Aggregate rating'])
print(f"\nCorrelation between Votes and Rating: {correlation:.2f}")

# Step 1: Analyze relationship between price range and online delivery
online_delivery = df.groupby('Price range')['Has Online delivery'].value_counts(normalize=True) * 100

# Step 2: Analyze relationship between price range and table booking
table_booking = df.groupby('Price range')['Has Table booking'].value_counts(normalize=True) * 100

# Step 3: Print the results
print(f"Percentage of Restaurants Offering Online Delivery by Price Range:{online_delivery}")

print(f"\nPercentage of Restaurants Offering Table Booking by Price Range:{table_booking}")
