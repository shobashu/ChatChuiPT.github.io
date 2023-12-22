import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
import os
from geopy.geocoders import Nominatim
import folium
import pycountry
import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def text_to_df(file_path):
    """
    Convert .txt files to dataframes,
    which will then be converted to csv afterwards (using txt2csv)
    """
    # List to keep dictionaries for each beer
    beers_dic = []

    # A temporary dictionary to store data for each beer
    current_beer = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split the line using the first colon found
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                # Add/update the key in the current beer dictionary
                current_beer[key] = value
            # If you encounter an empty line, it signifies the end of a beer record
            if line.strip() == '':
                beers_dic.append(current_beer)
                current_beer = {}

    # Make sure to add the last beer if the file doesn't end with an empty line
    if current_beer:
        beers_dic.append(current_beer)

    # Create a DataFrame from the list of beer dictionaries
    return pd.DataFrame(beers_dic)

def txt2csv(path_in, path_out):
    
    """
    Converts the original files from .txt to .csv, by calling the function txt2csv
    """
    # Check for presence of 'ratings_ba_clean.csv'
    if not os.path.isfile(path_in):
        # Convert .txt to csv
        df = text_to_df(path_in)
        # Convert .txt to csv
        df.to_csv(path_out)
        return df
    else:
        print('.csv already present')
        return None
    

def csv2cache(df, path_in, cache_path):
    
    """
    Put the big .csv files in cache memory, to gain time
    when running the notebook
    """
    # Check for presence of 'ratings_ba.pkl' (BeerAdvocate)
    if not os.path.isfile(cache_path):
        if df == None:
            # Load the newly created .csv file
            df = pd.read_csv(path_in)

        # Cache the data
        pickle.dump(df, open(cache_path, 'wb'))
    else:
        print('file already loaded and cached')


def data_reading(FOLDER_BA, FOLDER_RB):
    
    """
    Read the relevant dataframes from BeerAdvocate and RateBeer
    Returns: the dataframes in pd.DataFrame
    """
    
    ratings_ba = txt2csv(FOLDER_BA + 'ratings.txt', FOLDER_BA + 'ratings_ba_clean.csv')
    ratings_rb = txt2csv(FOLDER_RB + 'ratings.txt', FOLDER_RB + 'ratings_rb_clean.csv')

    # Caching (run only once)
    csv2cache(ratings_ba, FOLDER_BA + 'ratings_ba_clean.csv', 'ratings_ba.pkl')
    csv2cache(ratings_rb, FOLDER_BA + 'ratings_rb_clean.csv', 'ratings_rb.pkl')

    # Loading data
    beers_ba = pd.read_csv(FOLDER_BA + 'beers.csv')
    breweries_ba = pd.read_csv(FOLDER_BA + 'breweries.csv')
    users_ba = pd.read_csv(FOLDER_BA + 'users.csv')
    ratings_ba = pickle.load(open('ratings_ba.pkl', 'rb'))

    beers_rb = pd.read_csv(FOLDER_RB + 'beers.csv')
    breweries_rb = pd.read_csv(FOLDER_RB + 'breweries.csv')
    users_rb = pd.read_csv(FOLDER_RB + 'users.csv')
    ratings_rb = pickle.load(open('ratings_rb.pkl', 'rb'))

    return beers_ba, breweries_ba, users_ba, ratings_ba, beers_rb, breweries_rb, users_rb, ratings_rb

def data_cleaning(user_ratings):
    """
    Perform data cleaning operations on the user ratings DataFrame.
    This function cleans the DataFrame by removing unnecessary columns, renaming columns for clarity,
    handling missing values in the 'abv' (alcohol by volume) column by imputing with the mean 'abv' of the corresponding style,
    and dropping rows with missing 'location' data.
    Args:
        user_ratings (pd.DataFrame): The DataFrame containing user ratings and other related information.
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Remove unnecessary columns
    user_ratings.drop(['Unnamed: 0', 'user_name_y'], axis=1, inplace=True)

    # Rename columns for clarity
    user_ratings.rename(columns={'user_name_x': 'user_name'}, inplace=True)

    # Calculate the percentage of missing values in the 'abv' column
    missing_values_abv = user_ratings['abv'].isna().mean() * 100

    # Impute missing 'abv' values with the mean 'abv' of the corresponding beer style
    style_mean_abv = user_ratings.groupby('style')['abv'].transform('mean')
    user_ratings['abv'].fillna(style_mean_abv, inplace=True)

    # Calculate the percentage of missing values in the 'location' column
    missing_values_location = user_ratings['location'].isna().mean() * 100

    # Drop rows where 'location' data is missing
    user_ratings = user_ratings.dropna(subset=['location'])

    return user_ratings

def data_pre_processing(data_to_merge1, data_to_merge2):
    """
        Merge users and ratings to obtain location of each rating especially and hange the date format and isolate month and year
        :param data_to_merge1: pd.DataFrame, typically users
        :param data_to_merge2: pd.DataFrame, typically ratings
        :return: pd.DataFrame
        """

    user_ratings = data_to_merge1.merge(data_to_merge2, how='right', on='user_id')
    user_ratings['date'] = pd.to_datetime(user_ratings['date'], unit='s')

    user_ratings['joined'] = pd.to_datetime(user_ratings['joined'], unit='s')

    # Create columns 'month', 'year' & 'year_month' on 'user_ratings' dataframe
    user_ratings['month'] = user_ratings['date'].dt.month
    user_ratings['year'] = user_ratings['date'].dt.year

    user_ratings['year_month'] = user_ratings['date'].dt.to_period('M')
    user_ratings = data_cleaning(user_ratings)
    return user_ratings


def extract_country(location):
    
    """
    Exctract only the country name from a string where other
    informations are displayed (e.g transform 'USA, Missouri' into 'USA')
    """
    
    if ',' in location:
        # If there is a comma in the location, split the string and take the first part
        return location.split(',')[0].strip()
    else:
        # If there is no comma, return the original location
        return location.strip()


def get_coordinates(country):
    """
    Obtain the coordinates to plot the geopy maps
    """
    
    # Initialize a geolocator using Nominatim with a specific user_agent
    geolocator = Nominatim(user_agent="geoapiExercices")
    try:
        # obtain the location (latitude and longitude) for the given country
        location = geolocator.geocode(country, language='en', timeout=1)
        return (location.latitude, location.longitude)
    except:
        return (None, None)


manual_mapping = {
    'England': 'GBR',
    'Russia': 'RUS',
    'Scotland': 'GBR',
    'Northern Ireland': 'GBR',
    'Taiwan': 'TWN',
    'Czech Republic': 'CZE',
    'Venezuela': 'VEN',
    'Turkey': 'TUR',
    'Aotearoa': 'NZL',
    'Svalbard and Jan Mayen Islands': 'SJM',
    'Bolivia': 'BOL',
    'Wales': 'GBR',
    'Vietnam': 'VNM',
    'Heard and McDonald Islands': 'HMD',
    'Fiji Islands': 'FJI',
    'Slovak Republic': 'SVK',
    'Macedonia': 'MKD',
    'Tanzania': 'TZA',
    'Moldova': 'MDA',
    'South Georgia and South Sandwich Islands': 'SGS',
    'Palestine': 'PSE',
    'Malvinas': 'FLK',
    'Sint Maarten': 'SXM',
}



def get_alpha3_code(country_name):
    
    """
    Convert country name to ISO3166-1-Alpha-3 code
    """
    # Check if the country is in the manual mapping
    if country_name in manual_mapping:
        return manual_mapping[country_name]

    # Try to get the code using pycountry
    try:
        return pycountry.countries.get(name=country_name).alpha_3
    except AttributeError:
        return None  # Handle cases where the country name is not found


def plot_map_ratings(user_ratings):
    """
    Create maps using geopy
    """
    # Count the number of ratings for each country
    country_counts = user_ratings['country'].value_counts().reset_index()
    # Rename columns
    country_counts.columns = ['country', 'count']
    # Add a new column proportion
    country_counts['proportion'] = round(100 * country_counts['count'] / country_counts['count'].sum(), 2)
    # Add a new column proportion
    country_counts['coord'] = country_counts['country'].apply(get_coordinates)
    country_counts.country = country_counts.country.apply(get_alpha3_code)
    # display(country_counts)
    # Initialize a Folium map with an initial center at latitude 0 and longitude 0
    m = folium.Map(location=[0, 0], zoom_start=1)

    # Iterate over each row in the country_counts dataFrame
    for _, row in country_counts.iterrows():
        # Check if coordinates for the country are available
        if row['coord'][0] is not None:
            # Add a Circle marker to the map for each country
            folium.Circle(
                location=row['coord'],
                radius=row['count'],
                color='crimson',
                fill=True,
                fill_color='crimson',
                popup='{}: {} %, {} ratings'.format(row['country'], row['proportion'], row['count'])
            ).add_to(m)

    return m


def plot_STL(ratings_per_month, type, plotTrend, plotSeasonality, plotResiduals):
    """
    Plot the general trends, the seasonal trends, and the noise for the evolution of normalized number of ratings of a beer subset
    
    ratings_per_month: Normalized number of ratings for each month, in an interval of years
    type: color of the plot
    plotTrend, plotSeasonality, plotResiduals: boolean values, to decide which subplot to display
    """
    # Apply Seasonal-Trend decomposition using LOESS (STL)
    stl = STL(ratings_per_month, seasonal=13, period=12)
    result = stl.fit()  # fit the model

    # Extract components from the decomposition
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    # Create 4 subplot figure
    plt.figure(figsize=(10, 6))

    if plotTrend == True:
        # Subplot 1: Trend
        plt.subplot(411)
        plt.plot(trend, label='Trend', color=type)
        plt.legend(loc='best')
        plt.grid()

    if plotSeasonality == True:
        # Subplot 2: Seasonality
        plt.subplot(412)
        plt.plot(seasonal, label='Seasonality', color=type)
        plt.legend(loc='best')
        plt.grid()

    if plotResiduals == True:
        # Subplot 3: Residuals
        plt.subplot(413, sharey=plt.gca())
        plt.plot(residual, label='Residuals', color=type)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid()

    # Subplot 4: Placeholder for potential additional plots
    plt.subplot(414)
    plt.axis('off')


def proportion_nbr_ratings(df, beer_subset, date_start, date_end, countries=[0]):
    """
    Given a subset of beers, a start date and end date, returns the normalized number of ratings per month
    (i.e. the number of ratings of the beer subset normalized according to the number of ratings for all beers)
    of the subset in the given period.

    df: global dataframe, considering all the beer ratings
    beer_subset: subset of beers (generally a subset of df)
    date_start: first date to consider
    date_end: last date to consider
    countries: country of origin of the considered ratings.
    """

    # filter the dataframe information from date_start to date_end for all the beers
    all_beers = df[
        (df['year'] >= date_start) &
        (df['year'] <= date_end)
        ]

    if (countries != [0]):
        all_beers = all_beers[all_beers['country'].isin(countries)]
        beer_subset = beer_subset[beer_subset['country'].isin(countries)]

    # for the beer subset
    beer_subset = beer_subset[
        (beer_subset['year'] >= date_start) &
        (beer_subset['year'] <= date_end)
        ]

    # Define the number of ratings per month for all beers around the world
    all_beer_ratings = all_beers.groupby('year_month')["rating"].count()

    # Number of ratings per month for the beer subset
    beer_subset_nbr_ratings_per_month = beer_subset.groupby('year_month')["rating"].count()

    # Proportion of the number of ratings per month
    beer_subset_prop_nbr_ratings = beer_subset_nbr_ratings_per_month / all_beer_ratings

    return beer_subset_prop_nbr_ratings


def feature_standardized(feature, df, beer_subset, date_start, date_end):
    """
    Given a subset of beers, a start date and end date, returns the standardized feature mean(rate, aroma, palate...) per month
    (i.e. z-scores), of the subset in the given period.
    
    feature: chosen characteristics (aroma, palate...)
    df: global dataframe, considering all the beer ratings
    beer_subset: subset of beers (generally a subset of df)
    date_start: first date to consider
    date_end: last date to consider
    """

    # filter the dataframe information from date_start to date_end
    all_beers = df[
        (df['year'] >= date_start) &
        (df['year'] <= date_end)
        ]

    beer_subset = beer_subset[
        (beer_subset['year'] >= date_start) &
        (beer_subset['year'] <= date_end)
        ]

    # Compute mean and variance of feature for the beer style, in the defined period
    mean_feature = beer_subset[feature].mean()
    std_feature = beer_subset[feature].std()

    # Mean feature value per month
    beer_subset_feature_per_month = beer_subset.groupby('year_month')[feature].mean()

    # z-score of the feature per month
    beer_subset_z_score = (beer_subset_feature_per_month - mean_feature) / std_feature

    return beer_subset_z_score


def plot_seasonal_trends(beer_feature, title, ylabel, color, month_increment=3, plotSTL=True, plotTrend=True,
                         plotSeasonality=True, plotResiduals=True):
    """
    Plot the seasonal trends, given roportion of ratings per month, or ratings per month...

    Given a pandas Series showing the feature of a beer subset per month (e.g. rates per month)
    returns plots showing the seasonal trend for this particular feature
    
    beer_feature: pandas Series with per month values
    title: title of the plot
    ylabel: label of the y axis, depending on the chosen feature (e.g. rate, or proportion of number of ratings)
    color: plot color
    month_increment: intervals of month to display. this only affects the labels, not the computation.
    """

    plt.figure(figsize=(14, 4))
    x = beer_feature.index.astype(str)
    plt.plot(x, beer_feature.values, marker='o', color=color)
    plt.xlabel('Month')
    plt.ylabel(ylabel)
    plt.title(title)

    # We show only labels by intervals of 3 months, to have a clearer visualisation
    plt.xticks(rotation=90, fontsize=9)
    tick_positions = range(0, len(x), month_increment)
    plt.xticks(tick_positions, [x[i] for i in tick_positions], rotation=45)

    plt.grid()
    plt.show()

    # Convert the index to timestamp. A new variable is created to avoid changing the original dataframe
    beer_feature_STL = beer_feature.copy()
    beer_feature_STL.index = beer_feature_STL.index.to_timestamp()

    if plotSTL == True:
        # Plot seasonal trends
        plot_STL(beer_feature_STL, color, plotTrend, plotSeasonality, plotResiduals)


def get_trend_seasons(df, beer_subset, trend_months=[0], no_trend_months=[0], date_start=2003, date_end=2016):
    """
    
    Returns two dataframe with the proportion of ratings by months: one for the 'trend months', and one for the 'no_trend_months'
    It allows us to keep the proportion od ratings for interesting months (e.g. winter months vs. summer months)
    
    df: global dataframe, considering all the beer ratings
    beer_subset: subset of beers (generally a subset of df)
    trend_months: if = [0], 3 months with higher normalized number of ratings
    no_trend_months: if = [0], 3 months with lower normalized number of ratings
    
    """
    # Define proportion of number of ratings of the beer subset
    prop_nbr_ratings_df = proportion_nbr_ratings(df, beer_subset, date_start, date_end).copy().reset_index()

    # Create column month
    prop_nbr_ratings_df['month'] = prop_nbr_ratings_df['year_month'].astype(str).str[-2:]

    # Rename column
    prop_nbr_ratings_df = prop_nbr_ratings_df.rename(columns={'rating': 'prop_nbr_of_ratings'})

    if (trend_months == [0]):
        by_month = prop_nbr_ratings_df.groupby('month')['prop_nbr_of_ratings'].mean()
        trend_months = by_month.nlargest(3).index.to_list()

    # Isolate rows corresponding to 'trend'
    trend_data = prop_nbr_ratings_df[prop_nbr_ratings_df['month'].isin(trend_months)]

    if (no_trend_months == [0]):
        by_month = prop_nbr_ratings_df.groupby('month')['prop_nbr_of_ratings'].mean()
        no_trend_months = by_month.nsmallest(3).index.to_list()

    # Isolate rows corresponding to 'trend' or 'no_trend'
    no_trend_data = prop_nbr_ratings_df[prop_nbr_ratings_df['month'].isin(no_trend_months)]

    return trend_data, no_trend_data


def boxplot_winter_vs_summer(summer_data, winter_data, beer_type_name):
    """
    
    Compute the p-value between the two datasets
    and plot the two boxplots corresponding to each dataset
    
    summer_data: normalized number of ratings of the beer_subset only for summer months
    winter_data: normalized number of ratings of the beer_subset only for winter months
    beer_type name: name of the beer type for the title
    
    """

    t_stat, p_value = ttest_ind(summer_data, winter_data)

    plt.figure(figsize=(5, 4))

    plt.boxplot([winter_data, summer_data], labels=['Winter', 'Summer'])
    plt.title(f'{beer_type_name} : Proportion of ratings during the summer vs winter')
    plt.xlabel('Season')
    plt.ylabel('Proportion of ratings')

    formatted_p_value = "{:.2e}".format(p_value)
    plt.text(1.5, max(max(winter_data), max(summer_data)), f'p-value: {formatted_p_value}', ha='center', va='bottom')

    plt.show()


def seasonality_degree(df, beer_subset, date_start=2003, date_end=2016):
    
    """
    
    Compute a 'degree' of seasonality: the higher the amplitude of the oscillations,
    the higher the seasonality degree. Value between from 0 to +infinity
    
    df: global dataframe, considering all the beer ratings
    beer_subset: subset of beers (generally a subset of df)
    date_start: first date to consider
    date_end: last date to consider
    
    """
    
    beer_subset_prop_nbr_ratings = proportion_nbr_ratings(df, beer_subset, date_start, date_end)

    stl = STL(beer_subset_prop_nbr_ratings, seasonal=13, period=12)
    result = stl.fit()  # fit the model

    # Extract components from the decomposition
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    # Mean Proportion of ratings for each year
    prop_ratings_by_year = beer_subset_prop_nbr_ratings.groupby(beer_subset_prop_nbr_ratings.index.year).mean()
    prop_ratings_by_year = prop_ratings_by_year.rename_axis('year')

    # Get the standard deviation of seasonal value per year, and the std value for the noise
    seasonal_df = seasonal.copy().reset_index()
    residual_df = residual.copy().reset_index()

    # Create column month
    seasonal_df['month'] = seasonal_df['year_month'].astype(str).str[-2:]
    residual_df['month'] = residual_df['year_month'].astype(str).str[-2:]

    # Create column year
    seasonal_df['year'] = seasonal_df['year_month'].astype(str).str[0:4]
    residual_df['year'] = residual_df['year_month'].astype(str).str[0:4]

    # Group by year and calculate std
    seasonal_std_per_year = seasonal_df.groupby('year')['season'].std()
    residual_std_per_year = residual_df.groupby('year')['resid'].std()

    # Change type of index to have int
    seasonal_std_per_year.index = seasonal_std_per_year.index.astype('int')
    residual_std_per_year.index = residual_std_per_year.index.astype('int')

    return (seasonal_std_per_year - residual_std_per_year) / (prop_ratings_by_year)


def plot_STL_pyplot(ratings_per_m, type, height=400, width=800, plotTrend=True, plotSeasonality=True,
                    plotResiduals=True):
    """
    Same as plot_STL(), but using Plotly
    
    ratings_per_month: Normalized number of ratings for each month, in an interval of years
    type: color of the plot
    plotTrend, plotSeasonality, plotResiduals: boolean values, to decide which subplot to display
    """
    ratings_per_month = ratings_per_m.copy()
    ratings_per_month.index = ratings_per_month.index.to_timestamp()

    # Apply Seasonal-Trend decomposition using LOESS (STL)
    stl = STL(ratings_per_month, seasonal=13, period=12)
    result = stl.fit()  # fit the model

    # Extract components from the decomposition
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    # Count the number of subplots to create
    num_subplots = sum([plotTrend, plotSeasonality, plotResiduals])

    # Create subplot figure with shared y-axis, dynamically adjusting the number of rows
    fig = make_subplots(rows=num_subplots, cols=1, shared_yaxes=True)

    # Counter variable to keep track of the current row
    current_row = 1

    if plotTrend:
        # Subplot: Trend
        fig.add_trace(go.Scatter(x=ratings_per_month.index, y=trend, mode='lines', name='Trend', line=dict(color=type)),
                      row=current_row, col=1)
        fig.update_yaxes(title_text="Trend", row=current_row, col=1)

        current_row += 1

    if plotSeasonality:
        # Subplot: Seasonality
        fig.add_trace(
            go.Scatter(x=ratings_per_month.index, y=seasonal, mode='lines', name='Seasonality', line=dict(color=type)),
            row=current_row, col=1)
        fig.update_yaxes(title_text="Seasonality", row=current_row, col=1)

        current_row += 1

    if plotResiduals:
        # Subplot: Residuals
        fig.add_trace(
            go.Scatter(x=ratings_per_month.index, y=residual, mode='lines', name='Residuals', line=dict(color=type)),
            row=current_row, col=1)
        fig.update_yaxes(title_text="Residuals", row=current_row, col=1)

    # Update layout
    fig.update_layout(height=height, width=width,
                      showlegend=False,
                      margin=dict(l=0, r=0, b=0, t=0))

    # Set shared y-axes for all subplots
    if plotTrend == False:
        for i in range(2, num_subplots + 1):
            fig.update_yaxes(matches='y', row=i, col=1)

    return fig


def plot_seasonal_trends_pyplot(beer_feature, title, ylabel, color, height=400, width=800):
    """
    Same as plot_seasonal_trends but using Plotly
    
    Given a pandas Series showing the feature of a beer subset per month (e.g. rates per month)
    returns plots showing the seasonal trend for this particular feature
    
    beer_feature: pandas Series with per month values
    title: title of the plot
    ylabel: label of the y axis, depending on the chosen feature (e.g. rate, or proportion of number of ratings)
    color: plot color
    month_increment: intervals of month to display. this only affects the labels, not the computation.
    """

    # Assuming beer_feature is a pandas Series
    x = beer_feature.index.astype(str)
    y = beer_feature.values

    # Create a line plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', marker=dict(color=color)))
    fig.update_layout(xaxis=dict(title='Month', tickangle=45, tickfont=dict(size=9)),
                      yaxis=dict(title=ylabel),
                      title=title,
                      showlegend=False,
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout(height=height, width=width, showlegend=False)

    # Show the plot
    return fig



def boxplot_plotly(summer_data, winter_data, name):
    
    """
    
    Same as boxplot_winter_vs_summer() but using Plotly
    
    summer_data: normalized number of ratings of the beer_subset only for summer months
    winter_data: normalized number of ratings of the beer_subset only for winter months
    beer_type name: name of the beer type for the title
    
    """
    
    # Assuming summer_data_lb and winter_data_lb are pandas DataFrames
    summer_data = summer_data['prop_nbr_of_ratings']
    winter_data = winter_data['prop_nbr_of_ratings']
    beer_type_name = name

    # Perform t-test
    t_stat, p_value = ttest_ind(summer_data, winter_data)

    # Create boxplot traces
    boxplot_trace_winter = go.Box(y=winter_data, boxpoints='all', jitter=0.3, pointpos=-1.8, name='Winter')
    boxplot_trace_summer = go.Box(y=summer_data, boxpoints='all', jitter=0.3, pointpos=1.8, name='Summer')

    # Create layout
    layout = go.Layout(
        title=f'{beer_type_name} normalized number of ratings: Summer vs Winter',
        title_font=dict(size=14),
        xaxis=dict(title='Season'),
        yaxis=dict(title='Normalized number of ratings'),
        height=400,  # Adjust the height as needed
        width=600
    )

    # Create figure
    fig = go.Figure(data=[boxplot_trace_winter, boxplot_trace_summer], layout=layout)

    # Add p-value annotation
    formatted_p_value = "{:.2e}".format(p_value)
    fig.add_annotation(
        x=0.5, y=1,
        text=f'p-value: {formatted_p_value}',
        showarrow=False, xref='paper', yref='paper', xanchor='center', yanchor='bottom'
    )

    # Save Plotly figure to HTML file
    return fig


# For seasonal estimation part
def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())


def seasonality_degree_2(df, ids, date_start=2003, date_end=2016):
    
    """
    
    Computes a new seasonality degree, but takes into account
    if the 'trend' is during the summer or the winter. Values from -infinity and +infinity.
    
    df: global dataframe, considering all the beer ratings
    ids: index of the chosen beers on the corresponding df 
    date_start: first date to consider
    date_end: last date to consider
    
    """
    
    degrees = np.array([])
    output = pd.DataFrame(columns=['beer_id', 'degree'])
    i = 0
    for id in ids:
        beer_subset_prop_nbr_ratings = proportion_nbr_ratings(df, df.loc[df.beer_id == id], date_start, date_end)
        # display(beer_subset_prop_nbr_ratings)
        stl = STL(beer_subset_prop_nbr_ratings, period=12)
        result = stl.fit()  # fit the model

        # Extract components from the decomposition
        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid

        if not np.all(np.isnan(seasonal)):
            # Mean Proportion of ratings for each year
            prop_ratings_by_year = beer_subset_prop_nbr_ratings.groupby(beer_subset_prop_nbr_ratings.index.year).mean()
            prop_ratings_by_year = prop_ratings_by_year.rename_axis('year')

            # Get the standard deviation of seasonal value per year, and the std value for the noise
            seasonal_df = seasonal.copy().reset_index()
            residual_df = residual.copy().reset_index()

            # Create column month
            seasonal_df['month'] = seasonal_df['year_month'].astype(str).str[-2:]
            residual_df['month'] = residual_df['year_month'].astype(str).str[-2:]

            # Create column year
            seasonal_df['year'] = seasonal_df['year_month'].astype(str).str[0:4]
            residual_df['year'] = residual_df['year_month'].astype(str).str[0:4]

            # Group by year and calculate std
            seasonal_std_per_year = seasonal_df.groupby('year')['season'].max() - seasonal_df.groupby('year')[
                'season'].min()
            residual_std_per_year = residual_df.groupby('year')['resid'].max() - residual_df.groupby('year')[
                'resid'].min()

            # Change type of index to have int
            seasonal_std_per_year.index = seasonal_std_per_year.index.astype('int')
            residual_std_per_year.index = residual_std_per_year.index.astype('int')

            # display(seasonal)
            highest_month = int(seasonal_df.loc[seasonal_df.season.idxmax(), 'month'])
            summer = -1 + (highest_month in range(4, 9)) * 2
            #print(highest_month, summer)
            degree = (abs(seasonal_std_per_year.mean() - residual_std_per_year.mean())) / (prop_ratings_by_year.mean()) * summer * 100
            #degree = (abs(seasonal_std_per_year.mean() - residual_std_per_year.mean())) * summer
            print(f"{i} out of {len(ids)}, degree: {degree}, month: {highest_month}, summer: {summer}")
            output.loc[i] = [id, degree]
            i += 1
    output = output.dropna()
    output['beer_id'] = output['beer_id'].astype(int)
    return output
