# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: 
# Collaborators (discussion):
# Time:

import pylab
import re

# suppress warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pylab.RankWarning)

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = pylab.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

"""
End helper code
"""

def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    return [pylab.polyfit(x, y, z) for z in degs]

def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    SEE = ((y - estimated)**2).sum()
    mMean = y.sum()/len(y)
    MV = ((y - mMean)**2).sum()
    return 1 - SEE/MV

def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    estimated = PolyCoefficients(x, models)
    for i in range(len(estimated)):
        pylab.plot(x, y, 'bo', label='Measured points')
        pylab.title("Temperature vs. Time\n" + order_title(models[i])
                    + r"$R^{2}=$" + str(r_squared(y, estimated[i])*100//1/100)
                    + se_title(x, y, estimated[i], models[i]))
        pylab.xlabel('Year')
        pylab.ylabel('$Temperature\ (C\degree)$')
        pylab.plot(x, estimated[i], "r", label="Model")
    
        pylab.legend(loc = 'best')
        pylab.show()
        
def PolyCoefficients(x, coeffs):
    y = [0] * len(coeffs)
    for i in range(len(coeffs)):
        for j in range(len(coeffs[i])):
            y[i] += coeffs[i][j]*x**(len(coeffs[i])-1-j)
    return y

def order_title(model):
    num = len(model)-1
    if num % 10 == 1:
        return str(num) + "st-order: "
    elif num % 10 == 2:
        return str(num) + "nd-order: "
    elif num % 10 == 3:
        return str(num) + "rd-order: "
    return str(num) + "th-order: "

def se_title(x, y, estimated, model):
    if ((len(model)-1)%10) == 1:
        seos_value = str(se_over_slope(x, y, estimated, model)*100//1/100)
        return "\nse_over_slope: " + seos_value
    return ""

def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    cities_averages = [pylab.array([climate.get_yearly_temp(city, year)
        for city in multi_cities]).mean(axis=0).mean() for year in years]
    return pylab.array(cities_averages)

def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    average = [0 for _ in range(len(y))]
    for i in range(len(y)):
        for j in range(i+1 if i+1 < window_length else window_length):
            average[i] += y[i-j]
        average[i] /= j+1    
    return pylab.array(average)

def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """    
    return ((y - estimated) ** 2).mean() ** 0.5

def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """
    cities_std = [pylab.array([climate.get_yearly_temp(city, year)
        for city in multi_cities]).mean(axis=0).std() for year in years]
    return pylab.array(cities_std)

def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    estimated = PolyCoefficients(x, models)
    for i in range(len(estimated)):
        pylab.plot(x, y, 'bo', label='Measured points')
        pylab.title("Temperature vs. Time\n" + order_title(models[i])
                    + "RMSE=" + str(rmse(y, estimated[i])*100//1/100))
        pylab.xlabel('Year')
        pylab.ylabel('$Temperature\ (C\degree)$')
        pylab.plot(x, estimated[i], "r", label="Model")
    
        pylab.legend(loc = 'best')
        pylab.show()

if __name__ == '__main__':

    climate = Climate("./data.csv")
    training_interval = pylab.array(TRAINING_INTERVAL)
    testing_interval = pylab.array(TESTING_INTERVAL)

    # Part A.4.I
    # First, generate your data samples.
    sample = pylab.array([climate.get_daily_temp('NEW YORK', 1, 10, year)
              for year in TRAINING_INTERVAL])
    # Next, fit your data to a degree-one polynomial with generate_models,
    # and plot the regression results using evaluate_models_on_training.
    models = generate_models(training_interval, sample, [1])
    evaluate_models_on_training(training_interval, sample, models)
    
    # Part A.4.II
    # First, generate your data samples.
    sample = pylab.array([pylab.average(climate.get_yearly_temp
            ('NEW YORK', year)) for year in TRAINING_INTERVAL])
    # Next, fit your data to a degree-one polynomial with generate_models,
    # and plot the regression results using evaluate_models_on_training.
    models = generate_models(training_interval, sample, [1])
    evaluate_models_on_training(training_interval, sample, models)

    # Part B
    # compute national yearly temperature
    national_yearly_temps = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL)
    # fit your data to a degree-one polynomial with generate_models
    # and plot the regression results with evaluate_models_on_training.
    models = generate_models(training_interval, national_yearly_temps, [1])
    evaluate_models_on_training(training_interval, national_yearly_temps, models)

    # Part C
    # Use this function on the national yearly temperatures from 1961-2009 in
    # order to generate the moving average temperatures with a window size of 5.
    sample = moving_average(national_yearly_temps, 5)
    
    # Then, fit the (year, moving average) samples a to a degree-one
    # polynomial with g enerate_models, and plot the regression results with
    # evaluate_models_on_training.
    models = generate_models(training_interval, sample, [1])
    evaluate_models_on_training(training_interval, sample, models)

    # Part D.2
    # Compute 5-year moving averages of the national yearly temperature
    # from 1961-2009 as your training data samples.
    sample = moving_average(national_yearly_temps, 5)
    # Fit the samples to polynomials of degree 1, 2 and 20.
    models = generate_models(training_interval, sample, [1, 2, 20])
    # Use evaluate_models_on_training to plot your fitting results.
    evaluate_models_on_training(training_interval, sample, models)
    
    # Compute 5-year moving averages of the national yearly temperature
    # from 2010-2015 as your test data samples.
    national_yearly_temps = gen_cities_avg(climate, CITIES, TESTING_INTERVAL)
    sample = moving_average(national_yearly_temps, 5)
    # For each model obtained in the previous problem (i.e., the curves fit to
    # 5-year moving averages of the national yearly temperature from 1961-2009
    # with degree 1, 2, and 20), apply the model to your test data (defined
    # above), and graph the predicted and the real observations (i.e., 5-year
    # moving average of test data). You should use evaluate_models_on_testing
    # for applying the model and plotting the results.
    evaluate_models_on_testing(testing_interval, sample, models)
    
    # Part E
    # Use gen_std_devs to compute the standard deviations using all 21 cities
    # over the years in the training interval, 1961-2009.
    national_yearly_temps = gen_std_devs(climate, CITIES, TRAINING_INTERVAL)
    # Compute 5-year moving averages on the yearly standard deviations.
    sample = moving_average(national_yearly_temps, 5)
    # Finally, fit your data to a degree-one polynomial with generate_models
    # and plot the regression results with evaluate_models_on_training.
    models = generate_models(training_interval, sample, [1])
    evaluate_models_on_training(training_interval, sample, models)