import os
import subprocess
import numpy as np
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import scipy.interpolate
import datetime


class WeatherPoints:

    def __init__(self):

        self.places = []
        self.longitudes = []
        self.latitudes = []
        self.scores = []

    def add_point(self, place, longitude, latitude, score):

        self.places.append(place)
        self.longitudes.append(longitude)
        self.latitudes.append(latitude)
        self.scores.append(score)


class WeatherScorer:

    def __init__(self, ideal_temperature=25, temperature_leeway=1, wind_speed_leeway=1, precipitation_leeway=1, weights={'s': 1, 'p': 1, 'w': 1, 't': 1}):

        self.ideal_temperature = ideal_temperature
        self.temperature_leeway = temperature_leeway
        self.wind_speed_leeway = wind_speed_leeway
        self.precipitation_leeway = precipitation_leeway

        weight_sum = np.sum(list(weights.values()))
        for name in weights:
            weights[name] /= weight_sum
        self.weights = weights

        self.numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.partly_clouded_nums = [3, 40, 5, 41, 24, 6, 25, 42, 7, 43, 26, 20, 27, 44, 8, 45, 28, 21, 29]

    def score(self, symbol_var, precipitation, wind_speed, temperature):

        return self.weights['s']*self.__score_symbol(symbol_var) \
            + self.weights['p']*self.__score_precipitation(precipitation) \
            + self.weights['w']*self.__score_wind_speed(wind_speed) \
            + self.weights['t']*self.__score_temperature(temperature)

    def __score_symbol(self, symbol_var):

        if not symbol_var[-1] in self.numbers:
            symbol_var = symbol_var[:-1]

        num = int(symbol_var)

        if num == 1:
            score = 0
        elif num == 2:
            score = 1
        elif num in self.partly_clouded_nums:
            score = 2
        else:
            score = 3

        return score*10/3

    def __score_precipitation(self, precipitation):

        return (precipitation/(4.5*self.precipitation_leeway))**2

    def __score_wind_speed(self, wind_speed):

        return (wind_speed/(4.5*self.wind_speed_leeway))**2

    def __score_temperature(self, temperature):

        return 10*(1 - np.exp(-((temperature - self.ideal_temperature)/(10*self.temperature_leeway))**2))


class WeatherFinder:

    def __init__(self, weather_scorer=WeatherScorer(), periods=[0, 1, 2, 3], offset=0, duration=10, use_existing=False, keep_files=False):

        self.weather_scorer = weather_scorer
        self.periods = periods
        self.use_existing = use_existing
        self.keep_files = keep_files

        cdict = {'red':   ((0.0, 0.0, 0.0),
                           (0.5, 0.0, 0.0),
                           (1.0, 1.0, 1.0)),
                 'blue':  ((0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),
                 'green': ((0.0, 0.0, 1.0),
                           (0.5, 0.0, 0.0),
                           (1.0, 0.0, 0.0))}

        self.cmap = mcolors.LinearSegmentedColormap('green_to_red', cdict, 100)

        offset, duration = self.__validate_days(offset, duration)
        now = datetime.datetime.now()
        self.start_date = now + datetime.timedelta(days=offset)
        self.end_date = self.start_date + datetime.timedelta(days=duration)

        self.weather_points = WeatherPoints()

    def add_weather_points(self, places):

        for place in places:
            self.add_weather_point(place)

    def add_weather_point(self, place):

        url = 'http://www.yr.no/sted/{}/varsel.xml'.format('/'.join(place))
        filename = 'varsel_{}.xml'.format(place[-1])

        if not self.use_existing or not os.path.exists(filename):
            subprocess.check_call(['wget', url, '-O', filename, '-q'])

        xml = et.parse(filename)

        if not self.keep_files:
            os.remove(filename)

        weatherdata = xml.getroot()

        forecast = weatherdata.find('forecast')
        tabular = forecast.find('tabular')

        total_score = 0.0
        count = 0

        for time in tabular.iter('time'):

            if not int(time.attrib['period']) in self.periods:
                continue

            from_time = self.__get_datetime(time.attrib['from'])
            to_time = self.__get_datetime(time.attrib['to'])

            if from_time > self.end_date or to_time < self.start_date:
                continue

            symbol_attrib = time.find('symbol')
            precipitation_attrib = time.find('precipitation')
            wind_speed_attrib = time.find('windSpeed')
            temperature_attrib = time.find('temperature')

            symbol_var = symbol_attrib.attrib['var']
            precipitation = float(precipitation_attrib.attrib['value'])
            wind_speed = float(wind_speed_attrib.attrib['mps'])
            temperature = float(temperature_attrib.attrib['value'])

            total_score += self.weather_scorer.score(symbol_var, precipitation, wind_speed, temperature)
            count += 1

        location = weatherdata.find('location')
        location_attrib = location.find('location')
        latitude = float(location_attrib.attrib['latitude'])
        longitude = float(location_attrib.attrib['longitude'])

        self.weather_points.add_point(place, longitude, latitude, total_score/count)

    def __get_datetime(self, time_str):

        return datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')

    def __validate_days(self, offset, duration):

        if offset < 0:
            offset = 0.0

        if duration < 0:
            duration = 0.0

        return float(offset), float(duration)

    def print_scores(self):

        for i in np.argsort(self.weather_points.scores):
            print('{:.3f} ({})'.format(self.weather_points.scores[i],
                                       self.weather_points.places[i][-1]))

    def plot_map(self, extent=(4, 33, 57, 72), n_interp_points=(150, 150)):

        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        m = Basemap(projection='merc', resolution='i',
                    llcrnrlon=extent[0], urcrnrlon=extent[1],
                    llcrnrlat=extent[2], urcrnrlat=extent[3])

        m.drawcoastlines()
        m.drawcountries()
        m.drawmapboundary(fill_color='#97bdee')
        m.fillcontinents(color='#dfdfd7', lake_color='#97bdee')

        x0, y0 = m(extent[0], extent[2])
        x1, y1 = m(extent[1], extent[3])
        extent_xy = [x0, x1, y0, y1]

        x, y = m(self.weather_points.longitudes, self.weather_points.latitudes)

        grid_x, grid_y = np.meshgrid(np.linspace(extent_xy[0], extent_xy[1], n_interp_points[0]),
                                     np.linspace(extent_xy[2], extent_xy[3], n_interp_points[1]))

        interpolated = scipy.interpolate.griddata(np.array([x, y]).T,
                                                  self.weather_points.scores,
                                                  (grid_x, grid_y),
                                                  method='linear')

        m.imshow(interpolated,
                 cmap=self.cmap,
                 alpha=0.5,
                 zorder=2)

        self.text = ax.text(0, -0.04, '', transform=ax.transAxes)

        def onpick3(event):

            i = event.ind[0]
            self.text.set_text('{}: {:.3f}'.format(self.weather_points.places[i][-1],
                                                   self.weather_points.scores[i]))
            plt.draw()

        sc = m.scatter(*m(self.weather_points.longitudes, self.weather_points.latitudes),
                       c=self.weather_points.scores,
                       edgecolor='black',
                       linewidth=1,
                       cmap=self.cmap,
                       picker=True,
                       zorder=3)

        fig.canvas.mpl_connect('pick_event', onpick3)

        m.colorbar(sc, location='right', pad='5%')

        plt.show()


if __name__ == '__main__':

    f = open('places.dat', 'r')
    lines = f.read().splitlines()
    f.close()

    places = [tuple(line.split(',')) for line in lines]

    w = WeatherFinder(periods=[1, 2], use_existing=True, keep_files=True)
    w.add_weather_points(places)
    #w.print_scores()
    w.plot_map(n_interp_points=(200, 200))
