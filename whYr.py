import os
import urllib.request
import urllib.parse
import shutil
import datetime
import webbrowser
import numpy as np
import scipy.interpolate
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
from mpl_toolkits.basemap import Basemap
from matplotlib.widgets import CheckButtons, Slider, Button


class WeatherPoints:

    def __init__(self, n_periods, n_categories, n_points, period_length, weather_scorer):

        self.n_periods = n_periods
        self.n_categories = n_categories
        self.n_points = n_points
        self.period_length = period_length
        self.weather_scorer = weather_scorer

        self.period_length_timedelta = datetime.timedelta(seconds=self.period_length*3600)

        self.now = datetime.datetime.now()

        self.scores = np.zeros((5, self.n_points), dtype='float64')

    def create_points(self, places,
                            longitudes,
                            latitudes,
                            data,
                            n_times,
                            init_time,
                            final_time,
                            start_period):

        self.places = places
        self.longitudes = longitudes
        self.latitudes = latitudes
        self.data = data
        self.n_times = n_times
        self.init_time = init_time
        self.final_time = final_time
        self.start_period = start_period
        self.period_labels = self.__get_period_labels()

    def set_time_data(self, time_offset, duration, active_periods):

        self.time_offset = float(time_offset)
        self.duration = float(duration)
        self.active_periods = active_periods
        self.combined_period_label = self.__get_combined_period_label()

        start_time = self.now + datetime.timedelta(days=self.time_offset)
        end_time = start_time + datetime.timedelta(days=self.duration)

        if start_time < self.init_time:
            start_time = self.init_time

        if end_time > self.final_time:
            end_time = self.final_time

        start_time_offset = round((start_time - self.init_time).total_seconds()/(self.period_length*3600))
        end_time_offset = round((self.final_time - end_time).total_seconds()/(self.period_length*3600))

        self.start_time = self.init_time + datetime.timedelta(seconds=start_time_offset*self.period_length*3600)
        self.end_time = self.final_time - datetime.timedelta(seconds=end_time_offset*self.period_length*3600)

        if start_time_offset < 0:
            t_start = 0
        else:
            t_start = start_time_offset

        if end_time_offset < 0:
            t_end = self.n_times
        else:
            t_end = self.n_times - end_time_offset

        self.t_vals = [t for t in range(t_start, t_end) if (self.start_period + t) % self.n_periods in self.active_periods]

    def compute_scores(self):

        self.weather_scorer.score(self.scores, self.data, self.t_vals)

    def __get_period_labels(self):

        from_time = self.init_time

        period_labels = ['']*self.n_periods
        for i in range(self.n_periods):

            period = (self.start_period + i) % self.n_periods

            to_time = from_time + self.period_length_timedelta

            period_labels[period] = '{} to {}'.format(from_time.strftime('%H:%M'),
                                                      to_time.strftime('%H:%M'))

            from_time = to_time

        return period_labels

    def __get_combined_period_label(self):

        sorted_labels = sorted([self.period_labels[period] for period in self.active_periods])
        n_labels = len(sorted_labels)

        if n_labels == 0:
            combined_label = 'never'
        elif n_labels == 1:
            combined_label = sorted_labels[0]
        else:
            for i in range(n_labels-1, 0, -1):

                if sorted_labels[i-1][-5:] == sorted_labels[i][:5]:

                    sorted_labels[i-1] = sorted_labels[i-1][:5] + sorted_labels[i][5:]
                    sorted_labels.remove(sorted_labels[i])

            if len(sorted_labels) == 1 and sorted_labels[0][:5] == sorted_labels[0][-5:]:
                combined_label = 'all day'
            elif len(sorted_labels) == 1:
                combined_label = sorted_labels[0]
            else:
                combined_label = ' and '.join(sorted_labels)

        return combined_label


class WeatherScorer:

    def __init__(self, bad_precipitation=1,
                       bad_wind_speed=12,
                       ideal_temperature=20,
                       bad_temperature_diff=10,
                       overcast_weight=1,
                       precipitation_weight=1,
                       wind_speed_weight=1,
                       temperature_weight=1):

        self.bad_score = 10

        self.set_bad_precipitation(bad_precipitation)
        self.set_bad_wind_speed(bad_wind_speed)
        self.set_ideal_temperature(ideal_temperature)
        self.set_bad_temperature_diff(bad_temperature_diff)

        self.weights = np.array([overcast_weight, precipitation_weight, wind_speed_weight, temperature_weight], dtype='float64')
        self.__normalize_weights()

    def score(self, scores, data, t_vals):

        n_times = len(t_vals)
        scores[:, :] = 0.0

        for t in t_vals:

            scores[1, :] += self.__score_overcast(data[t, 0, :])
            scores[2, :] += self.__score_precipitation(data[t, 1, :])
            scores[3, :] += self.__score_wind_speed(data[t, 2, :])
            scores[4, :] += self.__score_temperature(data[t, 3, :])
        
        if n_times > 0:
            scores[1:, :] /= n_times
        else:
            scores[1:, :] = 0.0

        scores[0, :] = np.sum(np.broadcast_to(self.weights, (scores.shape[1], 4)).T*scores[1:, :], axis=0)

    def set_bad_precipitation(self, bad_precipitation):

        self.bad_precipitation = bad_precipitation
        self.precipitation_leeway = self.__get_precipitation_leeway(bad_precipitation)

    def set_bad_wind_speed(self, bad_wind_speed):

        self.bad_wind_speed = bad_wind_speed
        self.wind_speed_leeway = self.__get_wind_speed_leeway(bad_wind_speed)

    def set_ideal_temperature(self, ideal_temperature):

        self.ideal_temperature = ideal_temperature

    def set_bad_temperature_diff(self, bad_temperature_diff):

        self.bad_temperature_diff = bad_temperature_diff
        self.temperature_leeway = self.__get_temperature_leeway(bad_temperature_diff)

    def update_weights(self, overcast_weight, precipitation_weight, wind_speed_weight, temperature_weight):

        self.weights[0] = overcast_weight
        self.weights[1] = precipitation_weight
        self.weights[2] = wind_speed_weight
        self.weights[3] = temperature_weight
        self.__normalize_weights()

    def __normalize_weights(self):

        self.weights /= np.sum(self.weights)

    def __score_overcast(self, overcast):

        return overcast*10/3

    def __score_precipitation(self, precipitation):

        return (precipitation/self.precipitation_leeway)**2

    def __score_wind_speed(self, wind_speed):

        return (wind_speed/self.wind_speed_leeway)**2

    def __score_temperature(self, temperature):

        return ((temperature - self.ideal_temperature)/self.temperature_leeway)**2

    def plot_precipitation_scoring_func(self):

        precipitation = np.linspace(0, 5, 250)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(precipitation, self.__score_precipitation(precipitation))

        ax.set_xlabel('Precipitation [mm]')
        ax.set_ylabel('Score')
        ax.set_title('Precipitation scoring function')

        plt.show()

    def plot_wind_speed_scoring_func(self):

        wind_speed = np.linspace(0, 20, 250)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(wind_speed, self.__score_wind_speed(wind_speed))

        ax.set_xlabel('Wind speed [m/s]')
        ax.set_ylabel('Score')
        ax.set_title('Wind speed scoring function')
        
        plt.show()

    def plot_temperature_scoring_func(self):

        temperature = np.linspace(-10, 40, 250)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(temperature, self.__score_temperature(temperature))

        ax.set_xlabel('Temperature [C]')
        ax.set_ylabel('Score')
        ax.set_title('Temperature scoring function')
        
        plt.show()

    def __get_precipitation_leeway(self, bad_precipitation):

        return bad_precipitation/np.sqrt(self.bad_score)

    def __get_wind_speed_leeway(self, bad_wind_speed):

        return bad_wind_speed/np.sqrt(self.bad_score)

    def __get_temperature_leeway(self, bad_temperature_diff):

        return bad_temperature_diff/np.sqrt(self.bad_score)


class WeatherFinder:

    def __init__(self, places,
                       weather_scorer=WeatherScorer(),
                       from_file=False,
                       use_existing=False,
                       keep_files=False,
                       verbose=False):

        if from_file:
            places = WeatherFinder.__get_places_from_files(places)

        self.weather_points = WeatherPoints(4, 4, len(places), 6, weather_scorer)

        self.use_existing = use_existing
        self.keep_files = keep_files
        self.verbose = verbose

        self.numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.partly_clouded_nums = [3, 40, 5, 41, 24, 6, 25, 42, 7, 43,
                                    26, 20, 27, 44, 8, 45, 28, 21, 29]

        self.__init_weather_points(places)

        cdict = {'red':   ((0.0, 0.0, 0.0),
                           (0.5, 0.0, 0.0),
                           (1.0, 1.0, 1.0)),
                 'blue':  ((0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),
                 'green': ((0.0, 0.0, 1.0),
                           (0.5, 0.0, 0.0),
                           (1.0, 0.0, 0.0))}

        self.cmap = mpl_colors.LinearSegmentedColormap('green_to_red', cdict, 100)


        self.min_time = datetime.datetime.max
        self.max_time = datetime.datetime.min

    @staticmethod
    def __get_places_from_files(filenames):

        if isinstance(filenames, str):
            filenames = [filenames]

        lines = []

        for filename in filenames:

            f = open(filename, 'r', encoding='utf-8')
            lines += f.read().splitlines()
            f.close()

        return list(set([tuple(line.split(',')) for line in lines]))

    def __fetch_file(self, place):

        orig_url = 'http://www.yr.no/sted/{}/varsel.xml'.format('/'.join(place))
        url_parts = list(urllib.parse.urlsplit(orig_url))
        url_parts[2] = urllib.parse.quote(url_parts[2])
        url = urllib.parse.urlunsplit(url_parts)

        filename = os.path.join('forecasts', 'forecast_{}.xml'.format(place[-1]))

        if not self.use_existing or not os.path.exists(filename):

            self.__print('Downloading {} as {}...'.format(orig_url, filename), end=' ')

            with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

            self.__print('Done')

        return filename

    def __read_weatherdata(self, filename):

        xml = et.parse(filename)

        if not self.keep_files:

            self.__print('Deleting {}...'.format(filename), end=' ')

            os.remove(filename)

            self.__print('Done')

        weatherdata = xml.getroot()

        location = weatherdata.find('location').find('location')
        longitude = float(location.attrib['longitude'])
        latitude = float(location.attrib['latitude'])

        tabular = weatherdata.find('forecast').find('tabular')

        return longitude, latitude, tabular

    def __init_weather_points(self, places):

        longitudes = np.zeros(self.weather_points.n_points, dtype='float64')
        latitudes = np.zeros(self.weather_points.n_points, dtype='float64')

        longitudes[0], latitudes[0], tabular = self.__read_weatherdata(self.__fetch_file(places[0]))

        times = list(tabular.iter('time'))[2:-2]
        n_times = len(times)

        from_time = times[0].attrib['from']
        to_time = times[-1].attrib['to']

        data = np.zeros((n_times, self.weather_points.n_categories, self.weather_points.n_points), dtype='float64')

        for t in range(n_times):
            data[t, :, 0] = self.__get_category_values(times[t])

        init_time = WeatherFinder.__get_datetime(times[0].attrib['from'])
        final_time = WeatherFinder.__get_datetime(times[-1].attrib['to'])
        start_period = int(times[0].attrib['period'])

        for i in range(1, len(places)):

            longitudes[i], latitudes[i], tabular = self.__read_weatherdata(self.__fetch_file(places[i]))

            times = list(tabular.iter('time'))

            while times[0].attrib['from'] != from_time and len(times) > n_times:
                del times[0]
            while times[-1].attrib['to'] != to_time and len(times) > n_times:
                del times[-1]

            if times[0].attrib['from'] != from_time:
                raise ValueError('start times do not match: {} and {}'.format(from_time, times[0].attrib['from']))
            if times[-1].attrib['to'] != to_time:
                raise ValueError('end times do not match: {} and {}'.format(to_time, times[-1].attrib['to']))
            if len(times) != n_times:
                raise ValueError('number of time entries do not match')

            self.__print('Reading forecast for {}...'.format(places[i][-1]), end=' ')

            for t in range(n_times):
                data[t, :, i] = self.__get_category_values(times[t])

            self.__print('Done')

        self.weather_points.create_points(places,
                                          longitudes,
                                          latitudes,
                                          data,
                                          n_times,
                                          init_time,
                                          final_time,
                                          start_period)

    def __get_category_values(self, time):

        overcast = time.find('symbol').attrib['var']
        if not overcast[-1] in self.numbers:
            overcast = overcast[:-1]
        overcast = int(overcast)
        if overcast == 1 or overcast == 2:
            overcast -= 1
        elif overcast in self.partly_clouded_nums:
            overcast = 2
        else:
            overcast = 3

        precipitation = float(time.find('precipitation').attrib['value'])/self.weather_points.period_length
        wind_speed = float(time.find('windSpeed').attrib['mps'])
        temperature = float(time.find('temperature').attrib['value'])

        return overcast, precipitation, wind_speed, temperature

    @staticmethod
    def __get_datetime(time_str):

        return datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')

    def print_score_ranking(self, time_offset=0, duration=9.9, active_periods=[0, 1, 2, 3]):

        self.weather_points.set_time_data(time_offset, duration, active_periods)
        self.weather_points.compute_scores(time_offset, duration, active_periods)

        for i in np.argsort(self.weather_points.scores[0, :]):
            print('{:.3f} ({})'.format(self.weather_points.scores[0, :][i],
                                       self.weather_points.places[i][-1]))

    def plot_map(self, time_offset=0, duration=9.9, active_periods=[0, 1, 2, 3], map_resolution='i', extent=(3, 34, 55.9, 72), n_interp_points=(150, 150)):

        self.weather_points.set_time_data(time_offset, duration, active_periods)
        self.weather_points.compute_scores()

        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes([0.0, 0.005, 0.7, 0.995])

        self.__print('Drawing map...', end=' ')

        m = Basemap(projection='merc', resolution=map_resolution,
                    llcrnrlon=extent[0], urcrnrlon=extent[1],
                    llcrnrlat=extent[2], urcrnrlat=extent[3])

        m.drawcoastlines()
        m.drawcountries()
        m.drawmapboundary(fill_color='#97bdee')
        m.fillcontinents(color='#dfdfd7', lake_color='#97bdee')

        self.__print('Done')

        param_text_raw = 'Start time: {}\n' + \
                         'End time: {}\n' + \
                         'Period: {}\n'

        def get_updated_param_text():

            return param_text_raw.format(self.weather_points.start_time.strftime('%d/%m/%Y %H:%M'),
                                         self.weather_points.end_time.strftime('%d/%m/%Y %H:%M'),
                                         self.weather_points.combined_period_label)

        param_text = plt.text(0.01, 0.99, get_updated_param_text(),
                              horizontalalignment='left',
                              verticalalignment='top',
                              transform=ax.transAxes)

        if self.weather_points.n_points > 2 and self.weather_points.n_times > 0:

            self.__print('Interpolating scores...', end=' ')

            x0, y0 = m(extent[0], extent[2])
            x1, y1 = m(extent[1], extent[3])
            extent_xy = [x0, x1, y0, y1]

            xy = np.array(m(self.weather_points.longitudes, self.weather_points.latitudes)).T

            grid_xy = tuple(np.meshgrid(np.linspace(extent_xy[0], extent_xy[1], n_interp_points[0]),
                                        np.linspace(extent_xy[2], extent_xy[3], n_interp_points[1])))

            interpolated = scipy.interpolate.griddata(xy,
                                                      self.weather_points.scores[0, :],
                                                      grid_xy,
                                                      method='linear')

            im = m.imshow(interpolated,
                          cmap=self.cmap,
                          vmin = np.min(self.weather_points.scores[0, :]),
                          vmax = np.max(self.weather_points.scores[0, :]),
                          alpha=0.5,
                          zorder=2)

            self.__print('Done')

        self.__print('Drawing places...', end=' ')

        sc = m.scatter(*m(self.weather_points.longitudes, self.weather_points.latitudes),
                       c=self.weather_points.scores[0, :],
                       edgecolor='black',
                       linewidth=1,
                       cmap=self.cmap,
                       picker=True,
                       zorder=3)

        cb = m.colorbar(sc, location='right', pad='5%')

        radar_ax = plt.axes([0.15, 0.675, 0.1, 0.2], facecolor='none', polar=True)

        radar_angles = [2*np.pi*n/self.weather_points.scores.shape[0] for n in range(self.weather_points.scores.shape[0])]
        radar_angles += radar_angles[:1]

        radar_ax.set_theta_offset(0.5*np.pi)
        radar_ax.set_rlabel_position(0)

        radar_ax.xaxis.set_ticks(radar_angles[:-1])
        radar_ax.xaxis.set_ticklabels(['C', 'O', 'P', 'W', 'T'])
        radar_ax.yaxis.set_ticks(range(0, 11, 2))
        radar_ax.yaxis.set_ticklabels([])
        
        radar_ax.set_ylim(0, 10)

        radar_title = radar_ax.text(0.5, -0.03, '',
                                    horizontalalignment='center',
                                    verticalalignment='top',
                                    transform=radar_ax.transAxes)

        self.selected_idx = None
        self.radar_fill = []

        def update_radar(vmin, vmax):

            if self.selected_idx is None:
                return

            radar_values = list(self.weather_points.scores[:, self.selected_idx])
            radar_values += radar_values[:1]

            for patch in self.radar_fill:
                patch.remove()

            color_normalizer = mpl_colors.Normalize(vmin=vmin, vmax=vmax)

            self.radar_fill = radar_ax.fill(radar_angles, radar_values,
                                            color=self.cmap(color_normalizer(radar_values[0])),
                                            alpha=0.5,
                                            zorder=2)

        def onpick3(event):

            self.selected_idx = event.ind[0]
            update_radar(np.min(self.weather_points.scores[0, :]),
                         np.max(self.weather_points.scores[0, :]))
            radar_title.set_text(self.weather_points.places[self.selected_idx][-1])
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('pick_event', onpick3)

        self.__print('Done')

        slider_bg_color = 'lightgrey'
        slider_origin_x = 0.73
        slider_origin_y = 0.9
        slider_width = 0.17
        slider_height = 0.03
        slider_offset_y = 0.05
        slider_center = slider_origin_x + slider_width/2
        button_width = 0.07
        button_height = 0.03

        time_slider_origin_x = slider_origin_x - 0.3*slider_width
        chbuttons_origin_x = time_slider_origin_x + 1.35*slider_width
        chbuttons_width = 0.08
        chbuttons_height = 3*slider_offset_y

        timelabel_ax = plt.axes([slider_origin_x, slider_origin_y, slider_width, slider_height], facecolor='none'); slider_origin_y -= slider_offset_y
        timelabel_ax.axis('off')
        timeoffset_ax = plt.axes([time_slider_origin_x, slider_origin_y, slider_width, slider_height], facecolor=slider_bg_color); slider_origin_y -= slider_offset_y
        timeduration_ax = plt.axes([time_slider_origin_x, slider_origin_y, slider_width, slider_height], facecolor=slider_bg_color); slider_origin_y -= slider_offset_y
        periods_ax = plt.axes([chbuttons_origin_x, slider_origin_y, chbuttons_width, chbuttons_height])
        periods_ax.axis('off')
        resettime_ax = plt.axes([slider_center - button_width/2, slider_origin_y, button_width, button_height]); slider_origin_y -= slider_offset_y

        slider_origin_y -= slider_offset_y/2

        paramlabel_ax = plt.axes([slider_origin_x, slider_origin_y, slider_width, slider_height], facecolor='none'); slider_origin_y -= slider_offset_y
        paramlabel_ax.axis('off')
        badprec_ax = plt.axes([slider_origin_x, slider_origin_y, slider_width, slider_height], facecolor=slider_bg_color); slider_origin_y -= slider_offset_y
        badwind_ax = plt.axes([slider_origin_x, slider_origin_y, slider_width, slider_height], facecolor=slider_bg_color); slider_origin_y -= slider_offset_y
        idealtemp_ax = plt.axes([slider_origin_x, slider_origin_y, slider_width, slider_height], facecolor=slider_bg_color); slider_origin_y -= slider_offset_y
        badtemp_ax = plt.axes([slider_origin_x, slider_origin_y, slider_width, slider_height], facecolor=slider_bg_color); slider_origin_y -= slider_offset_y
        resetparam_ax = plt.axes([slider_center - button_width/2, slider_origin_y, button_width, button_height]); slider_origin_y -= slider_offset_y

        slider_origin_y -= slider_offset_y/2

        weightlabel_ax = plt.axes([slider_origin_x, slider_origin_y, slider_width, slider_height], facecolor='none'); slider_origin_y -= slider_offset_y
        weightlabel_ax.axis('off')
        overcweight_ax = plt.axes([slider_origin_x, slider_origin_y, slider_width, slider_height], facecolor=slider_bg_color); slider_origin_y -= slider_offset_y
        precweight_ax = plt.axes([slider_origin_x, slider_origin_y, slider_width, slider_height], facecolor=slider_bg_color); slider_origin_y -= slider_offset_y
        windweight_ax = plt.axes([slider_origin_x, slider_origin_y, slider_width, slider_height], facecolor=slider_bg_color); slider_origin_y -= slider_offset_y
        tempweight_ax = plt.axes([slider_origin_x, slider_origin_y, slider_width, slider_height], facecolor=slider_bg_color); slider_origin_y -= slider_offset_y
        resetweight_ax = plt.axes([slider_center - button_width/2, slider_origin_y, button_width, button_height]); slider_origin_y -= slider_offset_y

        slider_origin_y -= slider_offset_y/2

        hyperlink_ax = plt.axes([slider_center - button_width, slider_origin_y, button_width*2, button_height]); slider_origin_y -= slider_offset_y

        timelabel_ax.text(0.5, 0.0, 'Time', size='larger', horizontalalignment='center', verticalalignment='bottom', transform=timelabel_ax.transAxes)
        timeoffset_slider = Slider(timeoffset_ax, 'Offset', -2.0, 9.9, valfmt='%.1f days', valinit=self.weather_points.time_offset)
        timeduration_slider = Slider(timeduration_ax, 'Duration', 0.0, 9.0, valfmt='%.1f days', valinit=self.weather_points.duration)
        chbuttons_labels = ['Morning', 'Forenoon', 'Afternoon', 'Evening']
        active_periods_init = list(active_periods)
        chbuttons_actives = [period in self.weather_points.active_periods for period in range(self.weather_points.n_periods)]
        chbuttons_actives_init = list(chbuttons_actives)
        periods_chbuttons = CheckButtons(periods_ax, tuple(chbuttons_labels), tuple(chbuttons_actives))
        resettime_button = Button(resettime_ax, 'Reset', color=slider_bg_color)

        paramlabel_ax.text(0.5, 0.0, 'Parameters', size='larger', horizontalalignment='center', verticalalignment='bottom', transform=paramlabel_ax.transAxes)
        badprec_slider = Slider(badprec_ax, 'Bad precipitation', 0.0, 5.0, valfmt='%.1f mm/hr', valinit=self.weather_points.weather_scorer.bad_precipitation)
        badwind_slider = Slider(badwind_ax, 'Bad wind speed', 0.0, 20.0, valfmt='%.1f m/s', valinit=self.weather_points.weather_scorer.bad_wind_speed)
        idealtemp_slider = Slider(idealtemp_ax, 'Ideal temp.', -20.0, 40.0, valfmt='%.1f \u00b0C', valinit=self.weather_points.weather_scorer.ideal_temperature)
        badtemp_slider = Slider(badtemp_ax, 'Bad temp. diff.', 0.0, 20.0, valfmt='%.1f \u00b0C', valinit=self.weather_points.weather_scorer.bad_temperature_diff)
        resetparam_button = Button(resetparam_ax, 'Reset', color=slider_bg_color)

        weightlabel_ax.text(0.5, 0.0, 'Weights', size='larger', horizontalalignment='center', verticalalignment='bottom', transform=weightlabel_ax.transAxes)
        overcweight_slider = Slider(overcweight_ax, 'Overcast', 0.0, 100.0, valfmt='%.1f%%', closedmin=False, closedmax=False, valinit=self.weather_points.weather_scorer.weights[0]*100)
        precweight_slider = Slider(precweight_ax, 'Precipitation', 0.0, 100.0, valfmt='%.1f%%', closedmin=False, closedmax=False, valinit=self.weather_points.weather_scorer.weights[1]*100)
        windweight_slider = Slider(windweight_ax, 'Wind speed', 0.0, 100.0, valfmt='%.1f%%', closedmin=False, closedmax=False, valinit=self.weather_points.weather_scorer.weights[2]*100)
        tempweight_slider = Slider(tempweight_ax, 'Temperature', 0.0, 100.0, valfmt='%.1f%%', closedmin=False, closedmax=False, valinit=self.weather_points.weather_scorer.weights[3]*100)
        resetweight_button = Button(resetweight_ax, 'Reset', color=slider_bg_color)

        hyperlink_button = Button(hyperlink_ax, 'View on Yr.no', color=slider_bg_color)

        self.timeoffset_slider_id = 0
        self.timeduration_slider_id = 0
        self.periods_chbuttons_id = 0
        self.badprec_slider_id = 0
        self.badwind_slider_id = 0
        self.idealtemp_slider_id = 0
        self.badtemp_slider_id = 0
        self.overc_slider_id = 0
        self.prec_slider_id = 0
        self.wind_slider_id = 0
        self.temp_slider_id = 0

        def update():

            self.weather_points.compute_scores()

            vmin = np.min(self.weather_points.scores[0, :])
            vmax = np.max(self.weather_points.scores[0, :])

            if self.weather_points.n_points > 2 and self.weather_points.n_times > 0:

                im.set_data(scipy.interpolate.griddata(xy, self.weather_points.scores[0, :],
                                                        grid_xy, method='linear'))
                im.set_clim(vmin=vmin, vmax=vmax) 

            sc.set_array(self.weather_points.scores[0, :])

            cb.set_clim(vmin=vmin, vmax=vmax)

            param_text.set_text(get_updated_param_text())
            update_radar(vmin, vmax)

            cb.draw_all()
            fig.canvas.draw_idle()

        def time_update(val):

            self.weather_points.set_time_data(timeoffset_slider.val,
                                              timeduration_slider.val,
                                              [period for period in range(self.weather_points.n_periods) if chbuttons_actives[period]])
            update()

        def period_update(label):

            period = chbuttons_labels.index(label)
            chbuttons_actives[period] = not chbuttons_actives[period]
            time_update(0)

        def param_update(val):

            self.weather_points.weather_scorer.set_bad_precipitation(badprec_slider.val)
            self.weather_points.weather_scorer.set_bad_wind_speed(badwind_slider.val)
            self.weather_points.weather_scorer.set_ideal_temperature(idealtemp_slider.val)
            self.weather_points.weather_scorer.set_bad_temperature_diff(badtemp_slider.val)

            update()

        def weight_update():

            self.weather_points.weather_scorer.update_weights(overcweight_slider.val,
                                                              precweight_slider.val,
                                                              windweight_slider.val,
                                                              tempweight_slider.val)
            update()

        def overcast_weight_update(val):

            disconnect_weight_sliders(exclude=['overcast'])

            partial_sum = (precweight_slider.val + windweight_slider.val + tempweight_slider.val)/(100.0 - val)
            precweight_slider.set_val(precweight_slider.val/partial_sum)
            windweight_slider.set_val(windweight_slider.val/partial_sum)
            tempweight_slider.set_val(tempweight_slider.val/partial_sum)

            connect_weight_sliders(exclude=['overcast'])

            weight_update()

        def precipitation_weight_update(val):

            disconnect_weight_sliders(exclude=['precipitation'])

            partial_sum = (overcweight_slider.val + windweight_slider.val + tempweight_slider.val)/(100.0 - val)
            overcweight_slider.set_val(overcweight_slider.val/partial_sum)
            windweight_slider.set_val(windweight_slider.val/partial_sum)
            tempweight_slider.set_val(tempweight_slider.val/partial_sum)

            connect_weight_sliders(exclude=['precipitation'])

            weight_update()

        def wind_speed_weight_update(val):

            disconnect_weight_sliders(exclude=['wind speed'])

            partial_sum = (overcweight_slider.val + precweight_slider.val + tempweight_slider.val)/(100.0 - val)
            overcweight_slider.set_val(overcweight_slider.val/partial_sum)
            precweight_slider.set_val(precweight_slider.val/partial_sum)
            tempweight_slider.set_val(tempweight_slider.val/partial_sum)

            connect_weight_sliders(exclude=['wind speed'])

            weight_update()

        def temperature_weight_update(val):

            disconnect_weight_sliders(exclude=['temperature'])

            partial_sum = (overcweight_slider.val + precweight_slider.val + windweight_slider.val)/(100.0 - val)
            overcweight_slider.set_val(overcweight_slider.val/partial_sum)
            precweight_slider.set_val(precweight_slider.val/partial_sum)
            windweight_slider.set_val(windweight_slider.val/partial_sum)

            connect_weight_sliders(exclude=['temperature'])

            weight_update()

        def connect_time_widgets():

            self.timeoffset_slider_id = timeoffset_slider.on_changed(time_update)
            self.timeduration_slider_id = timeduration_slider.on_changed(time_update)
            self.periods_chbuttons_id = periods_chbuttons.on_clicked(period_update)

        def disconnect_time_widgets():

            timeoffset_slider.disconnect(self.timeoffset_slider_id)
            timeduration_slider.disconnect(self.timeduration_slider_id)
            periods_chbuttons.disconnect(self.periods_chbuttons_id)

        def connect_param_sliders():

            self.badprec_slider_id = badprec_slider.on_changed(param_update)
            self.badwind_slider_id = badwind_slider.on_changed(param_update)
            self.idealtemp_slider_id = idealtemp_slider.on_changed(param_update)
            self.badtemp_slider_id = badtemp_slider.on_changed(param_update)

        def disconnect_param_sliders():

            badprec_slider.disconnect(self.badprec_slider_id)
            badwind_slider.disconnect(self.badwind_slider_id)
            idealtemp_slider.disconnect(self.idealtemp_slider_id)
            badtemp_slider.disconnect(self.badtemp_slider_id)

        def connect_weight_sliders(exclude=[]):

            if not 'overcast' in exclude:
                self.overc_slider_id = overcweight_slider.on_changed(overcast_weight_update)
            if not 'precipitation' in exclude:
                self.prec_slider_id = precweight_slider.on_changed(precipitation_weight_update)
            if not 'wind speed' in exclude:
                self.wind_slider_id = windweight_slider.on_changed(wind_speed_weight_update)
            if not 'temperature' in exclude:
                self.temp_slider_id = tempweight_slider.on_changed(temperature_weight_update)

        def disconnect_weight_sliders(exclude=[]):

            if not 'overcast' in exclude:
                overcweight_slider.disconnect(self.overc_slider_id)
            if not 'precipitation' in exclude:
                precweight_slider.disconnect(self.prec_slider_id)
            if not 'wind speed' in exclude:
                windweight_slider.disconnect(self.wind_slider_id)
            if not 'temperature' in exclude:
                tempweight_slider.disconnect(self.temp_slider_id)

        def reset_time(event):

            disconnect_time_widgets()

            timeoffset_slider.reset()
            timeduration_slider.reset()

            for period in active_periods_init:
                if chbuttons_actives_init[period] != chbuttons_actives[period]:
                    periods_chbuttons.set_active(period)
                    chbuttons_actives[period] = not chbuttons_actives[period]

            time_update(0)

            connect_time_widgets()

        def reset_params(event):

            disconnect_param_sliders()

            badprec_slider.reset()
            badwind_slider.reset()
            idealtemp_slider.reset()
            badtemp_slider.reset()

            param_update(0)

            connect_param_sliders()

        def reset_weights(event):

            disconnect_weight_sliders()

            overcweight_slider.reset()
            precweight_slider.reset()
            windweight_slider.reset()
            tempweight_slider.reset()

            weight_update()

            connect_weight_sliders()

        def view_on_yr(event):

            if self.selected_idx is not None:
                webbrowser.open('https://www.yr.no/sted/{}/'.format('/'.join(self.weather_points.places[self.selected_idx])), new=2)

        connect_time_widgets()
        connect_param_sliders()
        connect_weight_sliders()

        resettime_button.on_clicked(reset_time)
        resetparam_button.on_clicked(reset_params)
        resetweight_button.on_clicked(reset_weights)
        hyperlink_button.on_clicked(view_on_yr)

        plt.show()

    def __print(self, *args, **kwargs):

        if self.verbose:
            print(*args, **kwargs, flush=True)


if __name__ == '__main__':

    f = open('whYr_config.ini', 'r')
    lines = [line.split('#')[0] for line in f.read().splitlines()]
    f.close()

    i = 0
    filenames = lines[i].replace(' ', '').split(','); i += 1
    use_existing = int(lines[i]); i += 1
    keep_files = int(lines[i]); i += 1
    extent = list(map(float, lines[i].split(','))); i += 1
    map_resolution = lines[i].strip(); i += 1
    n_interp_points = list(map(int, lines[i].split(','))); i += 1

    if not os.path.isdir('forecasts'):
        os.mkdir('forecasts')

    w = WeatherFinder(filenames, from_file=True, use_existing=use_existing, keep_files=keep_files, verbose=False)
    w.plot_map(n_interp_points=n_interp_points, map_resolution=map_resolution, extent=extent)
