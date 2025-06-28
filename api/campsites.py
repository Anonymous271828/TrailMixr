class Campsite:
    def __init__(self, index, distance_along_path, distance_from_trail):
        """
        init function for the Camsite class
        :param index: like the name or id of the campsite. idk, we need a system for this later.
        :param distance_along_path: distance that the campsite is along the trail
        :param distance_from_trail: shortest distance that the campsite is from the trail
        """
        self.index = index
        self.distance_along_path = distance_along_path
        self.distance_from_trail = distance_from_trail

    def __repr__(self):
        """
        used for testing
        :return: all the attributes of the Campsite
        """
        return f"Campsite(index={self.index}, distance_along_path={self.distance_along_path}, distance_from_trail={self.distance_from_trail})"

@DeprecationWarning
class Day:
    def __init__(self, sunset, sunrise, weatherscore, events):
        self.hours = {1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False, 8: False, 9: False,
                      10: False, 11: False, 12: False, 13: False, 14: False, 15: False, 16: False, 17: False, 18: False,
                      19: False, 20: False, 21: False, 22: False, 23: False}
        self.weatherscore = weatherscore
        self.stop = None

    def change_hour(self, hour, val):
        if hour in self.hours:
            self.hours[hour] = val
        else:
            raise ValueError("Hour must be between 1 and 23")

    def get_hour(self, hour):
        return self.hours.get(hour, False)

@DeprecationWarning
class Event:
    def __init__(self, index, distance_along_path, distance_from_trail, hours=None, location=None):
        self.name = index
        self.id = id
        self.hours = hours
        self.location = location

