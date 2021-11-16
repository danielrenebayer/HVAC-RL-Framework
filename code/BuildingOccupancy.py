from abc import ABC, abstractmethod
import numpy as np
import datetime

"""
This class represents an individual person.
"""
class Person:
    # TODO: implement a schedule for every day, when the person is in the office
    #       the person should be there, if it has a meeting
    def __init__(self, name, gender, default_office, comfort_temp = None):
        self.name = name
        self.gender = gender
        self.default_office = default_office
        self._meetings = []
        self.presence_start = datetime.time(hour=8, minute=0) #hour=np.random.randint(8,11), minute=0)
        self.presence_end   = datetime.time(hour=self.presence_start.hour + 8, minute=0)
        
        if comfort_temp is None:
            self.comfort_temp = 20.5 if gender == "m" else 21.5
        else:
            self.comfort_temp = comfort_temp
    
    def __str__(self):
        return f"{self.name}, {self.gender}, {self.default_office}, {self.comfort_temp}"
    
    def __repr__(self):
        return f"Person(name='{self.name}', gender='{self.gender}', default_office='{self.default_office}', comfort_temp={self.comfort_temp})"
    
    def add_meeting(self, meeting):
        self._meetings.append(meeting)

    """
    Stores the object as a dictionary
    """
    def to_dict(self):
        # TODO: implement meeting storing
        return {"name": self.name,
                "gender": self.gender,
                "default_office": self.default_office,
                "meetings": []}
    
    """
    Get the meeting for this person at a given date and a given time
    If there is no meeting scheduled for the person, it returns none
    
    Parameters
    ----------
    day : datetime.date
        the date for the search
    time : datetime.time
        the time of the query
    """
    def get_meeting_at(self, date, time):
        for meeting in self._meetings:
            if time >= meeting.time_start and time <= meeting.time_end:
                if type(meeting) is WeeklyMeeting:
                    if meeting.day_of_week == date.weekday():
                        return meeting
                else:
                    if meeting.day == date:
                        return meeting
        return None

    """
    Gets the meetings for this person during a given timespan.
    If there are no meetings scheduled for the person, it returns an empty list.
    
    Parameters
    ----------
    day : datetime.date
        the date for the search
    ts_start : datetime.time
        the start time of the timespan
    ts_end : datetime.time
        the end time of the timespan
    """
    def get_meetings_during(self, date, ts_start, ts_end):
        meetings = []
        for meeting in self._meetings:
            if ts_start < meeting.time_end and ts_end > meeting.time_start:
                if type(meeting) is WeeklyMeeting:
                    if meeting.day_of_week == date.weekday():
                        meetings.append(meeting)
                else:
                    if meeting.day == date:
                        meetings.append(meeting)
        return meetings


"""
A meeting has to end at the end of a day, i.e. 23:59:59 and cannot start before 00:00:00.
"""
class Meeting(ABC):
    def __init__(self, room, time_start, time_end):
        self.room = room
        self.time_start = time_start
        self.time_end   = time_end
        self._participants = []
    
    def get_participants(self):
        return self._participants

    """
    Adds a participant (Person) to the meeting
    Ir returns true on success, false if the person cannot attend the meeting, because of another meeting
    """
    @abstractmethod
    def add_participant(self, person): pass
    
    """
    Get all dates as datetime.datetime-object, where this meeting takes place in the form of a list
    containing the tupels (day, start time, end time)
    """
    @abstractmethod
    def get_all_dates(self): pass

"""
This class represents a weekly meeting.
A meeting has to end at the end of a day, i.e. 23:59:59 and cannot start before 00:00:00.
"""
class WeeklyMeeting(Meeting):
    """
    Constructs a new weekly meeting object
    
    Parameters
    ----------
    room : str
        The name of the room, where the meeting will be
    day_of_week : int
        The day of the week (as int, starting with monday as 0)
    time_start : datetime.time
        The start of the date
    time_end : datetime.time
        The end time of the date
    """
    def __init__(self, room, day_of_week, time_start, time_end):
        super().__init__(room, time_start, time_end)
        self.day_of_week = int(day_of_week)

    """
    Implements Meeting.add_participant(person)
    """
    def add_participant(self, person):
        # iterate over all days of this meeting
        for meetingday in self.get_all_dates():
            if len(person.get_meetings_during(meetingday[0], self.time_start, self.time_end)) > 0:
                return False
        person._meetings.append(self)
        self._participants.append(person)
        return True

    """
    Implements Meeting.get_all_dates()
    """
    def get_all_dates(self):
        all_dates = []
        # find first day of the year, that ist the correct day of the week
        first_day_in_year = datetime.date(2017,1,1)
        weekdays_diff = self.day_of_week - first_day_in_year.weekday()
        if weekdays_diff > 0:
            first_day_in_year = first_day_in_year + datetime.timedelta(days=weekdays_diff)
        elif weekdays_diff < 0:
            first_day_in_year = first_day_in_year + datetime.timedelta(days=7+weekdays_diff)
        # iterate over all weeks
        for weekn in range(0, 53):
            curr_day = first_day_in_year + datetime.timedelta(weeks=weekn)
            if curr_day.year > 2017:
                break
            all_dates.append((curr_day, self.time_start, self.time_end))
            #all_dates.append((datetime.datetime.combine(curr_day, self.time_start),
            #                  datetime.datetime.combine(self.day, self.time_end)))
        return all_dates

    def __str__(self):
        return f"Weekly meeting in room {self.room} at weekday {self.day_of_week} from {self.time_start} until {self.time_end}"
    
    def __repr__(self):
        return f"WeeklyMeeting(room='{self.room}', day_of_week={self.day_of_week}, time_start={self.time_start}, time_end={self.time_end})"


"""
This class represents a meeting, that happens only once.
A meeting has to end at the end of a day, i.e. 23:59:59 and cannot start before 00:00:00.
"""
class OneTimeMeeting(Meeting):
    """
    Constructs a new one-time meeting object
    
    Parameters
    ----------
    room : str
        The name of the room, where the meeting will be
    day : datetime.date
        The date of the meeting as datetime.date-object
    time_start : datetime.time
        The start of the date
    time_end : datetime.time
        The end time of the date
    """
    def __init__(self, room, day, time_start, time_end):
        super().__init__(room, time_start, time_end)
        self.day = day

    """
    Implements Meeting.add_participant(person)
    """
    def add_participant(self, person):
        if len(person.get_meetings_during(self.day, self.time_start, self.time_end)) > 0:
            return False
        person._meetings.append(self)
        self._participants.append(person)
        return True

    """
    Implements Meeting.get_all_dates()
    """
    def get_all_dates(self):
        return [(self.day, self.time_start, self.time_end)]
        #return [(datetime.datetime.combine(self.day, self.time_start),
        #         datetime.datetime.combine(self.day, self.time_end))]

    def __str__(self):
        return f"One-time meeting in room {self.room} at {self.day} from {self.time_start} until {self.time_end}"
    
    def __repr__(self):
        return f"OneTimeMeeting(room='{self.room}', day={self.day}, time_start={self.time_start}, time_end={self.time_end})"


class BuildingOccupancy:
    
    """
    Constructs a new building occupancy controller
    
    Parameters
    ----------
    building_name : str
        Name of the building
    """
    def __init__(self, building_name=""):
        self._initialized_rooms = False
        self.occupants = []
        self._weekly_meetings = []
        self._onetime_meetings = []
        self.building_name = building_name
        self.bank_holidays = [(1,1), (1,6), (12,24), (12,25), (12,26), (12,31)]
        self.maximal_numer_of_people_in_office_room = 50.0
        # TODO: maximal_numer_of_people_in_office_room fuer jeden office raum getrennt vorgeben


    """
    This method defines the room settings of the building, it cannot be called twice.
    
    Parameters
    ----------
    office_rooms : int or list of string or dict
        If rooms is an integer, it will create rooms rooms with the name "OfficeRoom x" for x in range(rooms).
        If romms is a list of strings, it will create the rooms with the given names.
        If rooms is a dict (Roomname: Max. persons), it will create the rooms with the given names, and the maximum ammount of people inside
        An office room represents a open space office (i.e. bullpen) and a single person office room.
    conference_rooms : int or dict
        If rooms is an integer, it will create rooms rooms with the name "ConferenceRoom x" for x in range(rooms).
        If romms is a dict (Roomname: Max. persons), it will create the rooms with the given names, and the maximum ammount of people inside.
    """
    def set_room_settings(self, office_rooms, conference_rooms, conference_rooms_default_max_pers = 15):
        if self._initialized_rooms:
            raise Exception("This building occupancy object is already initialized.")
        self._initialized_rooms = True
        
        if type(office_rooms) == int:
            self.office_rooms = [f"OfficeRoom {x}" for x in range(office_rooms)]
        elif type(office_rooms) == list:
            self.office_rooms = {room: conference_rooms_default_max_pers for room in office_rooms}
        else:
            self.office_rooms = office_rooms

        if type(conference_rooms) == int:
            self.conference_rooms = {f"ConferenceRoom {x}": conference_rooms_default_max_pers
                                                        for x in range(conference_rooms)}
        else:
            self.conference_rooms = conference_rooms


    """
    Add occupants to the model
    """
    def add_occupants(self, occupant_list):
        for p in occupant_list:
            if not type(p) is Person:
                raise("Every element in the list has to be of type Person.")
            self.occupants.append(p)


    """
    Generates random occupants, if the list is empty
    It returns the number of generated occupants
    """
    def generate_random_occupants(self, number_occupants):
        if not self._initialized_rooms:
            raise Exception("This building occupancy object is not initialized.")
        if len(self.occupants) != 0:
            return 0

        # count the number of people inside an office
        free_people_in_office_room = {room: maxpers - len([pers for pers in self.occupants if pers.default_office == room]) for room, maxpers in self.office_rooms.items()}
        for i in range(number_occupants):
            if len(free_people_in_office_room.keys()) <= 0:
                break
            #pDefaultOff = np.random.choice(list(free_people_in_office_room.keys()))
            pDefaultOff = list(free_people_in_office_room.keys())[0]
            free_people_in_office_room[pDefaultOff] -= 1
            if free_people_in_office_room[pDefaultOff] <= 0:
                del free_people_in_office_room[pDefaultOff]
            pName       = f"Person {i}"
            pGender     = "m" if np.random.random() < 0.5 else "w"
            pComfortTemp= 21.3 #np.random.normal(21.3 if pGender == "w" else 20.3, 0.4)
            self.occupants.append(Person( pName, pGender, pDefaultOff, pComfortTemp ))

        return number_occupants


    """
    Generates random meetings and add existing people to those newly generated meetings.
    It returns a tuple, containing the number of created weekly meetings, and the number of created onetime meetings.
    It may occurs, that a meeting has less than min_people_joining_meeting participants
    
    Parameters
    ----------
    number_meetings_per_week : int
        Number of generated meetings, that occure every week
    number_onetime_meetings : int
        Number of generated meetings, that happens only once in a year
    min_people_joining_meeting : int
        Minimal number of people joining a meeting
        Defaults to 3
    max_peope_joining_meeting : int
        Maximal number of people joining a meeting
        Defaults to 10
    """
    def generate_random_meetings(self,
                                 number_meetings_per_week,
                                 number_onetime_meetings,
                                 min_people_joining_meeting=3, max_peope_joining_meeting=10,
                                 conference_room_names = None):
        generated_weekly_meetings = 0
        generated_onetime_meetings = 0

        # generate weekly meetings
        if conference_room_names is None:
            conference_room_names = list(self.conference_rooms.keys())
        selected_room = np.random.choice(conference_room_names, number_meetings_per_week)
        selected_dow  = np.random.randint(0,  5, number_meetings_per_week)
        selected_ts_h = np.random.randint(7, 15, number_meetings_per_week)
        selected_ts_m = np.random.randint(0, 60, number_meetings_per_week)
        for n in range(number_meetings_per_week):
            te_h = np.random.randint(selected_ts_h[n] + 2,
                                     min(selected_ts_h[n] + 5, 20))
            te_m = np.random.randint(0, 60)
            ts   = datetime.time(selected_ts_h[n], selected_ts_m[n])
            te   = datetime.time(te_h, te_m)
            new_meeting = self.add_weekly_meeting(selected_room[n], selected_dow[n], ts, te)
            if not new_meeting is None:
                generated_weekly_meetings += 1
                # add people
                for _ in range(np.random.randint(min_people_joining_meeting, max_peope_joining_meeting)):
                    # if a person cannot attend the meeting, try it at maximum 3 to times
                    if not new_meeting.add_participant( np.random.choice(self.occupants) ):
                        if not new_meeting.add_participant( np.random.choice(self.occupants) ):
                            new_meeting.add_participant( np.random.choice(self.occupants) )

        # generate onetime meetings
        selected_room = np.random.choice(list(self.conference_rooms.keys()), number_onetime_meetings)
        selected_day  = np.random.randint(0,365, number_onetime_meetings)
        selected_ts_h = np.random.randint(7, 15, number_onetime_meetings)
        selected_ts_m = np.random.randint(0, 60, number_onetime_meetings)
        for n in range(number_onetime_meetings):
            te_h = np.random.randint(selected_ts_h[n] + 1,
                                     min(selected_ts_h[n] + 5, 20))
            te_m = np.random.randint(0, 60)
            dayobj= datetime.date(2017,1,1) + datetime.timedelta(days=int(selected_day[n]))
            ts   = datetime.time(selected_ts_h[n], selected_ts_m[n])
            te   = datetime.time(te_h, te_m)
            new_meeting = self.add_onetime_meeting(selected_room[n], dayobj, ts, te)
            if not new_meeting is None:
                generated_onetime_meetings += 1
                # add people
                for _ in range(np.random.randint(min_people_joining_meeting, max_peope_joining_meeting)):
                    # if a person cannot attend the meeting, try it at maximum 3 to times
                    if not new_meeting.add_participant( np.random.choice(self.occupants) ):
                        if not new_meeting.add_participant( np.random.choice(self.occupants) ):
                            new_meeting.add_participant( np.random.choice(self.occupants) )

        return generated_weekly_meetings, number_onetime_meetings


    """
    Adds a weekly meeting to the list of meetings.
    It returns the created meeting object.
    If the room is already taken (by a weekly meeting or at some time in the year by a one-time meeting),
    it returns None.
    
    Parameters
    ----------
    room : str
        The name of the room, where the meeting will be
    day_of_week : int
        The day of the week (as int, starting with monday as 1)
    time_start : datetime.time
    time_end : datetime.time
    """
    def add_weekly_meeting(self, room, day_of_week, time_start, time_end):
        if not self._initialized_rooms:
            raise Exception("This building occupancy object is not initialized.")

        if not room in self.conference_rooms.keys():
            raise AttributeError(f"Room {room} is not in the list of known conference rooms.")

        new_meeting = WeeklyMeeting(room, day_of_week, time_start, time_end)
        for meetingDay, meetingStart, meetingEnd in new_meeting.get_all_dates():
            if len(self.get_meetings_during(room, meetingDay, meetingStart, meetingEnd)) > 0:
                return None
        self._weekly_meetings.append(new_meeting)
        return new_meeting


    """
    Adds a one-time meeting to the list of meetings
    It returns the created meeting object.
    If the room is already taken (by a weekly meeting or at some time in the year by a one-time meeting),
    it returns None.
    
    Parameters
    ----------
    room : str
        The name of the room, where the meeting will be
    day : datetime.date
        The date of the meeting as datetime.date-object
    time_start : datetime.time
    time_end : datetime.time
    """
    def add_onetime_meeting(self, room, day, time_start, time_end):
        if not self._initialized_rooms:
            raise Exception("This building occupancy object is not initialized.")

        if not room in self.conference_rooms.keys():
            raise AttributeError(f"Room {room} is not in the list of known conference rooms.")

        if len(self.get_meetings_during(room, day, time_start, time_end)) == 0:
            new_meeting = OneTimeMeeting(room, day, time_start, time_end)
            self._onetime_meetings.append(new_meeting)
            return new_meeting

        return None


    """
    Returns all meetings that take place during a given timespan in a given room.
    If there are no meetings during that timespan, it returns an empty list.
    
    Parameters
    ----------
    room : str
        the name of the room
    day : datetime.date
        the date for the search
    ts_start : datetime.time
        the start time of the timespan
    ts_end : datetime.time
        the end time of the timespan
    """
    def get_meetings_during(self, room, day, ts_start, ts_end):
        meetings = []
        for meeting in self._weekly_meetings:
            if ts_start < meeting.time_end and ts_end > meeting.time_start:
                if meeting.room == room and meeting.day_of_week == day.weekday():
                    meetings.append(meeting)
        for meeting in self._onetime_meetings:
            if ts_start < meeting.time_end and ts_end > meeting.time_start:
                if meeting.room == room and meeting.day == day:
                    meetings.append(meeting)
        return meetings


    """
    Returns the meeting that is at a given date and time in a given room.
    If there is no meeting at that time, it returns none.
    
    Parameters
    ----------
    room : str
        the name of the room
    day : datetime.date
        the date for the search
    time : datetime.time
        the time for the search
    """
    def get_meeting_at(self, room, day, time):
        for meeting in self._weekly_meetings:
            if time >= meeting.time_start and time <= meeting.time_end:
                if meeting.room == room and meeting.day_of_week == day.weekday():
                    return meeting
        for meeting in self._onetime_meetings:
            if time >= meeting.time_start and time <= meeting.time_end:
                if meeting.room == room and meeting.day == day:
                    return meeting
        return None


    """
    Adds a bank holiday to the list of known bank holidays
    """
    def add_bank_holiday(month, day):
        self.bank_holidays.append((month, day))


    """
    Reads a file with all configuration variables that was created using BuildingOccupancy.save_to_file()
    """
    def load_from_file(self, filepath):
        self._initialized_rooms = True
        # TODO
        pass


    """
    Save the complete instance of the class to a file
    """
    def save_to_file(self, filepath, with_occupants_and_meetings=True):
        if not self._initialized_rooms:
            raise Exception("This building occupancy object is not initialized.")
        # TODO: save
        # building name
        # conference rooms
        # office rooms
        # occupants and meetings, if set
        pass


    """
    Draw a sample for a given day
    Returns a dictionary containing the room names and the percentage of maximum people inside
    
    Parameters
    ----------
    dto : datetime
        the date and time for which a sample should be drawn
    use_holiday_settings : Boolean
        apply the defined holiday (and bank holiday) settings
    random_absence_ratio : float
        Ratio for the occupants to be not present, although they should be present
    output_occupants : Boolean
        Output the people in the rooms as secondary output, not only the sum of occupants
    """
    def draw_sample(self,
                    dto,
                    use_holiday_settings = True,
                    random_absence_ratio = 0.01,
                    output_occupants     = False):

        if not self._initialized_rooms:
            raise Exception("This building occupancy object is not initialized.")

        roomdict = {}
        # TODO: random_absence_ratio sollte nicht in jedem Schritt vorgegeben werden,
        #       sondern einmal global fuer das Model. Es macht keinen Sinn, wenn ein Mitarbeiter
        #       zu einem Zeitpunkt da ist, aber eine Minute später weg ist. Dies sollte
        #       lieber tagesweise geschehen
        # TODO: wenn eine Person in einem Meeting ist, wird diese Wohl nicht in einem Raum sein
        #       es wäre wohl besser, wenn man die Frage, in welchem Raum eine Person gerade ist,
        #       als Methode fuer jeden Person definiert
        holiday = False
        if use_holiday_settings:
            if (dto.month, dto.day) in self.bank_holidays:
                holiday = True
        # TODO: holiday settings bearbeiten

        occupant_room_list = {pers: None for pers in self.occupants}
        for office_name in self.office_rooms.keys():
            comfort_temp_list = []
            number_occup = 0
            if dto.weekday() < 5:
                for pers in self.occupants:
                    # TODO: Test, ob die person nicht gerade in einem Meeting ist
                    if pers.default_office == office_name \
                    and pers.presence_start <= dto.time() \
                    and pers.presence_end > dto.time():
                        number_occup += 1
                        comfort_temp_list.append(pers.comfort_temp)
                        occupant_room_list[pers] = office_name
            comfort_temp = 0 if len(comfort_temp_list) == 0 else np.mean(comfort_temp_list)
            roomdict[office_name] = {"relative number occupants": number_occup/self.maximal_numer_of_people_in_office_room,
                                     "absolute number occupants": number_occup,
                                     "mean comfort temp": comfort_temp}

        for conf_room_name, maxp in self.conference_rooms.items():
            comfort_temp_list = []
            number_occup = 0
            if dto.hour < 20 and dto.hour > 7 and dto.weekday() < 5:
                for pers in self.occupants:
                    pMeeting = pers.get_meeting_at(dto.date(), dto.time())
                    if not pMeeting is None and pMeeting.room == conf_room_name:
                        number_occup += 1
                        comfort_temp_list.append(pers.comfort_temp)
                        occupant_room_list[pers] = conf_room_name
            comfort_temp = 0 if len(comfort_temp_list) == 0 else np.mean(comfort_temp_list)
            roomdict[conf_room_name] = {"relative number occupants": number_occup/float(maxp),
                                        "absolute number occupants": number_occup,
                                        "mean comfort temp": comfort_temp}

        if output_occupants:
            return roomdict, occupant_room_list
        return roomdict


    """
    Obtain the number of manual setpoint changes made by the occupants.
    
    Parameters
    ----------
    dto : datetime
        the date and time for which a the setpoint changes should be calculated
    temp_values : dict
        Dictionary containing all current zone temperature values.
    current_setpoints : dict
        Dictionary containing all current setpoint values

    Returns the number of manual setpoint changes [and the new setpoints ?]
    """
    def manual_setpoint_changes(self, dto, temp_values, current_setpoints):
        # TODO: diese Funktion genauer auf die Wuensche der Occupants abstimmen
        changed_setpoints = {}
        no_manual_setp_changes = 0
        changed_magnitude = 0

        for office_name in self.office_rooms.keys():
            if dto.weekday() < 5 and dto.hour >= 7 and dto.hour < 18:
                temp_values
                # if the temperature is not in the range [20,24], change the setpoint
                if temp_values[office_name] < 20:
                    no_manual_setp_changes += 1
                    changed_magnitude += 20 - temp_values[office_name]
                elif temp_values[office_name] > 24:
                    no_manual_setp_changes += 1
                    changed_magnitude += temp_values[office_name] - 24

        for conf_room_name, _ in self.conference_rooms.items():
            if dto.weekday() < 5 and dto.hour >= 7 and dto.hour < 18:
                # if the temperature is not in the range [20,24], change the setpoint
                if temp_values[office_name] < 20:
                    no_manual_setp_changes += 1
                    changed_magnitude += 20 - temp_values[office_name]
                elif temp_values[office_name] > 24:
                    no_manual_setp_changes += 1
                    changed_magnitude += temp_values[office_name] - 24

        return no_manual_setp_changes, changed_magnitude


class BuildingOccupancyAsMatrix:

    def __init__(self, args, building):
        building_occ = BuildingOccupancy()
        office_rooms     = ["SPACE5-1","SPACE3-1"]
        conference_rooms = ["SPACE1-1","SPACE2-1","SPACE4-1"]
        building_occ.set_room_settings(
                { r: building.max_pers_per_room[r] for r in office_rooms },
                { r: building.max_pers_per_room[r] for r in conference_rooms } )
        building_occ.generate_random_occupants(args.number_occupants)
        # mo. 9-12h meeting in Space4-1 with high setpoint (24 deg), 5 people
        #      these group meets on friday, 8-12h, in Space2-1, meeting5
        # wen. 13-16h meeting in Space4-1 with low setpoint (20 deg), 5 people (different than on monday)
        #      these group meets on friday, 14-19h in Space2-1, meeting6
        # don. 8-17h two persons in Space4-1 with low avgerage (22 deg)
        #      on monday these two people are in Space2-1 (meeting 4)
        # tue. 7-11h meeting with completly different persons in SPACE1-1, meeting7
        #      same meeting on don., meeting 8 and fri., meeting 9
        meeting1 = building_occ.add_weekly_meeting("SPACE4-1", 0, datetime.time(hour=9), datetime.time(hour=12))
        meeting2 = building_occ.add_weekly_meeting("SPACE4-1", 2, datetime.time(hour=9), datetime.time(hour=12))
        meeting3 = building_occ.add_weekly_meeting("SPACE4-1", 3, datetime.time(hour=8), datetime.time(hour=17))
        meeting4 = building_occ.add_weekly_meeting("SPACE2-1", 0, datetime.time(hour=8), datetime.time(hour=17))
        meeting5 = building_occ.add_weekly_meeting("SPACE2-1", 4, datetime.time(hour=8), datetime.time(hour=12))
        meeting6 = building_occ.add_weekly_meeting("SPACE2-1", 4, datetime.time(hour=14), datetime.time(hour=19))
        meeting7 = building_occ.add_weekly_meeting("SPACE1-1", 1, datetime.time(hour=7), datetime.time(hour=11))
        meeting8 = building_occ.add_weekly_meeting("SPACE1-1", 3, datetime.time(hour=7), datetime.time(hour=11))
        meeting9 = building_occ.add_weekly_meeting("SPACE1-1", 4, datetime.time(hour=7), datetime.time(hour=10))
        for pers in building_occ.occupants[0:5]:
            meeting1.add_participant(pers)
            meeting5.add_participant(pers)
            pers.comfort_temp = 24.5
        for pers in building_occ.occupants[5:10]:
            meeting2.add_participant(pers)
            meeting6.add_participant(pers)
            pers.comfort_temp = 19.5
        for pers in building_occ.occupants[10:12]:
            meeting3.add_participant(pers)
            meeting4.add_participant(pers)
            pers.comfort_temp = 22
        for pers in building_occ.occupants[13:17]:
            meeting7.add_participant(pers)
            meeting8.add_participant(pers)
            meeting9.add_participant(pers)
        for pers in building_occ.occupants[17:]:
            pers.comfort_temp = 21
        # transform building_occ to a matrix
        year = 2017
        month = args.episode_start_month
        # find first day of the year, that ist the correct day of the week
        start_day = datetime.date(year, month,1)
        while start_day.weekday() > 0:
            start_day += datetime.timedelta(days=1)
        start_day = start_day.day
        time_resolution = datetime.timedelta(minutes = 60 // args.ts_per_hour)
        self.ts_per_hour = args.ts_per_hour
        #
        start_dto = datetime.datetime(year, month, start_day)
        dto       = datetime.datetime(year, month, start_day)
        occ_lst   = []
        dto_lst   = []
        #
        while (dto - start_dto).days < 7:
            occ_obj = building_occ.draw_sample(dto, False, 0, True)
            occ_lst.append( occ_obj[1] )
            dto_lst.append( occ_obj[0] )
            dto += time_resolution
        #
        number_steps     = len(occ_lst)
        occupants        = list(occ_lst[0].keys())
        number_occupants = len(occupants)
        number_rooms     = len(building.room_names)
        self.building_room_names = building.room_names.copy()
        self.schedule_table = np.zeros(shape=(number_steps, number_occupants, number_rooms))
        for idx, occ_obj in enumerate(occ_lst):
            idpers = 0
            for pers, room in occ_obj.items():
                if not room is None:
                    roomid = building.room_names.index(room)
                    self.schedule_table[idx, idpers, roomid] = 1
                idpers += 1
        #
        self.occupants = np.array([p.comfort_temp for p in building_occ.occupants])
        self.max_occupants_per_room = self.schedule_table.sum(axis=1).max(axis=0)

    def dto_to_idx(self, dto):
        """
        Get the position in time (first axis in self.schedule_table) for a given datetime object
        """
        weekday = dto.weekday()
        hour    = dto.hour
        minute  = dto.minute
        return self.ts_per_hour * 24 * weekday + self.ts_per_hour * hour + self.ts_per_hour * minute // 60

    def manual_setpoint_changes(self, dto, temp_values, current_setpoints, discomfort_step_offset = 0.0):
        """
        Obtain the number of manual setpoint changes made by the occupants.

        Parameters
        ----------
        dto : datetime
            the date and time for which a the setpoint changes should be calculated
        temp_values : dict
            Dictionary containing all current zone temperature values.
        current_setpoints : dict
            Dictionary containing all current setpoint values

        Returns the number of manual setpoint changes and some more information
        """
        changed_setpoints = {}
        no_manual_setp_changes = 0
        changed_magnitude = 0
        target_temp_per_room = {}

        idx = self.dto_to_idx(dto)
        selected_schedule_T = self.schedule_table[idx, :, :].T
        # target temperatures are just the matrix-vector-product
        # of (Room,Person)-matrix x (Person-Target-Temp)-vector
        # and than divided by the number of people inside a room
        people_in_rooms  = self.schedule_table[idx].sum(axis=0)
        room_target_temp = np.matmul(self.schedule_table[idx].T, self.occupants)
        room_target_temp = room_target_temp / people_in_rooms.clip(min=1)
        for idroom, room in enumerate(self.building_room_names):
            if people_in_rooms[idroom] <= 0:
                continue
            current_temp = temp_values[room]
            diff = np.abs(current_temp - room_target_temp[idroom])
            if diff > 1.0:
                no_manual_setp_changes += people_in_rooms[idroom]
                changed_magnitude += diff
                changed_magnitude += discomfort_step_offset
            target_temp_per_room[room] = room_target_temp[idroom]

        return no_manual_setp_changes, changed_magnitude, target_temp_per_room

    def draw_sample(self,
                    dto,
                    use_holiday_settings = True,
                    random_absence_ratio = 0.01):
        """
        Draw a sample for a given day
        Returns a dictionary containing the room names and the percentage of maximum people inside

        Parameters
        ----------
        dto : datetime
            the date and time for which a sample should be drawn
        use_holiday_settings : Boolean
            apply the defined holiday (and bank holiday) settings
        random_absence_ratio : float
            Ratio for the occupants to be not present, although they should be present
        """
        roomdict = {}
        holiday = False
        #if use_holiday_settings:
        #    if (dto.month, dto.day) in self.bank_holidays:
        #        holiday = True
        # TODO: holiday settings bearbeiten

        idx = self.dto_to_idx(dto)
        people_in_rooms  = self.schedule_table[idx].sum(axis=0)
        room_target_temp = np.matmul(self.schedule_table[idx].T, self.occupants)
        room_target_temp = room_target_temp / people_in_rooms.clip(min=1)

        for idroom, room in enumerate(self.building_room_names):
            maxp = self.max_occupants_per_room[idroom]
            if type(maxp) == int:
                maxp = float(maxp)
            if maxp <= 0:
                maxp = 1.0
            roomdict[room] = {"relative number occupants": people_in_rooms[idroom] / maxp,
                              "absolute number occupants": people_in_rooms[idroom],
                              "mean comfort temp": room_target_temp[idroom]}

        return roomdict



