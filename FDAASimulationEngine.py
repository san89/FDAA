""" 
Jheronimus Academy of Data Science
Fire Department Amsterdam Amstelland

authors: Joep van den Bogaert
         Santiago Ruiz Zapata

date: June 25, 2018
"""
import numpy as np
import pandas as pd
from pyproj import Proj
from utils import projections, haversine, pre_process_station_name, get_safe_random_value_normal, lonlat_to_xy, pythagoras
import time

class SimulationEngine():
    """ Main class to simulate incidents and responses. """
    
    def __init__(self, demand_location_definition="postcode_digits", verbose=True):

        self.time = 0
        self.demand_location_definition = demand_location_definition
        self.verbose = verbose

        self.building_to_target_dict = {'Bijeenkomstfunctie' : 10,
                                        'Industriefunctie' : 8,
                                        'Woonfunctie' : 8,
                                        'Straat' : 10,
                                        'Overige gebruiksfunctie' : 10,
                                        'Kantoorfunctie' : 10,
                                        'Logiesfunctie' : 8,
                                        'Onderwijsfunctie' : 8,
                                        'Grachtengordel' : 10,
                                        'Overig' : 10,
                                        'Winkelfunctie' : 5,
                                        'Kanalen en rivieren' : 10,
                                        'nan' : 10,
                                        'Trein' : 5,
                                        'Sportfunctie' : 10,
                                        'Regionale weg' : 10,
                                        'Celfunctie' : 5,
                                        'Tram' : 5,
                                        'Sloten en Vaarten' : 10,
                                        'Gezondheidszorgfunctie' : 8,
                                        'Lokale weg' : 5,
                                        'Polders' : 10,
                                        'Haven' : 10,
                                        'Autosnelweg' : 10,
                                        'Meren en plassen' : 10,
                                        'Hoofdweg' : 10,
                                        'unknown': 10}


    def fit_parameters(self, incident_log, deployment_log, from_post_to_coor, station_locations, initial_allocation, time_of_day_filter=None, filter_demo_incidents=True):
        """
        Unique function to fit all the parameters needed to run the simulation. This is required because of the on scene duration time has to be
        calculated on fit_deployment_parameters first and later use on fit_incident_parameters

        """
        self.fit_deployment_parameters(incident_log, deployment_log, station_locations)
        self.fit_incident_parameters(incident_log, deployment_log, from_post_to_coor,time_of_day_filter, filter_demo_incidents)
        self.fit_vehicle_parameters(initial_allocation)

    
    def fit_vehicle_parameters(self, initial_allocation):
        """ Create the vehicle objects based on the defined allocation 
        
        :param initial_allocation: Pandas DataFrame that maps the fire stations with the different 
                            vehicle types
        :return: vehicle object
        """

        def preprocess_inital_allocation(initial_allocation):
            """
                Transform the inital input into an usuable format
            """
            initial_allocation['kazerne'] = initial_allocation['kazerne'].apply(lambda x: pre_process_station_name(x)) 
            kazerne  = initial_allocation['kazerne']
            vehicle_type = initial_allocation.columns[1::]

            unique_id = 0
            initial_allocation_new = pd.DataFrame()

            for index, row in initial_allocation.iterrows():
                for v in vehicle_type:
                    for i in range(row[v]):
                        temp = pd.DataFrame([v, unique_id, row['kazerne'], 0.00])
                        initial_allocation_new = pd.concat([initial_allocation_new, temp], axis = 1)
                        unique_id += 1
                        
            initial_allocation_new = initial_allocation_new.T
            initial_allocation_new.columns =  ['vehicle_type', 'ID', 
                                               'fire_station_assigned', 'available_from_time']
            return initial_allocation_new

        ################################# PARAMETERS ##############################################
        self.vehicles_status = preprocess_inital_allocation(initial_allocation.copy())




    def fit_incident_parameters(self, incident_log, deployment_log, from_post_to_coor, time_of_day_filter=None, filter_demo_incidents=True):
        """ Calculates the parameter values required for simulation from the raw data
            and stores them in the desired formats.

        :param incident_log: Pandas DataFrame with incident data.
        :param deployment_log: Pandas DataFrame with deployment data.
        :param from_post_to_coor: Pandas DataFrame with the information to translate postal 
                                       code digits to latitud and longitud.
        :param time_of_day_filter: tuple of two integers from 0 to 23 representing the start
                                       and end hour of the day to take into account.
        :param verbose: if true, print status updates. 
        """

        def calculate_interarrival_times(incident_data, time_in="minutes"):
            """ Calculate the mean time between consecutive incidents.

            :param incident_data: Pandas DataFrame with the incident data.
            :param time_in: String, one of ["seconds", "minutes"], unit to return the value in.
            :return: mean interarrival time.
            """

            incident_data["dim_incident_start_datumtijd"] = pd.to_datetime(incident_data["dim_incident_start_datumtijd"])
            incident_data.sort_values("dim_incident_start_datumtijd", ascending=True, inplace=True)

            if time_in == "minutes":
                incident_data["interarrival_time"] = incident_data["dim_incident_start_datumtijd"].diff().dt.seconds / 60
            elif time_in == "seconds":
                incident_data["interarrival_time"] = incident_data["dim_incident_start_datumtijd"].diff().dt.seconds
            else:
                ValueError("time_in must be one of ['seconds', 'minutes']")

            return incident_data["interarrival_time"]


        def get_prob_per_demand_location(incident_data, location="postcode_digits"):
            """ Calculate the proportion of incidents that happens in every demand location. 

            :param incident_data: Pandas DataFrame with incident data.
            :param location: How to define the demand locations. Must be one of ["postcode_digits", "grid"].
            :return: Tuple of two numpy arrays with the probabilities and demand location names respectively.
            """

            if location=="postcode_digits":
                incident_data = incident_data[~incident_data["dim_incident_postcode_digits"].isnull()].copy()
                incident_data.sort_values("dim_incident_postcode_digits", ascending=True, inplace=True)
                incident_data = incident_data.groupby("dim_incident_postcode_digits") \
                                        ["dim_incident_id"].count() / len(incident_data)
                return np.array(incident_data), np.array(incident_data.index)
            else:
                print("Only 'postcode_digits' is currently implemented.")
                return False


        def get_type_probs_per_location(incident_data, location="postcode_digits"):
            """ Calculate the distribution of different incident types for each demand location. 
            
            :param incident_data: Pandas DataFrame with the incident data.
            :return: Tuple of (dict, type_names). Dict is a dictionary with demand locations
                     as keys and lists of probabilities over the incident types as elements. 
                     type_names is a list with the incident types in the same order as the lists
                     in the dictionary.
            """

            if location=="postcode_digits":
                incident_data.sort_values("dim_incident_postcode_digits", ascending=True, inplace=True)
                incident_data = incident_data.groupby(["dim_incident_postcode_digits",
                                                       "dim_incident_incident_type"]) \
                                              ["dim_incident_id"].count().reset_index()

                incident_data["type_prob_per_location"] = incident_data.groupby("dim_incident_postcode_digits") \
                                                                        ["dim_incident_id"].apply(lambda x: x/x.sum())

                probs_per_location = pd.pivot_table(incident_data, 
                                                    index="dim_incident_incident_type",
                                                    columns="dim_incident_postcode_digits", 
                                                    values="type_prob_per_location").fillna(0)
                
                types = np.array(probs_per_location.index)
                return {loc : list(probs_per_location[loc]) for loc in probs_per_location.columns}, types


        def get_prio_probabilities_per_type(incident_data):
            """ Create dictionary with the probabilities of having priority 1, 2, and 3 for 
                every incident type. 

            :param incident_data: Pandas DataFrame containing the log of incidents from which
                                  the probabilities should be obtained.
            :return: Dictionary with incident type names as keys and lists of length 3 as elements.
                     The lists have probabilities of prio 1, 2, 3 in position 0, 1, 2 respectively.
            """

            prio_per_type = incident_data.groupby(["dim_incident_incident_type", "dim_prioriteit_prio"])\
                                          ["dim_incident_id"].count().reset_index()

            prio_per_type["prio_probability"] = prio_per_type.groupby(["dim_incident_incident_type"])\
                                                              ["dim_incident_id"].apply(lambda x: x/x.sum())

            prio_probabilities = pd.pivot_table(prio_per_type, columns="dim_incident_incident_type", 
                                                values="prio_probability", index="dim_prioriteit_prio").fillna(0)

            return {col : list(prio_probabilities[col]) for col in prio_probabilities.columns}


        def get_vehicle_requirements_probabilities(incident_data, deployment_data):
            """ Calculate the probabilities of needing a number of vehicles of a specific type 
                for a specified incident type.

            :param incident_data: Pandas DataFrame with the incident data.
            :param deployment_data: Pandas DataFrame with the deployment data.
            :returns: nested dictionary like {"incident type" : {"vehicle type" : {'nr of vehicles' : prob}}}.
            """

            # add incident type to the deployment data
            deployment_data = deployment_data.merge(incident_data[["dim_incident_id", "dim_incident_incident_type"]], 
                                                    left_on="hub_incident_id", right_on="dim_incident_id", how="left")

            # create mock column to count (for interpretability of the code and resulting data)
            deployment_data["count"] = 1

            # count number of vehicles per incident and vehicle type
            deployment_data = deployment_data.groupby(["dim_incident_incident_type", "hub_incident_id",
                                                       "voertuig_groep"])["count"].count().reset_index()

            # retrieve dictionary for each incident type
            types = deployment_data["dim_incident_incident_type"].unique()
            prob_dict = dict()

            # loop over types for convenience, may be optimized
            for ty in types:
                # get information for this incident type
                temp = deployment_data[deployment_data["dim_incident_incident_type"] == ty].copy()
                nr_incidents = temp["hub_incident_id"].nunique()
                vehicles = temp["voertuig_groep"].unique()
                # get the probabilities
                temp = temp.groupby(["voertuig_groep", "count"])["hub_incident_id"].count().unstack().fillna(0)
                temp[0] = nr_incidents - temp.sum(axis=1)
                temp = temp / nr_incidents
                temp = temp.T
                prob_dict[ty] = {v : dict(temp[v][temp[v]!=0]) for v in temp.columns}

            return prob_dict


        def get_building_function_probs(incident_data):
            """ Calculate the probability of an incident occuring in a certain type of building
                given the demand location and incident type.
                
            :param incident_data: Pandas DataFrame with the incident data.
            :return: nested dictionary like {"location" : {"incident type" : {"building" : prob}}}
            """

            incident_data["inc_dim_object_functie"] = incident_data["inc_dim_object_functie"].fillna("unknown")

            building_function_probs = incident_data.groupby(["dim_incident_postcode_digits", "dim_incident_incident_type", "inc_dim_object_functie"])\
                                                   ["dim_incident_id"]\
                                                   .count()\
                                                   .reset_index()

            building_function_probs["building_function_probs"] = \
                                   building_function_probs.groupby(["dim_incident_postcode_digits", "dim_incident_incident_type"])\
                                   ["dim_incident_id"]\
                                   .transform(lambda x: x/x.sum())

            building_dict = \
                building_function_probs.groupby(["dim_incident_postcode_digits", "dim_incident_incident_type"])\
                                       [["inc_dim_object_functie", "building_function_probs"]]\
                                       .apply(lambda x: {x["inc_dim_object_functie"].iloc[i] : x["building_function_probs"].iloc[i] for i in range(len(x))})\
                                       .unstack()\
                                       .T\
                                       .to_dict()
            
            return building_dict

        def map_postcode_to_latlong(incident_data):

            inProj  = Proj("+init=EPSG:28992", preserve_units=True)
            outProj = Proj("+init=EPSG:4326")
            incident_data["dim_incident_postcode_digits"] = incident_data["dim_incident_postcode"].str[0:4]
            incident_data = incident_data.groupby("dim_incident_postcode_digits")[["st_x", "st_y"]].agg("mean").reset_index()
            incident_data["latlong"] = incident_data.apply(lambda x: projections(int(x["st_x"]), int(x["st_y"]), inProj, outProj), axis=1)
            return incident_data[["dim_incident_postcode_digits", "latlong"]].set_index("dim_incident_postcode_digits").to_dict()["latlong"]

        def map_postcode_to_xy(incident_data):
            incident_data = incident_data.groupby("dim_incident_postcode_digits")[["st_x", "st_y"]].agg("mean")
            incident_data["xy_coords"] = list(zip(incident_data["st_x"], incident_data["st_y"]))
            incident_data.to_dict()["xy_coords"]
            return incident_data.to_dict()["xy_coords"]


        ################################# PARAMETERS ##############################################
        # calculate interarrival times and add as column to the data
        incident_log["interarrival_time"] = calculate_interarrival_times(incident_log, time_in="minutes")

        # filter on specific period of the day if specified
        if time_of_day_filter is not None:
                assert(len(time_of_day_filter)==2), "time_of_day_filter must be array-like of length 2, or None."
                a, b = time_of_day_filter
                incident_log = incident_log[(incident_log["dim_tijd_uur"]>=a)&(incident_log["dim_tijd_uur"]<b)].copy()
                if self.verbose:
                    print("Only considering incidents between {} and {} O'clock.".format(a,b))

        # add demand location identifier if locations are by postcode
        if self.demand_location_definition == "postcode_digits":
            incident_log["dim_incident_postcode_digits"] = incident_log["dim_incident_postcode"].str[0:4]

        # filter out demo incidents, i.e., incidents without deployments
        incidents_before = len(incident_log)
        incident_log = incident_log[np.isin(incident_log["dim_incident_id"], deployment_log["hub_incident_id"])].copy()

        if self.verbose:
            print("{} incident(s) removed because there were no corresponding deployments.".format(incidents_before - len(incident_log)))

        # filter out postal code with no latitud and longitud
        incidents_before = len(incident_log)
        incident_log = incident_log[incident_log["dim_incident_postcode_digits"].isin(from_post_to_coor['Postal Code'])]

        if self.verbose:
            print("{} incident(s) removed because its postal code does not exist in the location dictionary.".format(incidents_before - len(incident_log)))

        incidents_before = len(incident_log)
        incident_log = incident_log[incident_log['dim_incident_incident_type'].isin(list(self.parameters_on_scene_time.keys()))]

        # filter out incident that does not have time parameters
        if self.verbose:
            print("{} incident(s) removed because its postal code does not exist in the on scene dictionary.".format(incidents_before - len(incident_log)))


        # set the mean interarrival time
        self.mean_interarrival_time = incident_log["interarrival_time"][1:].mean()

        # get the probabilities that an incident occurs in specific demand location
        self.location_probabilities, self.demand_location_ids = \
            get_prob_per_demand_location(incident_log, location=self.demand_location_definition)

        # set dictionary to translate postal code to latitud and longitud
        #self.location_dict = map_postcode_to_latlong(incident_log)
        self.location_dict = map_postcode_to_xy(incident_log)

        # get demand location information
        self.type_probs_per_location, self.type_probs_names = \
            get_type_probs_per_location(incident_log, location=self.demand_location_definition)

        # get priority probabilities per incident type
        self.prio_prob_dict = get_prio_probabilities_per_type(incident_log)

        # get probabilities of having certain vehicle requirements from an incident
        self.vehicle_prob_dict = get_vehicle_requirements_probabilities(incident_log, deployment_log)

        # get probabilities of an incident occuring in a certain building type/function,
        # given the location and type of incident
        self.building_prob_dict = get_building_function_probs(incident_log)

        if self.verbose:
            print("Incident parameters are obtained from the data.")
        """ End of fit_incident_parameters """

    def fit_deployment_parameters(self, incident_log, deployment_log, station_locations):
        """
        Calculates the parameter values for the deployed resourse required for simulation from the raw data and 
        stores them in the desired formats.
        
        :param incident_log: Pandas DataFrame with incident data.
        :param deployment_log: Pandas DataFrame with deployment data.
        :param station_locations: Pandas DAtaFrame with the fire station information.
        """

        def pre_process_data(incident_log, deployment_log, station_locations):
            """
            In this function we do all pre-processing necessary to calculate the different components of the response time.
            
            Note: the travel time depends on the speed and the incident location. Because the incident location is a 
            random variable, we only consider the speed parameters.
            
            Params
            -------------------------------------------
            df_in: data frame with the incident data
            df_dep: data frame with the deployment data
            station_locations: data frame with the fire station data

            
            Returns:
            M: dataframe with the merged datasets incident, deplyment and station_location
            """
            
            incindets, deployments = incident_log.copy(), deployment_log.copy()

            inProj  = Proj("+init=EPSG:28992", preserve_units=True)
            outProj = Proj("+init=EPSG:4326")

            # Some filters to have an somaller merged dataset
            keep_in_var = ['dim_incident_id','st_x', 'st_y', 'dim_incident_incident_type', 'inc_dim_object_naam', 
                       'dim_incident_start_datumtijd', 'dim_incident_eind_datumtijd', 'dim_prioriteit_prio']
            
            
            keep_dep_var = ['hub_incident_id', 'inzet_uitgerukt_datumtijd', 'inzet_gealarmeerd_datumtijd',
                        'inzet_terplaatse_datumtijd', 'voertuig_groep', 'kazerne_groep', 'inzet_kazerne_naam', 
                        'inzet_eind_inzet_datumtijd']
            

            time_stamps_var = ['dim_incident_start_datumtijd', 'dim_incident_eind_datumtijd', 'inzet_gealarmeerd_datumtijd',
                          'inzet_uitgerukt_datumtijd', 'inzet_terplaatse_datumtijd', 'inzet_eind_inzet_datumtijd']

            incindets = incindets[keep_in_var]
            deployments = deployments[keep_dep_var]
            
            M = deployments.merge(incindets, left_on='hub_incident_id', right_on='dim_incident_id', how = 'inner')
            M['inzet_kazerne_naam'] = M['inzet_kazerne_naam'].apply(lambda x: pre_process_station_name(x))

            # add x, y coordinates to station location data
            station_locations["x"], station_locations["y"] = \
                [list(l) for l in zip(*list(station_locations.apply(lambda x: lonlat_to_xy(x["lon"], x["lat"]), axis=1)))]

            # print(set(station_locations['kazerne'].unique()) - (set(station_locations['kazerne'].unique()) & set(M['inzet_kazerne_naam'].unique())))
            M = station_locations.merge(M, left_on='kazerne', right_on='inzet_kazerne_naam', how = 'inner')
            #M['lat_in'], M['lon_in'] = np.vectorize(projections)(M['st_x'], M['st_y'], inProj, outProj)
            #M['haversine_distance (Km)'] = np.vectorize(haversine)(M['lon'], M['lat'], M['lon_in'], M['lat_in'])

            # calculate pythagoras distance
            M['pythagoras_distance'] = M.apply(lambda x: pythagoras(x["x"], x["y"], x["st_x"], x["st_y"]), axis=1) / 1000 # km
            
            for date in time_stamps_var:
                M[date] = pd.to_datetime(M[date])

            # M['dispatch (seconds)'] = (M['inzet_gealarmeerd_datumtijd'] - M['dim_incident_start_datumtijd']).astype('timedelta64[s]')
            M['dispatch (min)'] = 1.5
            M['turn out time (min)'] = (M['inzet_uitgerukt_datumtijd'] - M['inzet_gealarmeerd_datumtijd']).astype('timedelta64[m]')
            M['travel time (min)'] = (M['inzet_terplaatse_datumtijd'] - M['inzet_uitgerukt_datumtijd']).astype('timedelta64[m]')
            M['response time (min)'] = M['turn out time (min)'] + M['travel time (min)'] + M['dispatch (min)'] 
            #M['Average Speed (Km/h)'] = M['haversine_distance (Km)']/(M['travel time (min)']/(60))
            M['Average Speed (Km/h)'] = M['pythagoras_distance']/(M['travel time (min)']/(60))
            total_incident_duration = (M['inzet_eind_inzet_datumtijd'] - M['dim_incident_start_datumtijd']).astype('timedelta64[m]')

            M_before = len(M)
            #We assume an average speed of 40 Kn/h in the way back to the station
            #M['on scene duration (min)'] = total_incident_duration - M['response time (min)'] - (M['haversine_distance (Km)'] * (1/30))
            M['on scene duration (min)'] = total_incident_duration - M['response time (min)'] - (M['pythagoras_distance'] * (1/30))

            #Some filters to remove outliers. This process can be done in a best way for sure
            M = M[(M['Average Speed (Km/h)']>0) & (M['Average Speed (Km/h)']<150) & (M['on scene duration (min)']>0) & 
                  (M['turn out time (min)']>0) & (M['dispatch (min)']>0) & (M['on scene duration (min)'] < 250) & (M['dim_prioriteit_prio']<3)].replace([np.inf, -np.inf], np.nan).dropna()

            if self.verbose:
                print("{} deployments out of {} have been deleted because they are labeled as 'outliers'".format(M_before - len(M), M_before))
            
            return M

        def create_time_parameters(merged_log):
            """
            Calculate different aggregate levels of the time dependent parameters

            Params:
            ------------------------------------------
            merged_log: see funtion pre_process_data

            Return
            -----------------------------------------
            Global dictionary response_time_parameters that have three levels of aggregation. Level one is the less aggregated level  

            """

            M = merged_log.copy() 

            # We only include priority one and two incidents
            M = M[M['dim_prioriteit_prio'] < 3]
            response_time_parameters = {}
            response_time_parameters['level_1'] = M.groupby(['kazerne', 'dim_prioriteit_prio', 'voertuig_groep'], as_index=False).agg({
                                                                          'dispatch (min)':['mean','std', 'count'],
                                                                          'turn out time (min)':['mean','std'],
                                                                          'response time (min)':['mean','std'],
                                                                          'Average Speed (Km/h)':['mean','std']}).dropna()

            response_time_parameters['level_2'] = M.groupby([ 'dim_prioriteit_prio', 'voertuig_groep'], as_index=False).agg({
                                                                          'dispatch (min)':['mean','std', 'count'],
                                                                          'turn out time (min)':['mean','std'],
                                                                          'response time (min)':['mean','std'],
                                                                          'Average Speed (Km/h)':['mean','std']}).dropna()

            response_time_parameters['level_3'] = M.groupby(['voertuig_groep'], as_index=False).agg({
                                                                          'dispatch (min)':['mean','std', 'count'],
                                                                          'turn out time (min)':['mean','std'],
                                                                          'response time (min)':['mean','std'],
                                                                          'Average Speed (Km/h)':['mean','std']}).dropna()

            return response_time_parameters

        def get_parameters_on_scene_time(merged_log):
            """ This function calculated the mean. """
            parameters_on_scene_time = self.merged_log.groupby(['dim_incident_incident_type'], as_index=False).agg({
                'on scene duration (min)':['mean', 'std']}).dropna()

            return dict(zip(parameters_on_scene_time['dim_incident_incident_type'], zip(parameters_on_scene_time['on scene duration (min)']['mean'], parameters_on_scene_time['on scene duration (min)']['std'])))

        ################################## PARAMETERS ###########################################################
        self.merged_log = pre_process_data(incident_log, deployment_log, station_locations)
        self.response_time_parameters = create_time_parameters(self.merged_log.copy())
        station_locations["x"], station_locations["y"] = \
                [list(l) for l in zip(*list(station_locations.apply(lambda x: lonlat_to_xy(x["lon"], x["lat"]), axis=1)))]
        self.station_locations = station_locations
        self.parameters_on_scene_time = get_parameters_on_scene_time(self.merged_log.copy())


    def initialize_demand_locations(self):
        self.demand_locations = \
            {self.demand_location_ids[i] : DemandLocation(self.demand_location_ids[i],
                                                          self.type_probs_per_location[self.demand_location_ids[i]],
                                                          self.type_probs_names,
                                                          self.building_prob_dict[self.demand_location_ids[i]]) \
                                           for i in range(len(self.demand_location_ids))}


    def return_finished_vehicles(self):
        """ Return vehicles that are finished solving an incident. """
        pass


    def generate_incident(self):


        def sample_location(locations, probabilities):
            """ Randomly generate the location of the incident.

            :param locations: array of location names.
            :param probabilities: array of probabilities for each location
                                  (same length as locations).
            :return: string value with location name.
            """
            return locations[np.digitize(np.random.sample(), np.cumsum(probabilities))]


        def sample_deployment_requirements(type_of_incident):
            """ Draw random sample of the priority, the required vehicles, and response time target
                given the incident type. For some incident types, some may be are fixed, in which 
                case they are still returned.

            :param type_of_incident: string representing the incident type.
            :return: Tuple of (priority, dictionary of {"vehicle type" : number}, response time target)
            """

            def sample(d):
                return np.random.choice(a=list(d.keys()), p=list(d.values()))

            # sample priority
            prio = np.random.choice(a=[1,2,3], p=self.prio_prob_dict[type_of_incident])

            # sample vehicle requirements
            # TODO: make safe -> draw again if no vehicles required
            v = self.vehicle_prob_dict[type_of_incident]
            vehicle_requirements =  {key : sample(v[key]) for key in v.keys()} # finish this line
            vehicle_requirements = {k : v for k, v in vehicle_requirements.items() if v > 0}
            
            # TODO: add incident duration
            return prio, vehicle_requirements


        def sample_incident_duration(type_of_incident):
            """ Randomly generate the duration of the incident.

            :param type_of_incident: string representing the incident type.
            :return: scalar value, the duration of the incident in minutes. 
            """
            # TODO
            return 20

        def get_response_time_target(priority, building_function):
            if priority == 1:
                return self.building_to_target_dict[building_function]
            else:
                return 30

        ################################# PARAMETERS ##############################################
        incident_location_id = sample_location(self.demand_location_ids, self.location_probabilities)
        incident_type = self.demand_locations[incident_location_id].sample_incident_type()
        building_function = self.demand_locations[incident_location_id].sample_building_function(incident_type)
        priority, vehicles = sample_deployment_requirements(incident_type)
        response_time_target = get_response_time_target(priority, building_function)
        on_scene_duration = get_safe_random_value_normal(self.parameters_on_scene_time[incident_type][0], self.parameters_on_scene_time[incident_type][1])


        return Incident(self.time, incident_type, priority, vehicles, incident_location_id, self.location_dict[incident_location_id], 
                        building_function, response_time_target, on_scene_duration)


    def get_vehicle_simulated_values(self, incident_priority, incident_location, vehicle_type, on_scene_time):

        def check_statistical_significance(df, n=20):
            """
            check if the filtered subset is statistical significant
            
            Parameters
            ----------------------------------------
            df: dataFrame with the parameter of the filter made at the funtion get_parameters
            n: minimun number of observations to consider the parameters as significant
            
            Return
            -----------------------------------------
            True if the parameters are statistically significant, False otherwise
            """
            
            names = df.columns.get_level_values(level=0).unique()[0:3]
            try:
                statistical_significance = df['dispatch (min)']['count'].values < n
                statistical_significance = statistical_significance[0]

            except IndexError:
                if self.verbose:
                    print('Empty dataFrame')
                statistical_significance = True

            if statistical_significance:
                if self.verbose:
                    print('No enougth observations, the minumin is n={}'.format(n))
                return False

            return True

        def get_parameters(station, vehicle, incident_priority):
            """
            get the mean and standar deviation of the time distribution of the required sources
            
            Parameters
            ----------------------------------------------
            See the response_time_simulation parameters
            
            Return
            --------------------------------------------
            lower aggregated parameters
            """
            
            # Level one of aggregation
            parameters = self.response_time_parameters['level_1'][ 
                                                 (self.response_time_parameters['level_1']['kazerne'] == station) & 
                                                 (self.response_time_parameters['level_1']['dim_prioriteit_prio'] == incident_priority) &
                                                 (self.response_time_parameters['level_1']['voertuig_groep'] == vehicle)
                                                 ]

            if check_statistical_significance(parameters):
                if self.verbose:
                    print('getting the parameters from level_1 of the vehicle {}'.format(vehicle))
                return parameters
            
            # Level two of aggregation
            parameters = self.response_time_parameters['level_2'][  
                                                 (self.response_time_parameters['level_2']['dim_prioriteit_prio'] == incident_priority) &
                                                 (self.response_time_parameters['level_2']['voertuig_groep'] == vehicle)
                                                 ]
                
            if check_statistical_significance(parameters):
                if self.verbose:
                    print('getting the parameters from level_2 of the vehicle {}'.format(vehicle))
                return parameters
            
            # Level one of aggregation
            parameters = self.response_time_parameters['level_3'][  
                                                 (self.response_time_parameters['level_3']['voertuig_groep'] == vehicle)
                                                 ]

            if self.verbose:
                print('getting the parameters from level_3 of the vehicle {}'.format(vehicle))

            return parameters


        def return_trip_duration(distance_km, station, vehicle):
            """
            Calculate the expected trip back duration in seconds. We assume priority two distribution per vehicle,
            We get higher distribution otherwise.
            
            Parameters
            ----------------------------------------
            See response_time_simulation funtion
            
            Return
            ---------------------------------------
            travel time back to the fire station in minutes
            """
            parameters = get_parameters(station, vehicle, 2)
            speed_km = get_safe_random_value_normal(parameters['Average Speed (Km/h)']['mean'], parameters['Average Speed (Km/h)']['std'])
            
            return  (distance_km / speed_km) * 60 


        def get_simulated_values(station, vehicle, incident_priority, incident_location, on_scene_time):
            """
            Function to calculate the simulated value
            
            Paramters:
            ------------------------------------------
            see response_time_simulation
            
            Return
            ----------------------------------------
            x: random value x that assumes normal ditsribution
            """
            x = {}
            parameters = get_parameters(station, vehicle, incident_priority)
            if parameters is None:
                return parameters
            
            reponse_time = 0.
            for var in list(self.response_time_parameters['level_1'].columns.get_level_values(level=0).unique()[4::]):
                temp_par = parameters[var]
                x[var] = get_safe_random_value_normal(temp_par['mean'], temp_par['std'])

            #self.station_locations[self.station_locations['kazerne'] == station][['lon', 'lat']]
            #x["distance_km"] = haversine(incident_location[1], incident_location[0], 
            #                            self.station_locations[self.station_locations['kazerne'] == station]['lon'], 
            #                            self.station_locations[self.station_locations['kazerne'] == station]['lat'])
            self.station_locations[self.station_locations['kazerne'] == station][['lon', 'lat']]
            station_info = self.station_locations[self.station_locations['kazerne'] == station]
            x["pythagoras_distance"] = float(pythagoras(station_info["x"], station_info["y"], incident_location[0], incident_location[1]) / 1000)

            x['dispatch (min)'] = 1.5
            #x['travel time (min)'] = (x["distance_km"] / x['Average Speed (Km/h)']) * 60
            x['travel time (min)'] = (x["pythagoras_distance"] / x['Average Speed (Km/h)']) * 60
            x['response time (min)'] = x['dispatch (min)'] + x['turn out time (min)'] + x['travel time (min)'] 
            x['on scene and trip back (min)'] = on_scene_time + return_trip_duration(x["pythagoras_distance"], station, vehicle)
            x['total incident duration (min)'] = x['response time (min)'] + x['on scene and trip back (min)']
            return x
            
            
        def response_time_simulation(incident_priority, incident_location, vehicle_ids, on_scene_time):
            """
            Calculate the simulated response time per vehicle. We assume exponential distribution of the time
            
            Parameters
            --------------------------------------------
            station: The station that is going to be used to deploy the vehicles
            incident_priority: (str) is the priority of the incident (dim_prioriteit_prio)
            incident_location: [latitud, longitud] latitud and longitud of the incident
            vehicle_type: [] list with the vehicles that will to be deployed 
            on_scene_time: (float) expected incident duration time in minutes
            
            Return
            --------------------------------------------
            response_time = (dict) every time component per vehicle
            """
            response_time = {}
            for id_ in vehicle_ids:
                station = self.vehicles_status[self.vehicles_status['ID']==id_]['fire_station_assigned'][0]
                vehicle = self.vehicles_status[self.vehicles_status['ID']==id_]['vehicle_type'][0]
                response_time[id_] = get_simulated_values(station, vehicle, incident_priority, incident_location, on_scene_time)

            return response_time

        return response_time_simulation(incident_priority, incident_location, vehicle_type, on_scene_time)


    def agent(self, incident, station_locations, vehicles_status):
        def shortest_trip_rule(incident, station_locations, vehicles_status):
            """
                This heuristic takes the vehicles that are expected to arrive first based on their current status and 
            """
            # we assume a constant speed of 40 Km/h
            #station_locations['Expected arrival time (min)'] = np.vectorize(haversine)(station_locations['lon'], station_locations['lat'], incident.location_coord[1], incident.location_coord[0]) * (1/40) * (60)
            station_locations['Expected arrival time (min)'] = station_locations.apply(lambda x: pythagoras(x["x"],
                                                                                                            x["y"],
                                                                                                            incident.location_coord[0],
                                                                                                            incident.location_coord[1]) / 1000 / 40 * 60,
                                                                                       axis=1)

            vehicles_status = vehicles_status[vehicles_status['vehicle_type'].isin(list(incident.required_vehicles.keys()))]
            # vehicles_status['Expected arrival time (min)'] = np.inf
            # This operation must be optimiced!

            vehicles_status = vehicles_status.merge(station_locations, left_on='fire_station_assigned', right_on='kazerne', how = 'inner')

            # for index, row in station_locations.iterrows():
            #     vehicles_status.loc[vehicles_status['fire_station_assigned'] == row['kazerne'], 'Expected arrival time (min)'] = row['Expected arrival time (min)'] 
            vehicles_status['Expected arrival time (min)']  += vehicles_status['available_from_time']
            # Find the fastest cars
            vehicles_status = vehicles_status.sort_values(by=['Expected arrival time (min)']).dropna()
            # Assign resourses
            vehicles_ids = []
            for key, value in incident.required_vehicles.items():
                temp_v = vehicles_status[vehicles_status['vehicle_type'] == key]
                if (value > len(temp_v)) & self.verbose:
                    print('Warning!! more vehicles type {} are required than avaibale'.format(key))

                n = min(value, len(temp_v))
                vehicles_ids.extend(list(temp_v['ID'][0:n]))

            return vehicles_ids
                    

        return shortest_trip_rule(incident, station_locations.copy(), vehicles_status.copy())


    def update_vehicle_status(self, deploymet_time):
        """
           Update the vehicles_status DataFrame
        """

        for key, value in deploymet_time.items():
            self.vehicles_status.loc[self.vehicles_status['ID'] == key, 'available_from_time'] = value['total incident duration (min)'] + max(self.time, float(self.vehicles_status[self.vehicles_status['ID'] == 0]['available_from_time']))

    def reset_results(self):

        cols = ["timestamp", "incident_type", "priority", "location", "vehicle_id", "vehicle_type", "station",
                "target", "dispatch_time", "turnout_time", "travel_time", "response_time", "on_time",
                "on_scene_and_return_time", "distance", "average_speed"]

        self.deployment_results = pd.DataFrame(columns=cols)


    def evaluate_deployments(self, incident, deployments):

        for vehicle in deployments.keys():
            # select nested dictionary
            d = deployments[vehicle]
            # get vehicle details
            vtype = str(self.vehicles_status.set_index("ID").loc[vehicle, "vehicle_type"])
            station = str(self.vehicles_status.set_index("ID").loc[vehicle, "fire_station_assigned"])
            # log
            self.deployment_results = \
                self.deployment_results.append(pd.DataFrame([[incident.start_time,
                                                              incident.type,
                                                              incident.priority,
                                                              incident.location,
                                                              vehicle,
                                                              vtype,
                                                              station,
                                                              incident.response_time_target,
                                                              d["dispatch (min)"],
                                                              d["turn out time (min)"],
                                                              d["travel time (min)"],
                                                              d["response time (min)"],
                                                              d["response time (min)"] < incident.response_time_target,
                                                              d["on scene and trip back (min)"],
                                                              d["pythagoras_distance"],
                                                              d["Average Speed (Km/h)"]]],
                                                              columns=self.deployment_results.columns),
                                                ignore_index=True)


    def get_deployment_results(self):
        return self.deployment_results


    def get_on_time_rate(self, ts_only=True):
        if not ts_only:
            return self.deployment_results["on_time"].mean()
        else:
            return self.deployment_results[self.deployment_results["vehicle_type"]=="TS"]["on_time"].mean()

    def step(self):
        """ Take one simulation step (one incident). """
        self.time = float(self.time + np.random.exponential(self.mean_interarrival_time, 1))
        #self.return_finished_vehicles()
        incident = self.generate_incident()
        
        if self.verbose:
            print("Time: {}. Incident: {} with priority {} at postcode {}.".format(
                  self.time, incident.type, incident.priority, incident.location))

        # define what vehicles to use
        vehicles_to_deploy = self.agent(incident, self.station_locations, self.vehicles_status)

        # calculate the time required based on the vehicles selected
        deployment_times = self.get_vehicle_simulated_values(incident.priority, list(incident.location_coord),
                                                             vehicles_to_deploy, incident.on_scene_duration)
        self.update_vehicle_status(deployment_times)

        # evaluate if the vehicles (probably only TS) reach the incident on time
        self.evaluate_deployments(incident, deployment_times)

    def simulate(self, simulation_time, nr_incidents=None, by_incidents=False):
        """ Run the simulation. """ 
        
        self.reset_time()
        self.reset_results()

        while self.time < simulation_time:
            self.step()


    def reset_time(self):
        self.time = 0


class Deployments():
    """ A response to an incident of a single vehicle """
    def __init__(self, incident_time, vehicle_id, priority, vehicle_type, response_time_target,
                 dispatch_time, turnout_time, travel_time, response_time, on_scene_time):
        self.incident_time = incident_time
        self.vehicle_id = vehicle_id
        self.priority = priority
        self.vehicle_type = vehicle_type
        self.response_time_target = response_time_target
        self.dispatch_time = dispatch_time
        self.turnout_time = turnout_time
        self.travel_time = travel_time
        self.response_time = response_time
        self.on_time = (self.response_time < self.response_time_target)



class Incident():
    """ An incident that requires a response from the fire department: 
        can be a fire or something else.
    """

    def __init__(self, start_time, incident_type, priority, required_vehicles, 
                 location, location_coord, building_function, response_time_target,
                 on_scene_duration):

        self.start_time = start_time
        self.type = incident_type
        self.priority = priority
        self.required_vehicles = required_vehicles
        self.location = location
        self.location_coord = location_coord
        self.building_function = building_function
        self.response_time_target = response_time_target
        self.on_scene_duration = on_scene_duration


class DemandLocation():
    """ An area in which incidents occur. """

    def __init__(self, location_id, incident_type_probs, incident_type_names, building_function_dict):
        self.id = location_id
        self.incident_type_probs = incident_type_probs
        self.incident_type_names = incident_type_names
        self.building_function_dict = building_function_dict

    def sample_incident_type(self):
        return np.random.choice(a=self.incident_type_names, p=self.incident_type_probs)

    def sample_building_function(self, incident_type):
        return np.random.choice(a=list(self.building_function_dict[incident_type].keys()),
                                p=list(self.building_function_dict[incident_type].values()))


class Agent():
    """ Autonomous agent that makes decisions on deployments (rule-based or otherwise). 
        Extendable to make decisions on the proactive relocating of vehicles.
    """

    def __init__(self):
        pass

    def deploy(self, incident, state):
        """ Determine what vehicles to deploy for the given incident,
            given the current state. 
        """
        pass


class FireStation():
    """ Represents a fire station. """
    
    def __init__(vehicles, name, location):
        self.name
        self.vehicles = vehicles
        self.location = location
        

class Vehicle():
    """ A vehicle of the fire department. """
    
    def __init__(vehicle_type, ID, fire_station, available_from):
        self.type = vehicle_type
        self.ID = ID
        self.fire_station_assigned = fire_station_assigned
        self.available_from = available_from


    
    
    