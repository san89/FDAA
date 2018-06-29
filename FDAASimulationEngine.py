""" 
Jheronimus Academy of Data Science
Fire Department Amsterdam Amstelland

authors: Joep van den Bogaert
         Santiago Ruiz Zapata

date: June 25, 2018
"""
import numpy as np
import pandas as pd

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


    def fit_incident_parameters(self, incident_log, deployment_log, time_of_day_filter=None, filter_demo_incidents=True):
        """ Calculates the parameter values required for simulation from the raw data
            and stores them in the desired formats.

        :param incident_log: Pandas DataFrame with incident data.
        :param deployment_log: Pdandas DataFrame with deployment data.
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

        # set the mean interarrival time
        self.mean_interarrival_time = incident_log["interarrival_time"][1:].mean()

        # get the probabilities that an incident occurs in specific demand location
        self.location_probabilities, self.demand_location_ids = \
            get_prob_per_demand_location(incident_log, location=self.demand_location_definition)

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

    def initialize_demand_locations(self):
        self.demand_locations = \
            {self.demand_location_ids[i] : DemandLocation(self.demand_location_ids[i],
                                                          self.type_probs_per_location[self.demand_location_ids[i]],
                                                          self.type_probs_names,
                                                          self.building_prob_dict[self.demand_location_ids[i]]) \
                                           for i in range(len(self.demand_location_ids))}

    def set_agent(self, agent):
        self.agent = agent


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

        incident_location_id = sample_location(self.demand_location_ids, self.location_probabilities)
        incident_type = self.demand_locations[incident_location_id].sample_incident_type()
        building_function = self.demand_locations[incident_location_id].sample_building_function(incident_type)
        priority, vehicles = sample_deployment_requirements(incident_type)
        response_time_target = get_response_time_target(priority, building_function)

        return Incident(self.time, incident_type, priority, vehicles, incident_location_id, building_function, response_time_target)


    def step(self):
        """ Take one simulation step (one incident). """
        self.time = float(self.time + np.random.exponential(self.mean_interarrival_time, 1))

        self.return_finished_vehicles()
        incident = self.generate_incident()

        if self.verbose:
            print("Time: {}. Incident: {} with priority {} at postcode {}.".format(
                  self.time, incident.type, incident.priority, incident.location))

        state = ...
        self.agent.deploy(incident, state)
        #self.agent.relocate(state)


    def simulate(self, simulation_time, nr_incidents=None, by_incidents=False):
        """ Run the simulation. """ 
        
        self.reset_time()

        while self.time < simulation_time:
            self.step()


    def reset_time(self):
        self.time = 0


class Incident():
    """ An incident that requires a response from the fire department: 
        can be a fire or something else.
    """
    
    def __init__(self, start_time, incident_type, priority, required_vehicles, location, building_function, response_time_target):
        self.start_time = start_time
        self.type = incident_type
        self.priority = priority
        self.required_vehicles = required_vehicles
        self.location = location


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
    
    def __init__(vehicle_type):
        self.type = vehicle_type


    
    
    
    