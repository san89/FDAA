""" 
Jheronimus Academy of Data Science
Fire Department Amsterdam Amstelland

authors: Joep van den Bogaert
         Santiago Ruiz Zapata

date: June 25, 2018
"""

class SimulationEngine():
    """ Main class to simulate incidents and responses. """
    
    def __init__(self, mean_interarrival_time, location_probs, demand_locations, decision_agent, 
                 demand_location_definition="postcode_digits"):

        self.time = 0
        self.L = mean_interarrival_time
        self.location_probs = location_probs
        self.demand_locations = demand_locations
        self.agent = decision_agent
        self.demand_location_definition = demand_location_definition

    
    def fit_incident_parameters(self, incident_log, deployment_log, verbose=True):
        """ Calculates the parameter values required for simulation from the raw data
            and stores them in the desired formats. 
        :param incident_log: Pandas DataFrame with incident data.
        :param deployment_log: Pandas DataFrame with deployment data. 
        """

        def get_prob_per_demand_location(incident_data, location="postcode_digits"):
            """ Calculate the proportion of incidents that happens in every demand location. 

            :param incident_data: Pandas DataFrame with incident data.
            :param location: How to define the demand locations. Must be one of ["postcode_digits", "grid"].
            :return: Tuple of two numpy arrays with the probabilities and demand location names respectively.
            """

            if location=="postcode_digits":
                incident_data = incident_data[~incident_data["dim_incident_postcode"].isnull()].copy()
                incident_data["dim_incident_postcode_digits"] = incident_data["dim_incident_postcode"].str[0:4]
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

        def get_response_time_targets(incident_data):
            """ Determine the probability of having a certain respons time target.. """
            pass

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

        if verbose:
            print("Incident parameters are obtained from the data.")
        """ End of fit_incident_parameters """

    def initialize_demand_locations(self):
        self.demand_locations = {self.demand_location_ids[i] : 
                                    DemandLocation(self.demand_location_ids[i],
                                                   self.type_probs_per_location[self.demand_location_ids[i]],
                                                   self.type_probs_names) \
                                    for i in range(len(self.demand_location_ids))}

    def return_finished_vehicles(self):
        """ Return vehicles that are finished solving an incident. """
        pass


    def generate_incident(self):

        def sample_deployment_requirements(type_of_incident):
            """ Draw random sample of the priority, the required vehicles, and response time target
                given the incident type. For some incident types, some may be are fixed, in which 
                case they are still returned.

            :param type_of_incident: string representing the incident type.
            :return: Tuple of (priority, dictionary of {"vehicle type" : number}, response time target)
            """
            prio = np.random.choice([1,2,3], self.prio_prob_dict[type_of_incident])
            vehicle_requirements = self.vehicle_prob_dict[type_of_incident] # finish this line
            if prio == 1:
                response_time_target = 10 # TODO: implement response time targets
            elif prio == 2:
                response_time_target = 30
            else: 
                response_time_target = 60
            return prio, vehicle_requirements, response_time_target

        incident_location_id = np.random.choice(self.demand_location_ids, self.location_probabilities)
        incident_type = self.demand_locations[incident_location_id].sample_incident_type()
        priority, vehicles, response_time_target = self.sample_deployment_requirements(incident_type)

        return Incident(self.time, incident_type, priority, required_vehicles, incident_location_id)

    
    def get_state(self):
        pass


    def step(self):
        
        self.time = self.time + np.random.exponential(self.L, 1)
        self.return_finished_vehicles()
        incident = self.generate_incident()
        state = self.get_state()
        self.agent.deploy(incident, state)


class DemandLocation():
    """ An area in which incidents occur. """

    def __init__(self, location_id, incident_type_probs, incident_type_names):
        self.id = location_id
        self.incident_type_probs = incident_type_probs
        self.incident_type_names = incident_type_names

    def sample_incident_type(self):
        return np.random.choice(self.incident_type_names, p=self.incident_type_probs)


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
    
    def __init__(vehicles, location, incident_type_probs):
        self.vehicles = vehicles
        self.location = location
        self.incident_type_probs = incident_type_probs
        

class Incident():
    """ An incident that requires a response from the fire department: 
        can be a fire or something else.
    """
    
    def __init__(start_time, incident_type, priority, required_vehicles, location):
        pass
    

class Vehicle():
    """ A vehicle of the fire department. """
    
    def __init__(vehicle_type):
        self.type = vehicle_type


    
    
    
    