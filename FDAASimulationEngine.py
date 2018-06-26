""" 
Jheronimus Academy of Data Science
Fire Department Amsterdam Amstelland

authors: Joep van den Bogaert
         Santiago Ruiz Zapata

date: June 25, 2018
"""

class SimulationEngine():
    """ Main class to simulate incidents and responses. """
    
    def __init__(self, mean_interarrival_time, location_probs, demand_locations, decision_agent):
        self.time = 0
        self.L = mean_interarrival_time
        self.location_probs = location_probs
        self.demand_locations = demand_locations
        self.agent = decision_agent

    
    def fit_incident_parameters(incident_log, deployment_log):
        """ Calculates the parameter values required for simulation from the raw data
            and stores them in the desired formats. 
        :param incident_log: Pandas DataFrame with incident data.
        :param deployment_log: Pandas DataFrame with deployment data. 
        """


        def get_prio_probabilities_per_type(incident_data):
            """ Create dictionary with the probabilities of having priority 1, 2, and 3 for 
                every incident type. 

            :param incident_data: Pandas DataFrame containing the log of incidents from which
                                  the probabilities should be obtained.
            :return: Dictionary with incident type names as keys and lists of length 3 as elements.
                     The lists have probabilities of prio 1, 2, 3 in position 0, 1, 2 respectively.
            """

            prio_per_type = incident_data.groupby(["dim_incident_incident_type", "dim_prioriteit_prio"])["dim_incident_id"].count().reset_index()
            prio_per_type["prio_probability"] = prio_per_type.groupby(["dim_incident_incident_type"])["dim_incident_id"].apply(lambda x: x/x.sum())
            prio_probabilities = pd.pivot_table(prio_per_type, columns="dim_incident_incident_type", values="prio_probability", index="dim_prioriteit_prio").fillna(0)
            return {col : list(prio_probabilities[col]) for col in prio_probabilities.columns}


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
                incident_data = incident_data.groupby("dim_incident_postcode_digits")["dim_incident_id"].count() / len(incident_data)
                return np.array(incident_data), np.array(incident_data.index)
            else:
                print("Only 'postcode_digits' is currently implemented.")

        def get_response_time_targets(incident_data):
            """ Determine the probability of having a certain respons time target.. """
            pass

        self.priority_probabilities_per_incident_type = get_prio_probabilities_per_type(incident_log)
        self.probability_per_demand_location, self.demand_location_names = get_prob_per_demand_location(incident_log)
        
        

    def return_finished_vehicles(self):
        """ Return vehicles that are finished solving an incident. """
        pass


    def generate_incident(self):

        def sample_deployment_requirements(type_of_incident):
            """ Returns the priority, response time target, and the required vehicles
                of the given incident type. For some incident types, these are fixed, 
                for others they have to be sampled.
            """
            # TODO: implement function
            return 1, 10

        incident_location_id = np.random.multinomial(1, self.location_probs)
        incident_type = self.demand_locations[incident_location_id].sample_incident_type()
        priority, response_target = self.sample_deployment_requirements(incident_type)

        return Incident(self.time, incident_type, priority, required_vehicles, incident_location_id)

    
    def get_state(self):
        pass


    def step(self):
        
        self.time = self.time + np.random.exponential(self.L, 1)
        self.return_finished_vehicles()
        incident = self.generate_incident()
        state = self.get_state()
        self.agent.deploy(incident, state)


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


class DemandLocation():
    """ An area in which incidents occur. """
    
    def __init__(self, location_id, incident_type_probs)
        self.id = location_id
        self.incident_type_probs = incident_type_probs
        
    def sample_incident_type(self):
        return np.random.multinomial(1, self.incident_type_probs)
    
    #def sample_priority(self, incident_type):
    #    return np.random.multinomial(1, self.priorities)


class FireStation():
    """ Represents a fire station. """
    
    def __init__(vehicles, location, incident_type_probs)
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


    
    
    
    