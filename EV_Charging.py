import os
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Parameters to be adjusted
##############################################################################################################################


number_of_buildings = 20
averageLengthOfRides_km = 45  # Unit: [km]
chargingPowerMaximum_EV  = 4.7 * 1000  # Unit: [W]=[kW]*[W/kW]



initialSOC_EV = 0  # Unit: [%]
targetSOCAtEndOfOptimization_EV  = 0   # Unit: [%]
building_nr_for_plotting_the_soc = 8 #This number has to be smaller than number_of_buildings. It just indicates the specific EV for plotting the resulting charging schedules and battery state of charge (SOC) values

consider_household_appliances_demand = True # Indicates if the electrical demand of other household appliances (e.g. lighting, washing machine, tumble dryer etc. is also conisdered. This demand is not flexible and can't be controlled)



# Define a dictionary with prices for each hour (The data is from the day-ahead electricity market. Freely available data for different European market zones are available e.g. at Entsoe-Transparency plattform or SMARD 
price_per_hour_euro_per_mwh = {
    "00:00": 37.45,
    "01:00": 21.57,
    "02:00": 20.94,
    "03:00": 25.01,
    "04:00": 28.98,
    "05:00": 57.27,
    "06:00": 67.77,
    "07:00": 104.94,
    "08:00": 115.92,
    "09:00": 114.04,
    "10:00": 108.40,
    "11:00": 102.38,
    "12:00": 102.72,
    "13:00": 106.20,
    "14:00": 109.63,
    "15:00": 116.42,
    "16:00": 117.10,
    "17:00": 127.33,
    "18:00": 121.60,
    "19:00": 108.10,
    "20:00": 98.51,
    "21:00": 95.84,
    "22:00": 92.44,
    "23:00": 70.66
}

date_for_titles_of_the_plots = "17 December 2024" #Just for the titles of the plot. Does not have any influence on the optimization and simulation


#Parameters of the electric vehicle EV (currently Opel Ampera-e/Chevrolet Bolt)
capacityMaximum_EV = 60  * 3600000 # Unit: [J]=[kWh]*[J/kWh]
chargingEfficiency_EV = 89   # Unit: [%]
energyConsumptionPer100km = 17.5 * 3600000  # Unit: [J]=[kWh]*[J/kWh]


##############################################################################################################################


#Read the CSV Files into pandas DataFrames
list_df_buildingData = []
for i in range(1, 31):  # From HH1 to HH30
    file_name = f'Data\HH{i}.csv'
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, sep=';')  # Read the CSV file using semicolon as separator
        list_df_buildingData.append(df)  # Append the DataFrame to the list
    else:
        print(f"File {file_name} not found")

#Convert the price data into half an hour resolution
price_per_30min_euro_per_mwh = {
    hour: price for hour, price in price_per_hour_euro_per_mwh.items() for hour in [hour, hour[:-3] + ":30"]
}



# Create a model
model = pyo.ConcreteModel()


number_of_time_slots = 48

# Define the sets
model.set_timeslots = pyo.RangeSet(1, number_of_time_slots)  # Time slots from 1 to 48
model.set_buildings = pyo.RangeSet(1, number_of_buildings)


#Chance price from Euro/MWh to Cent/kWh
price_per_30min_cent_per_kwh = {key: value * 0.1 for key, value in price_per_30min_euro_per_mwh.items()}


# Map each time slot index to its corresponding half-hour string
hour_to_index = {i: f"{(i - 1) // 2:02}:{'00' if (i - 1) % 2 == 0 else '30'}" for i in range(1, number_of_time_slots + 1)}
index_to_hour = {v: k for k, v in hour_to_index.items()}

# Convert prices dictionary to match the time slot indices
price_dict = {i: price_per_30min_cent_per_kwh[hour_to_index[i]] for i in model.set_timeslots}

# Define the parameter for prices
model.param_price_cent_per_kWh = pyo.Param(model.set_timeslots, initialize=price_dict)



#Crate the combined dataframes for the single parameters
combinedDataframe_electricalDemand = pd.DataFrame()
combinedDataframe_availabilityPatternEV = pd.DataFrame()
combinedDataframe_energyConsumptionEV_Joule = pd.DataFrame()

for index in range (0, len(list_df_buildingData)):
    combinedDataframe_electricalDemand[index] = list_df_buildingData[index] ["Electricity [W]"]
    combinedDataframe_availabilityPatternEV[index] = list_df_buildingData [index]  ['Availability of the EV']
    combinedDataframe_energyConsumptionEV_Joule[index] = list_df_buildingData[index] ["Energy Consumption of the EV"]


#Define the parameters
def init_electricalDemand(model, i, j):
    if consider_household_appliances_demand == True:
        return combinedDataframe_electricalDemand.iloc[j-1, i-1]
    else:
        return 0

model.param_electricalDemand_In_W = pyo.Param(model.set_buildings, model.set_timeslots, mutable=True, initialize=init_electricalDemand)


def init_availabilityPatternEV(model, i, j):
    return combinedDataframe_availabilityPatternEV.iloc[j-1, i-1]

model.param_availabilityPerTimeSlotOfEV = pyo.Param(model.set_buildings, model.set_timeslots, mutable=True, initialize=init_availabilityPatternEV)

def init_energyConsumptionEV_Joule(model, i, j):
    return combinedDataframe_energyConsumptionEV_Joule.iloc[j-1, i-1] *(averageLengthOfRides_km/45)

model.param_energyConsumptionEV_Joule = pyo.Param(model.set_buildings, model.set_timeslots, mutable=True, initialize=init_energyConsumptionEV_Joule)


#Define the variables
model.variable_currentChargingPowerEV = pyo.Var(model.set_buildings, model.set_timeslots, within=pyo.NonNegativeReals, bounds=(0,chargingPowerMaximum_EV))
model.variable_energyLevelEV = pyo.Var(model.set_buildings, model.set_timeslots, within=pyo.NonNegativeReals, bounds=(0, capacityMaximum_EV))
model.variable_SOC_EV= pyo.Var(model.set_buildings, model.set_timeslots,  within=pyo.NonNegativeReals, bounds=(0,100))
model.variable_electricalPowerBuilding = pyo.Var(model.set_buildings, model.set_timeslots)
model.variable_electricalPowerTotal_residentialArea = pyo.Var(model.set_timeslots, within=pyo.NonNegativeReals)
model.variable_costsPerTimeSlot = pyo.Var(model.set_timeslots)
model.variable_objectiveMaximumLoad = pyo.Var(within=pyo.NonNegativeReals)
model.variable_objectiveCosts = pyo.Var()

timeResolution_InMinutes = 30
endSOC_EVAllowedDeviationFromInitalValue = initialSOC_EV - targetSOCAtEndOfOptimization_EV  # Unit: [%]


# Defining the constraints


# EV Energy Level
def energyLevelOfEVRule(model, i, t):
    if t == model.set_timeslots.first():
        return model.variable_energyLevelEV[i, t] == ((initialSOC_EV / 100) * capacityMaximum_EV) + (model.variable_currentChargingPowerEV[i, t] * (chargingEfficiency_EV / 100) * timeResolution_InMinutes * 60 - model.param_energyConsumptionEV_Joule[i, t])
    return model.variable_energyLevelEV[i, t] == model.variable_energyLevelEV[i, t - 1] + (model.variable_currentChargingPowerEV[i, t] * (chargingEfficiency_EV / 100) * timeResolution_InMinutes * 60 - model.param_energyConsumptionEV_Joule[i, t])

model.constraint_energyLevelOfEV = pyo.Constraint(model.set_buildings, model.set_timeslots, rule=energyLevelOfEVRule)



# Constraints for the minimal and maximal energy level of the EV at the end of the optimization horizon
def constraint_energyLevelOfEV_lastLowerLimitRule(model, i, t):
    return model.variable_energyLevelEV[i, model.set_timeslots.last()] >= ((initialSOC_EV - endSOC_EVAllowedDeviationFromInitalValue) / 100) * capacityMaximum_EV

model.constraint_energyLevelOfEV_lastLowerLimit = pyo.Constraint(model.set_buildings, model.set_timeslots, rule=constraint_energyLevelOfEV_lastLowerLimitRule)


def constraint_energyLevelOfEV_lastUpperLimitRule(model, i, t):
  if endSOC_EVAllowedDeviationFromInitalValue >= 0:
    return model.variable_energyLevelEV[i, model.set_timeslots.last()] <= ((initialSOC_EV + endSOC_EVAllowedDeviationFromInitalValue) / 100) * capacityMaximum_EV
  else:
    return pyo.Constraint.Skip

model.constraint_energyLevelOfEV_lastUpperLimit = pyo.Constraint(model.set_buildings, model.set_timeslots, rule=constraint_energyLevelOfEV_lastUpperLimitRule)



# SOC of the EV
def socOfEVRule(model, i, t):
    return model.variable_SOC_EV[i, t] == (model.variable_energyLevelEV[i, t] / capacityMaximum_EV) * 100

model.constraint_SOCofEV = pyo.Constraint(model.set_buildings, model.set_timeslots, rule=socOfEVRule)


# Constraint for the charging power: The EV can only be charged if it is at home (available)
def chargingPowerOfTheEVRule(model, i, t):
    return model.variable_currentChargingPowerEV[i, t] <= model.param_availabilityPerTimeSlotOfEV[i, t] * chargingPowerMaximum_EV

model.constraint_chargingPowerOfTheEV = pyo.Constraint(model.set_buildings, model.set_timeslots, rule=chargingPowerOfTheEVRule)


# Constraints for the electrical power
def electricalPowerBuildingRule(model, i, t):
    return model.variable_electricalPowerBuilding[i, t] == model.variable_currentChargingPowerEV[i, t] + model.param_electricalDemand_In_W[i, t]

model.constraint_electricalPowerBuilding = pyo.Constraint(model.set_buildings, model.set_timeslots, rule=electricalPowerBuildingRule)


 #Equation for calculating the total electrical power
def electricalPowerTotalRule (model, t):
    return model.variable_electricalPowerTotal_residentialArea [t] == sum (  model.variable_electricalPowerBuilding [setIndex, t] for setIndex in model.set_buildings)
model.constraint_electricalPowerTotal = pyo.Constraint(model.set_timeslots, rule = electricalPowerTotalRule)


 #Equation for calculating the costs per timeslot
def costsPerTimeSlotRule (model, t):
    return model.variable_costsPerTimeSlot [t] ==  (model.variable_electricalPowerTotal_residentialArea [t]) * timeResolution_InMinutes * 60 * (model.param_price_cent_per_kWh[t]/3600000)

model.constraint_costsPerTimeSlots = pyo.Constraint(model.set_timeslots, rule =costsPerTimeSlotRule )


#Equation for the objective: Minimize costs
def objective_minimizeCostsRule (model):
    return model.variable_objectiveCosts == sum ((model.variable_costsPerTimeSlot [t]) for t in model.set_timeslots)

model.constraints_objective_minimizeCosts = pyo.Constraint(rule = objective_minimizeCostsRule)


#Equations for calculating the maxium load. The absolute function is linearized by using 2 greater or equal constraints

def objective_maximumLoadRule_1 (model, t):
      return model.variable_objectiveMaximumLoad >= model.variable_electricalPowerTotal_residentialArea [t]

model.constraints_objective_maxiumLoad_1 = pyo.Constraint(model.set_timeslots, rule = objective_maximumLoadRule_1)



#Run 3 iteration of the optimization problem with different objectives
result_costs= np.zeros(4)
result_peak_load = np.zeros(4)

for iteration in range (0, 2):

  #First iteration
  if iteration ==0:
    objective_minimize_costs = True
    objective_minimize_peak_load = False


  #Second iteration
  if iteration ==1:
    objective_minimize_costs = True
    objective_minimize_peak_load = True


  #Define combined objective function for the optimization depending on the objectives specified in the file Run_Simulations
  weight_minimize_peak_load = 0.0
  weight_minimize_costs = 1.0
  if objective_minimize_peak_load == True and objective_minimize_costs == True:
    weight_minimize_peak_load = 0.5
    weight_minimize_costs = 0.5

  if objective_minimize_peak_load == True and objective_minimize_costs == False:
    weight_minimize_peak_load = 1.0
    weight_minimize_costs = 0.0

  if objective_minimize_peak_load == False and objective_minimize_costs == True:
    weight_minimize_peak_load = 0.0
    weight_minimize_costs = 1.0

  if objective_minimize_peak_load == False and objective_minimize_costs ==False:
    weight_minimize_peak_load = 0
    weight_minimize_costs = 0


  normalization_value_minimize_peak_load = 1
  normalization_value_minimize_costs = 1
  def objectiveRule_combined_general (model):

      return (weight_minimize_peak_load * (model.variable_objectiveMaximumLoad /normalization_value_minimize_peak_load) + weight_minimize_costs * (model.variable_objectiveCosts /normalization_value_minimize_costs))*100

  model.objective_combined_general = pyo.Objective( rule=objectiveRule_combined_general, sense =pyo.minimize)


  # Solve the model using GLPK
  print("Start of solving")
  solver = pyo.SolverFactory('cbc')
  solver.options["executable"] = r"C:\cbc\bin\cbc.exe"  # Path to the correct CBC executable

  # Set solver options
  solver.options['tmlim'] = 60      # Time limit in seconds

  # Solve the model
  solution = solver.solve(model, tee=True)


  #Check if the problem is solved or infeasible
  if (solution.solver.status == SolverStatus.ok  and solution.solver.termination_condition == TerminationCondition.optimal) or  solution.solver.termination_condition == TerminationCondition.maxTimeLimit:
      print("Result Status: Optimal")

  #Define help parameters for the printed files
  def init_param_helpTimeSlots (model, i,j):
      return j

  model.param_helpTimeSlots = pyo.Param(model.set_buildings, model.set_timeslots, mutable = True, initialize=init_param_helpTimeSlots)


  #Create pandas dataframe for displaying the results
  outputVariables_list = [model.param_helpTimeSlots, model.variable_electricalPowerBuilding,  model.variable_currentChargingPowerEV, model.variable_energyLevelEV, model.variable_SOC_EV, model.param_electricalDemand_In_W, model.param_availabilityPerTimeSlotOfEV, model.param_energyConsumptionEV_Joule, model.param_price_cent_per_kWh, model.set_timeslots]
  optimal_values_list = [[pyo.value(model_item[key]) for key in model_item] for model_item in outputVariables_list]
  results = pd.DataFrame(optimal_values_list)
  results= results.T
  results = results.rename(columns = {0:'timeslot', 1:'variable_electricalPower',   2:'variable_currentChargingPowerEV', 3:'variable_energyLevelEV_kWh', 4:'variable_SOC_EV', 5:'param_electricalDemand_In_W',  6:'param_availabilityPerTimeSlotOfEV', 7:'param_energyConsumptionEV', 8:'param_PriceElectricity [Cents]', 9:'set_timeslots'})
  cols = ['set_timeslots']
  results.set_index('set_timeslots', inplace=True)
  results['variable_SOC_EV'] = results['variable_SOC_EV'].round(2)
  results['variable_energyLevelEV_kWh'] = results['variable_energyLevelEV_kWh']/3600000
  results['variable_energyLevelEV_kWh'] = results['variable_energyLevelEV_kWh'].round(2)


  #Plot the results

  #Plot price curve

  # Prepare data for plotting
  time_labels = list(price_per_30min_euro_per_mwh.keys())
  prices = list(price_per_30min_euro_per_mwh.values())

  # Plotting

  # Line plot
  plt.figure(figsize=(10, 5))

  # Line plot
  plt.plot(time_labels, prices, marker='o', linestyle='-', color='gold')
  plt.title('Electricity Prices')
  plt.xlabel('Time of Day (HH:MM)')
  plt.ylabel('Price (Euro/MWh)')
  plt.xticks(rotation=90)
  plt.grid(True)


  # Adjust layout
  plt.tight_layout()
  plt.show()


  # Suffix for the titles
  suffix_for_titles = " - "
  if objective_minimize_costs ==True and objective_minimize_peak_load ==False:
    suffix_for_titles = suffix_for_titles + "Minimize Costs"
  if objective_minimize_costs ==True and objective_minimize_peak_load ==True:
    suffix_for_titles = suffix_for_titles + "Minimize Costs and Peak Load"
  if objective_minimize_costs ==False and objective_minimize_peak_load ==True:
    suffix_for_titles = suffix_for_titles + "Minimize Peak Load"


  # Compact way to define time labels (half-hour increments)
  time_labels = [f'{t//2:02d}:{t%2*30:02d}' for t in range(48)]

  # Extract power values from the Pyomo variable
  power_values = [pyo.value(model.variable_electricalPowerTotal_residentialArea[t]/1000) for t in model.set_timeslots]

  # Generate the plot
  plt.figure(figsize=(10, 4))
  plt.plot(time_labels, power_values, marker='o', linestyle='-', color='b')

  # Adding labels and title
  plt.xlabel('Time of Day (HH:MM)')
  plt.ylabel('Electrical Power (kW)')  # Updated the unit to kW
  plt.title(f'Residential Area Electrical Power Consumption {suffix_for_titles}')

  # Rotate x-ticks for better readability
  plt.xticks(rotation=90)

  # Display grid
  plt.grid(True)

  # Tight layout for better spacing
  plt.tight_layout()

  # Show the plot
  plt.show()


  #Function for plotting ev charging and soc of a specific building
  def plot_ev_data_for_building(results, building_nr_for_plotting):
      # Calculate the start and end row indices for the specific building
      start_idx = (building_nr_for_plotting - 1) * 48
      end_idx = building_nr_for_plotting * 48

      # Extract the rows for the selected building
      building_data = results.iloc[start_idx:end_idx]

      # Generate compact time labels for each half-hour interval (00:00 to 23:30)
      time_labels = [f'{t//2:02d}:{t%2*30:02d}' for t in range(48)]

      # Extract the data for the 48 timeslots for the specific building
      charging_power_kW = building_data['variable_currentChargingPowerEV'] / 1000  # Convert to kW
      soc_percentage = building_data['variable_SOC_EV']  # Already in percentage


      # Plotting
      fig, ax1 = plt.subplots(figsize=(10, 4))

      # Bar plot for current charging power (kW) on the left y-axis
      ax1.bar(time_labels, charging_power_kW, color='b', alpha=0.6, label='Charging Power (kW)')
      ax1.set_xlabel('Time of Day (HH:MM)')
      ax1.set_ylabel('Charging Power (kW)', color='b')
      ax1.tick_params(axis='y', labelcolor='b')

      # Plot SOC on the right y-axis without markers
      ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis
      ax2.plot(time_labels, soc_percentage, 'r-', label='SOC (%)')  # No markers
      ax2.set_ylabel('State of Charge (%)', color='r')
      ax2.tick_params(axis='y', labelcolor='r')

      # Set x-axis labels to vertical
      ax1.set_xticks(range(len(time_labels)))  # Set ticks at proper positions
      ax1.set_xticklabels(time_labels, rotation=90)  # Rotate labels 90 degrees

      # Title
      plt.title(f'EV Charging Power and SOC for Building {building_nr_for_plotting} {suffix_for_titles}')

      # Remove grid lines
      ax1.grid(False)
      ax2.grid(False)
      #ax2.set_ylim(0, 100)

      # Tight layout for better spacing
      plt.tight_layout()

      # Show plot
      plt.show()


  plot_ev_data_for_building(results, building_nr_for_plotting_the_soc)


  # Print the objective values
  print("")
  obj_value_costs = round(pyo.value(model.variable_objectiveCosts) / 100, 2)
  print(f"Objective Value Costs: {obj_value_costs} Euro")
  obj_value_max_load = round(pyo.value(model.variable_objectiveMaximumLoad) / 1000, 1)
  print(f"Objective Value Max Load: {obj_value_max_load} kW")
  print("")
  result_costs [iteration] = obj_value_costs
  result_peak_load[iteration] = obj_value_max_load



###############################################################################################
#Uncoordinated charging


# Convert DataFrames to transposed NumPy arrays
electrical_demand_array = combinedDataframe_electricalDemand.to_numpy().T
availability_pattern_EV_array = combinedDataframe_availabilityPatternEV.to_numpy().T
energy_consumption_EV_array = combinedDataframe_energyConsumptionEV_Joule.to_numpy().T

#Arrays for the variables
uc_charging_power_EV = np.zeros ((number_of_buildings, number_of_time_slots))
uc_energy_level_EV_Joule = np.zeros ((number_of_buildings, number_of_time_slots))
uc_SOC_EV = np.zeros ((number_of_buildings, number_of_time_slots))
uc_electricity_consumption_total_building = np.zeros ((number_of_buildings, number_of_time_slots))
uc_electricity_consumption_total_residential_area = np.zeros ((number_of_time_slots))

cost_in_cents = 0
current_peak_load = 0
price_per_30min_cent_per_kwh_array = list(price_per_30min_cent_per_kwh.values())


#Set the electrical demand of the building to 0 if it should not be considered
if consider_household_appliances_demand == False:
  electrical_demand_array = np.zeros ((number_of_buildings, number_of_time_slots))


#Initialize values
for i in range (0, number_of_buildings):
  uc_energy_level_EV_Joule [i, 0] = (initialSOC_EV / 100) * capacityMaximum_EV
  uc_SOC_EV [i, 0] = initialSOC_EV
  uc_electricity_consumption_total_building [i, 0] = uc_charging_power_EV [i, 0] + electrical_demand_array [i, 0]
  uc_electricity_consumption_total_residential_area [0] = uc_electricity_consumption_total_residential_area [0] + uc_electricity_consumption_total_building [i, 0]
  current_peak_load = uc_electricity_consumption_total_residential_area [0]

#Simulate the uncoordinated charging
for i in range (0, number_of_buildings):
  help_variable_ev_has_driven_current_day = False
  for t in range (1, number_of_time_slots):

    #Case 1: EV is available
    energy_required_for_target_SOC =0
    if availability_pattern_EV_array [i, t]  == 1:
      #Charge with full power if the EV is available and the target SOC will not be reached in the next timeslot
      energy_required_for_target_SOC = ((initialSOC_EV - endSOC_EVAllowedDeviationFromInitalValue) / 100) * capacityMaximum_EV - uc_energy_level_EV_Joule [i,t-1]
      uc_charging_power_EV [i, t] = energy_required_for_target_SOC / (timeResolution_InMinutes*60)

      #Charge with adjusted power if the EV is available and the target SOC would be reached in the next timeslot with full power
      if uc_charging_power_EV [i, t]> chargingPowerMaximum_EV:
        uc_charging_power_EV [i, t] = chargingPowerMaximum_EV

      #Don't charge if the target SOC has been reached
      if uc_charging_power_EV [i, t] < 0:
        uc_charging_power_EV [i, t] = 0

      #Don't charge if the EV has not driven so far on this day
      if help_variable_ev_has_driven_current_day == False:
        uc_charging_power_EV [i, t] = 0

    #Case 2: EV is not available
    if availability_pattern_EV_array [i, t]  == 0:
      uc_charging_power_EV [i, t] = 0


    #Update the energy content of the battery
    uc_energy_level_EV_Joule [i, t] = uc_energy_level_EV_Joule [i, t-1]  + (uc_charging_power_EV[i, t] * (chargingEfficiency_EV / 100) * timeResolution_InMinutes * 60 - energy_consumption_EV_array[i, t]* (averageLengthOfRides_km/45))
    uc_SOC_EV [i, t] = (uc_energy_level_EV_Joule [i, t] /capacityMaximum_EV) * 100


    #Calculate the total electricity consumption of the individual buildings
    uc_electricity_consumption_total_building [i, t] = uc_charging_power_EV [i, t] + electrical_demand_array [i, t]


    #Calculate the total electricity consumption of the whole residential area
    uc_electricity_consumption_total_residential_area [t] = uc_electricity_consumption_total_residential_area [t] + uc_electricity_consumption_total_building [i, t]


    #Determine the current peak load
    if uc_electricity_consumption_total_residential_area [t] > current_peak_load:
      current_peak_load = uc_electricity_consumption_total_residential_area [t]

    #Update the help variable to indicate if the EV has already driven on the current day
    if energy_consumption_EV_array[i, t] > 0.1:
      help_variable_ev_has_driven_current_day =True




#Calculate the total costs
for t in range (0, number_of_time_slots):
  cost_in_cents = cost_in_cents + uc_electricity_consumption_total_residential_area [t] * timeResolution_InMinutes * 60 * (price_per_30min_cent_per_kwh_array[t]/3600000)



#Plot the results of the uncoordinated charging

# Convert Watts to kilowatts by dividing by 1000
uc_electricity_consumption_total_residential_area_kW = uc_electricity_consumption_total_residential_area / 1000

# Time labels for each half-hour of the day
time_labels = [f'{t//2:02d}:{t%2*30:02d}' for t in range(48)]

# Generate the plot
plt.figure(figsize=(10, 4))
plt.plot(time_labels, uc_electricity_consumption_total_residential_area_kW, marker='o', linestyle='-', color='b')

# Adding labels and title
plt.xlabel('Time of Day (HH:MM)')
plt.ylabel('Electrical Power (kW)')
plt.title('Residential Area Electrical Power Consumption - Uncoordinated Charging')

# Rotate x-ticks for better readability
plt.xticks(rotation=90)

# Display grid
plt.grid(True)

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


# Function for plotting EV charging and SOC for a specific building
def plot_ev_data_for_building(uc_SOC_EV, uc_charging_power_EV, building_nr_for_plotting):
    # Select data for the specific building (0-indexed for numpy arrays)
    building_idx = building_nr_for_plotting - 1  # Convert to 0-indexed

    # Generate compact time labels for each half-hour interval (00:00 to 23:30)
    time_labels = [f'{t//2:02d}:{t%2*30:02d}' for t in range(48)]

    # Convert charging power from watts to kW for the specific building
    charging_power_kW = uc_charging_power_EV[building_idx] / 1000

    # Extract the SOC percentage data for the specific building
    soc_percentage = uc_SOC_EV[building_idx]

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Bar plot for current charging power (kW) on the left y-axis
    ax1.bar(time_labels, charging_power_kW, color='b', alpha=0.6, label='Charging Power (kW)')
    ax1.set_xlabel('Time of Day (HH:MM)')
    ax1.set_ylabel('Charging Power (kW)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot SOC on the right y-axis without markers
    ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis
    ax2.plot(time_labels, soc_percentage, 'r-', label='SOC (%)')  # No markers
    ax2.set_ylabel('State of Charge (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Set x-axis labels to vertical
    ax1.set_xticks(range(len(time_labels)))  # Set ticks at proper positions
    ax1.set_xticklabels(time_labels, rotation=90)  # Rotate labels 90 degrees

    # Title
    plt.title(f'EV Charging Power and SOC for Building {building_nr_for_plotting} - Uncoordinated Charging')

    # Remove grid lines
    ax1.grid(False)
    ax2.grid(False)

    # Tight layout for better spacing
    plt.tight_layout()

    # Show plot
    plt.show()

# Call the function with the numpy arrays for a specific building
plot_ev_data_for_building(uc_SOC_EV, uc_charging_power_EV, building_nr_for_plotting_the_soc)


# Print the objective values onto the console
obj_value_costs_uncoordinated = round((cost_in_cents) / 100, 2)
print(f"Objective Value Costs: {obj_value_costs_uncoordinated} Euro")
obj_value_max_load_uncoordinated = round((current_peak_load) / 1000, 1)
print(f"Objective Value Max Load: {obj_value_max_load_uncoordinated} kW")
print("")
result_costs [2] = obj_value_costs_uncoordinated
result_peak_load[2] = obj_value_max_load_uncoordinated


# Create combined plot
group_names = ["Min Costs", "Min Costs & Peak", "Uncoordinated"]

result_costs = [result_costs[0], result_costs[1], result_costs[2]]
result_peak_load = [result_peak_load[0], result_peak_load[1], result_peak_load[2]]

# Set the positions of the bars on the x-axis
x = np.arange(len(group_names))

# Width of the bars
width = 0.40

# Create the bar chart
fig, ax1 = plt.subplots(figsize=(9, 5))

# Create bars for costs (left y-axis)
bars1 = ax1.bar(x - width/2, result_costs, width, label='Costs in Euro', color='dodgerblue')

# Add labels and title for the first y-axis (left)
ax1.set_ylabel('Costs (€)', color='dodgerblue')
ax1.set_title(f'Resulting Costs and Peak Load for different charging strategies')

# Add the subtitle with smaller font size below the main title
ax1.text(0.5, -0.14,
         f'(Date: {date_for_titles_of_the_plots}, Buildings: {number_of_buildings}, '
         f'Length of rides: {averageLengthOfRides_km} km, Charging Power: {chargingPowerMaximum_EV} W, '
         f'Household appliances demand: {consider_household_appliances_demand})',
         ha='center', va='bottom', transform=ax1.transAxes, fontsize=10, color='gray')

ax1.set_xticks(x)
ax1.set_xticklabels(group_names)

# Set the color of the y-axis labels to match the bars
ax1.tick_params(axis='y', labelcolor='dodgerblue')

# Create a second y-axis for the peak load
ax2 = ax1.twinx()

# Create bars for peak load (right y-axis)
bars2 = ax2.bar(x + width/2, result_peak_load, width, label='Peak Load in kW', color='#FFC700')

# Add labels for the second y-axis (right)
ax2.set_ylabel('Peak Load (kW)', color='#FFC700')

# Set the color of the y-axis labels to match the bars
ax2.tick_params(axis='y', labelcolor='#FFC700')

# Adding value labels on top of the bars for costs
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, f'€{yval}', ha='center', va='bottom', color='dodgerblue')

# Adding value labels on top of the bars for peak load
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval, f'{yval} kW', ha='center', va='bottom', color='#FFC700')

# Tight layout to prevent overlap
fig.tight_layout()

# Show the plot
plt.show()