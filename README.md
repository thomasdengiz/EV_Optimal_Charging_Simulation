# Electric Vehicle Charging Simulation

## Content
This repository contains code for a small simulation of electric vehicle (EV) charging in a residential area. Three charging strategies are implemented:  
1. **Smart charging**: Minimize electricity costs.  
2. **Smart charging**: Minimize electricity costs and electrical peak load.  
3. **Uncoordinated charging**: Charge the EVs when they are available for charging.  

The simulation horizon is 24 hours with a time resolution of 30 minutes (resulting in 48 decisions for each charging strategy). A time-variable electricity tariff is used, based on data from the day-ahead market for electricity trading. The price data can be adjusted using freely available data from wholesale electricity markets, such as:  
- [ENTSO-E Transparency Platform](https://newtransparency.entsoe.eu/market/)  
- [Smard](https://www.smard.de/)  

The simulation produces as output the resulting costs and peak load for the residential area, as well as some exemplary charging and battery state-of-charge curves for a selected electric vehicle.  

Further, the `Data` folder contains demand profiles for 30 different households (HH). The data includes:  
- The charging demand of the EVs.  
- The demand of unflexible household appliances (e.g., lighting, dishwashers, washing machines).  
- The availability profiles of the EVs.  

![Residential_Area_EV](https://github.com/user-attachments/assets/2afc2160-7ec7-482f-8aae-b00abe90b6e2)


## Setup

The code was tested with Python 3.9. You can install the necessary packages listed in the `requirements.txt` file with:

`pip install -r requirements.txt`

The modeling framework Pyomo (http://www.pyomo.org/) is used for defining the optimization problems for the corresponding smart charging strategies. The resulting optimization problems are linear (and continuous) optimization problems.

Additionally, a solver is required for the smart charging strategies to solve the linear optimization problems. The free Cbc-Solver (https://github.com/coin-or/Cbc) was used in this simulation. You can use any other solver for (mixed-integer) linear programming that is compatible with the optimization framework Pyomo, e.g., the free GLPK solver (https://www.gnu.org/software/glpk/).

## Base simulation runs
In the upper part of the code, there is an area labeled # Parameters to be adjusted which contains the parameters you can change, like `number_of_buildings`, `chargingPowerMaximum_EV`, `chargingEfficiency_EV`, etc. There is a dictionary price_per_hour_euro_per_mwh which contains the prices of the time-variable electricity tariff. You can alter them by using data from the electricity market or just design your own tariff.

## Exemplary output
![Exemplary_Output_EV](https://github.com/user-attachments/assets/ff6d8a6e-ddbd-4809-b7da-643c97fa19c5)

