# Electric Vehicle Charging Simulation

## Content
This repository contains code for a small simulation of electric vehicle (EV) charging in a residential area. Three charging strategies are implemented:
1. Smart charging: Minimize electricity costs
2. Smart charging: Minimize electricity costs and electrical peak load
3. Uncoodinated charging (charge the EVs when they are available for charging)

The simulation horizon is 24 hours with a time resolution of 30 minutes (thus 48 decisions to be made by the three charging strategies). A time-variable electricity tarif is used using data from the day-ahead market for electricity trading. The price data can be adjusted by using freely available data from the wholesale electriciy market e.g. from EntsoE Transparency Platform (https://newtransparency.entsoe.eu/market/), Smard (https://www.smard.de/) or other free sources. 

The simulation creates as output the resulting costs and peak load for the residential area and some expemplary charging and battery state of charge curves of a selected electric vehicle. 

Furhter the "Data" folder contains demand profiles of 30 different households (HH). The data includes the charging demand of the EVs, the demand of the unflexible electrical appliances of the household (like lighting, dish washser, waching machine etc.) and the availability profiles of the EVs. 


![Residential_Area_EV](https://github.com/user-attachments/assets/2afc2160-7ec7-482f-8aae-b00abe90b6e2)

## Setup


## Base simulation runs
