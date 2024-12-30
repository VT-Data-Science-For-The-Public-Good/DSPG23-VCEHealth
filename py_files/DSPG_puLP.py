# This script solves the baseline DSPG Health optimization problem

# Importing required modules

import pulp as pulp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Defining the project directory

direc = ''

# Reading in data

data = pd.read_csv(direc + 'data/program_data.csv')
a_districts = pd.read_csv(direc + 'data/district_agent_matrix.csv')
c_districts = pd.read_csv(direc + 'data/district_county_matrix.csv')

# Extracting data

a = data.a_long
c = data.c_long
d = data.d_long
p = data.p_long
z = data.z_long

agent_districts = [list(a_districts[a_districts.Agent == x][['Central', 'Northeast', 'Northwest', 'Southeast', 'Southwest']].reset_index(drop = True).iloc[0]) for x in a]
county_districts = [list(c_districts[c_districts.County == x][['Central', 'Northeast', 'Northwest', 'Southeast', 'Southwest']].reset_index(drop = True).iloc[0]) for x in c]

# Setting some parameters

n = len(c_districts.County.unique()) # counties
m = len(a_districts.Agent.unique()) # agents
T = 120 # commute threshold of 2 hours
P = 1000000 # population served upper bound

# Transforming / creating data

travel_constraint = d <= T # travel distance constraint
choice_vars = [a[i].replace(' ', '_') + '__' + c[i].replace(' ', '_') for i in range(len(a))] # choice variables
d = [1/x for x in d]
p = [np.log(x) for x in p]
z = [max(z) - x + 1 for x in z]

# Setting up the program

problem = pulp.LpProblem('Agent Optimization Problem', pulp.LpMaximize)

# Initialize a list of choice variables

x = [pulp.LpVariable(cv, lowBound = 0, upBound = 1, cat = 'Integer') for cv in choice_vars]

# Define the objective function

problem += pulp.lpSum([d[i]*x[i]*p[i]*z[i] for i in range(len(x))])

# Constraints

# Unique assignment of counties to agents

for i in range(n):
    
    problem += pulp.lpSum([x[j] for j in range(len(x)) if j%n == i]) <= 1

# Travel distance threshold

for i in range(len(x)):
    
    problem += pulp.lpSum(x[i] - travel_constraint[i]) <= 0

# Same region threshold

for i in range(len(x)):
    
    problem += pulp.lpSum(x[i] - np.matmul(agent_districts[i],county_districts[i])) <= 0

# Population upper bound threshold

for i in range(m):
    
    problem += pulp.lpSum([p[j]*x[i] for j in range(len(p)) if j%m == i]) <= P

# Solve this problem

problem.solve()

# Extracting the results

solution = []

for var in problem.variables():
    
    if var.varValue > 0:
        
        solution.append(var)

# Visualizing the number of counties per agent

ans = []

for i in range(m):
    
    count = 0
    
    for s in solution:
        
        if str(s).split('__')[0].replace('_', ' ') == a[133*i+1]:
            
            count += 1
            
    ans.append(count)

plt.hist(ans, bins = max(ans) - min(ans))

# Extracting the territories served by each agent

territories = []

for i in range(m):
    
    tmp = []
    
    for s in solution:
        
        if str(s).split('__')[0].replace('_', ' ') == a[133*i+1]:
            
            tmp.append(str(s).split('__')[1].replace('_', ' '))
            
    territories.append(tmp)

# Some descriptive statistics on the agents

def agent_data_function(locs, param, oper_mean, is_z, is_d, is_pz, ax):
    
    val = 0
    
    for loc in locs:
        
        tmp = data[data.c_long == loc].reset_index(drop = True)
        
        if is_z == True:
            
            val += (max(data.z_long) + 1 - tmp[param][0])
            
        elif is_d == True:
            
            tmp2 = tmp[tmp.a_long == ax].reset_index(drop = True)
            val += tmp2[param][0]
            
        elif is_pz == True:
            
            val += tmp['p_long'][0] * (max(data.z_long) + 1 - tmp['z_long'][0])
            
        else:
            
            val += tmp[param][0]
            
    if oper_mean == True:
        
        val = val / len(locs)
    
    return val

agent_p = [agent_data_function(t, 'p_long', False, False, False, False, list(a.unique())[territories.index(t)]) for t in territories]
agent_z = [agent_data_function(t, 'z_long', True, True, False, False, list(a.unique())[territories.index(t)]) for t in territories]
agent_d = [agent_data_function(t, 'd_long', True, False, True, False, list(a.unique())[territories.index(t)]) for t in territories]
agent_pz = [agent_data_function(t, '', False, False, False, True, list(a.unique())[territories.index(t)]) for t in territories]

agent_p = pd.Series(agent_p, name = 'Agent Population Served')
agent_z = pd.Series(agent_z, name = 'Agent Mean z-score Served')
agent_d = pd.Series(agent_d, name = 'Agent Mean Commute Time')
agent_pz = pd.Series(agent_pz, name = 'Agent Population Weighted Need Served')

agents_df = pd.concat([pd.Series(a.unique(), name = 'Agent'), agent_p, agent_z, agent_d, agent_pz, pd.Series(territories, name = 'Territory')], axis = 1)

# Save the descriptive statistics on the agents

agents_df.to_csv(direc + 'results/agent_descriptive_statistics.csv', index = False)

