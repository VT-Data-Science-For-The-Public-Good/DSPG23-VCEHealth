# This script solves the DSPG Health optimization problem for adding new agents for SNAP-Ed agents specifically

# Importing required modules

import pulp as pulp
import pandas as pd
import numpy as np
import itertools

# Defining the project directory

direc = ''

# Reading in data

data = pd.read_csv(direc + 'data/program_data.csv')
ndata = pd.read_csv(direc + 'data/program_data_new.csv')
sdata = pd.read_csv(direc + 'data/program_data_new_snap.csv')
nsdata = pd.read_csv(direc + 'data/program_data_new_non_snap.csv')
a_districts = pd.read_csv(direc + 'data/district_agent_matrix.csv')
na_districts = pd.read_csv(direc + 'data/district_new_agent_matrix.csv')
na_snap_districts = pd.read_csv(direc + 'data/district_new_agent_matrix_snap.csv')
na_non_snap_districts = pd.read_csv(direc + 'data/district_new_agent_matrix_non_snap.csv')
c_districts = pd.read_csv(direc + 'data/district_county_matrix.csv')
vce_data = pd.read_csv(direc + 'data/vce_agents.csv')

# Subsetting data for snap-ed agents

snap_agent_list = [vce_data.County[i] for i in range(len(vce_data)) if len(str(vce_data['SNAP-Ed?'][i])) > 3]
data = data[data.a_long.isin(snap_agent_list)].reset_index(drop = True)

# Extracting data

a = data.a_long
c = data.c_long
d = data.d_long
p = data.p_long
z = data.z_long
z_obese = data.z_obese_long
z_diabetes = data.z_diabetes_long
z_food = data.z_food_long
z_inactive = data.z_inactive_long
z_low_bw = data.z_low_bw_long

a2 = sdata.a_long
c2 = sdata.c_long
d2 = sdata.d_long
p2 = sdata.p_long
z2 = sdata.z_long
z2_obese = sdata.z_obese_long
z2_diabetes = sdata.z_diabetes_long
z2_food = sdata.z_food_long
z2_inactive = sdata.z_inactive_long
z2_low_bw = sdata.z_low_bw_long

snap_districts = a_districts[a_districts.Agent.isin(snap_agent_list)].reset_index(drop = True)

agent_districts = [list(snap_districts[snap_districts.Agent == x][['Central', 'Northeast', 'Northwest', 'Southeast', 'Southwest']].reset_index(drop = True).iloc[0]) for x in a]
new_agent_districts = [list(na_snap_districts[na_snap_districts.County == x][['Central', 'Northeast', 'Northwest', 'Southeast', 'Southwest']].reset_index(drop = True).iloc[0]) for x in a2]
county_districts = [list(c_districts[c_districts.County == x][['Central', 'Northeast', 'Northwest', 'Southeast', 'Southwest']].reset_index(drop = True).iloc[0]) for x in c]

# Setting some parameters

n = len(c_districts.County.unique()) # counties
m = len(snap_districts.Agent.unique()) # agents
m2 = len(na_snap_districts.County.unique()) # new agents
T = 180 # commute threshold of 2 hours
P = np.log(1200000) # population served upper bound

# Transforming / creating data

baseline_choice_vars = [a[i].replace(' ', '_') + '__' + c[i].replace(' ', '_') for i in range(len(a))] # choice variables
new_choice_vars = [a2[i].replace(' ', '_') + '__' + c2[i].replace(' ', '_') for i in range(len(a2))] # more potential choice variables

d = [1/x for x in d]
d2 = [1/x for x in d2]
p = [np.log(x) for x in p]
p2 = [np.log(x) for x in p2]

z = [max(z) - x + 1 for x in z]
z_obese = [max(z_obese) - x + 1 for x in z_obese]
z_diabetes = [max(z_diabetes) - x + 1 for x in z_diabetes]
z_food = [max(z_food) - x + 1 for x in z_food]
z_inactive = [max(z_inactive) - x + 1 for x in z_inactive]
z_low_bw = [max(z_low_bw) - x + 1 for x in z_low_bw]

z2 = [max(z2) - x + 1 for x in z2]
z2_obese = [max(z2_obese) - x + 1 for x in z2_obese]
z2_diabetes = [max(z2_diabetes) - x + 1 for x in z2_diabetes]
z2_food = [max(z2_food) - x + 1 for x in z2_food]
z2_inactive = [max(z2_inactive) - x + 1 for x in z2_inactive]
z2_low_bw = [max(z2_low_bw) - x + 1 for x in z2_low_bw]

new_agent_lists = []

for num_agents in range(1,3):
    
    tmp_list = []
    tmp_list_snap = []
    tmp_list_non_snap = []
    
    for subset in itertools.combinations(list(a2.unique()), num_agents):
        
        tmp_list.append(list(subset))
        
    new_agent_lists.append(tmp_list)

# Baseline model objective function optimized value

z_lists = [z, z_obese, z_diabetes, z_food, z_inactive, z_low_bw]
z2_lists = [z2, z2_obese, z2_diabetes, z2_food, z2_inactive, z2_low_bw]
model_names = ['aggregate', 'obesity', 'diabetes', 'food insecurity', 'physical inactivity', 'low birthweight']
of_vals = []

# Main loop

for num_agents in range(1,2):
    
    for z_score in range(len(z_lists)):
        
        good_z = z_lists[z_score]
        good_z2 = z2_lists[z_score]
        all_of_vals = []
        
        for k in range(len(new_agent_lists[num_agents-1])):
            
            # Status update
            
            print('Optimizing ' + model_names[z_score] + ' model with new agents at :: ' + str(new_agent_lists[num_agents-1][k]))
            
            # Update choice variable set
                    
            new_indices = [q for q in range(len(a2)) for qq in range(len(new_agent_lists[num_agents-1][k])) if a2[q] == new_agent_lists[num_agents-1][k][qq]]
            more_agents = [new_choice_vars[q] for q in new_indices]
            choice_vars = baseline_choice_vars + more_agents
            
            # Updating the problem data
            
            d2x = [d2[q] for q in new_indices]
            p2x = [p2[q] for q in new_indices]
            z2x = [good_z2[q] for q in new_indices]
            
            dd = d + d2x
            pp = p + p2x
            zz = good_z + z2x
            
            more_agent_districts = [new_agent_districts[q] for q in new_indices]
            more_county_districts = [county_districts[q] for q in range(n)]*int(len(more_agent_districts)/n)
            
            x_agent_districts = agent_districts + more_agent_districts
            x_county_districts = county_districts + more_county_districts
            
            travel_constraint = [q <= T for q in dd] # travel distance constraint
            
            # Setting up the program
            
            problem = pulp.LpProblem('Agent Optimization Problem', pulp.LpMaximize)
            
            # Initialize a list of choice variables
            
            x = [pulp.LpVariable(cv, lowBound = 0, upBound = 1, cat = 'Integer') for cv in choice_vars]
            
            # Define the objective function
            
            problem += pulp.lpSum([dd[i]*pp[i]*zz[i]*x[i] for i in range(len(x))])
            
            # Constraints
            
            # Unique assignment of counties to agents
            
            for i in range(n):
                
                problem += pulp.lpSum([x[j] for j in range(len(x)) if j%n == i]) <= 1
            
            # Travel distance threshold
            
            for i in range(len(x)):
                
                problem += pulp.lpSum(x[i] - travel_constraint[i]) <= 0
            
            # Same region threshold
            
            for i in range(len(x)):
                
                problem += pulp.lpSum(x[i] - np.matmul(x_agent_districts[i], x_county_districts[i])) <= 0
            
            # Population upper bound threshold
            
            for i in range(m + num_agents):
                
                problem += pulp.lpSum([pp[j]*x[i] for j in range(len(pp)) if j%m == i]) <= P
            
            # Solve this problem
            
            problem.solve()
            
            # Extracting the results
            
            obj_fx_val = problem.objective.value()
            all_of_vals.append(obj_fx_val)
            
        # Store the new_of_vals
        
        of_vals.append(all_of_vals)

# Generate outputs for the optimal combinations

all_agents = []
all_territories = []

for num_agents in range(1,2):
    
    for z_score in range(len(z_lists)):
        
        good_z = z_lists[z_score]
        good_z2 = z2_lists[z_score]
        
        # Update choice variable set
        
        new_indices = [q for q in range(len(a2)) if a2[q] in new_agent_lists[0][of_vals[z_score].index(max(of_vals[z_score]))]]
        more_agents = [new_choice_vars[q] for q in new_indices]
        choice_vars = baseline_choice_vars + more_agents
        
        # Updating the problem data
        
        d2x = [d2[q] for q in new_indices]
        p2x = [p2[q] for q in new_indices]
        z2x = [good_z2[q] for q in new_indices]
        
        dd = d + d2x
        pp = p + p2x
        zz = good_z + z2x
        
        more_agent_districts = [new_agent_districts[q] for q in new_indices]
        more_county_districts = [county_districts[q] for q in range(n)]*int(len(more_agent_districts)/n)
        
        x_agent_districts = agent_districts + more_agent_districts
        x_county_districts = county_districts + more_county_districts
        
        travel_constraint = [q <= T for q in dd] # travel distance constraint
        
        # Setting up the program
        
        problem = pulp.LpProblem('Agent Optimization Problem', pulp.LpMaximize)
        
        # Initialize a list of choice variables
        
        x = [pulp.LpVariable(cv, lowBound = 0, upBound = 1, cat = 'Integer') for cv in choice_vars]
        
        # Define the objective function
        
        problem += pulp.lpSum([dd[i]*pp[i]*zz[i]*x[i] for i in range(len(x))])
        
        # Constraints
        
        # Unique assignment of counties to agents
        
        for i in range(n):
            
            problem += pulp.lpSum([x[j] for j in range(len(x)) if j%n == i]) <= 1
        
        # Travel distance threshold
        
        for i in range(len(x)):
            
            problem += pulp.lpSum(x[i] - travel_constraint[i]) <= 0
        
        # Same region threshold
        
        for i in range(len(x)):
            
            problem += pulp.lpSum(x[i] - np.matmul(x_agent_districts[i], x_county_districts[i])) <= 0
        
        # Population upper bound threshold
        
        for i in range(m + num_agents):
            
            problem += pulp.lpSum([pp[j]*x[i] for j in range(len(pp)) if j%m == i]) <= P
        
        # Solve this problem
        
        problem.solve()
        
        # Extracting the results
        
        solution = []
        
        for var in problem.variables():
            
            if var.varValue > 0:
                
                solution.append(var)
                
        # Extracting the territories served by each agent
        
        agents = []
        territories = []
        a__ = [a2[ni] for ni in new_indices]
        a_ = list(a) + a__
        
        for i in range(m+1):
            
            tmp = []
            
            agents.append(a_[133*i+1])
            
            for s in solution:
                
                if str(s).split('__')[0].replace('_', ' ') == a_[133*i+1]:
                    
                    tmp.append(str(s).split('__')[1].replace('_', ' '))
                    
            territories.append(tmp)
            
        all_agents.append(agents)
        all_territories.append(territories)

