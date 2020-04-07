### DEV NOTES ###

###########
# IMPORTS #
###########

# Math
import math
import numpy as np
import scipy as sp

# Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Modeling
import tellurium as te
import roadrunner

# Warnings
import warnings

###########
# WARNING #
###########

# Supress warnings to avoid displaying messages about maximum output in simulartions
warnings.filterwarnings('ignore')

########
# DATA #
########

# Define data here
ZT_time = [6,12,18,24,30,36,42,48,54,60,66,72,78,84]
WTLD_raw = [0.281229082,0.231665789,0.206441193,0.112127825,0.240080584,0.293077347,0.278300003,0.165082388,0.293402011,0.195921232,0.209407965,0.123606977,0.211780881,0.204871613]
WTLD_avg = [0.258606591, 0.237126557, 0.231383054, 0.13360573, 0.258606591, 0.237126557, 0.231383054, 0.13360573, 0.258606591, 0.237126557, 0.231383054, 0.13360573, 0.258606591, 0.237126557]
WTLD_stdv = [0.044450904,0.077144211,0.04185786,0.028742127,0.044450904,0.077144211,0.04185786,0.028742127,0.044450904,0.077144211,0.04185786,0.028742127,0.044450904,0.077144211]

WTLL_raw = [0.275117457,0.166161164,0.245366441,0.331589035,0.184717796,0.121673682,0.256703756,0.308095114,0.245139991,0.120790619,0.238471027,0.315382649,0.263364342,0.236078465]
WTLL_avg = [0.245084357,0.152182379,0.236926219,0.305560796,0.245084357,0.152182379,0.236926219,0.305560796,0.245084357,0.152182379,0.236926219,0.305560796,0.245084357,0.152182379]
WTLL_stdv = [0.054058218,0.062642953,0.082472349,0.099781896,0.054058218,0.062642953,0.082472349,0.099781896,0.054058218,0.062642953,0.082472349,0.099781896,0.054058218,0.062642953]

CCA1_raw = [0.250483546,0.320523819,0.353884383,0.090987501,0.283542822,0.318876338,0.273294805,0.082071831,0.251416208,0.293756133,0.231249521,0.011293477,0.215529593,0.308025931]
CCA1_avg = [0.263446816,0.307190085,0.286142903,0.061450936,0.263446816,0.307190085,0.286142903,0.061450936,0.263446816,0.307190085,0.286142903,0.061450936,0.263446816,0.307190085]
CCA1_stdv = [0.06012161,0.052570095,0.078023862,0.048976372,0.06012161,0.052570095,0.078023862,0.048976372,0.06012161,0.052570095,0.078023862,0.048976372,0.06012161,0.052570095]

data_dict = {}
data_dict["WTLD"]={}
data_dict["WTLL"]={}
data_dict["CCA1"]={}

data_dict["WTLD"]["raw"] = WTLD_raw
data_dict["WTLD"]["avg"] = WTLD_avg
data_dict["WTLD"]["std"] = WTLD_stdv

data_dict["WTLL"]["raw"] = WTLL_raw
data_dict["WTLL"]["avg"] = WTLL_avg
data_dict["WTLL"]["std"] = WTLL_stdv

data_dict["CCA1"]["raw"] = CCA1_raw
data_dict["CCA1"]["avg"] = CCA1_avg
data_dict["CCA1"]["std"] = CCA1_stdv

#############
# FUNCTIONS #
#############

### Section 1 (Intro) ###

def update_model(model,parameter_dictionary):
    '''
       Update the parameters of Tellurium model using a dictionary of
       parameters (keys) and values (entries)
    '''

    model.resetToOrigin()
    for parameter in parameter_dictionary.keys():
        value  =  parameter_dictionary[parameter]
        model[parameter] = value

def load_model(model_file,parameter_dictionary):
    '''
       Load a model and set its parameters
    '''

    model = te.loada(model_file)
    update_model(model,parameter_dictionary)
    return model

def DisplayModel(model,format="Antimony"):
    '''
        Display the given model in a chosen format (Antimony, CellML, Matlab, SBML)
    '''

    if format == "Antimony":
        print(model.getCurrentAntimony())
    elif format == "Matlab":
        print(model.getCurrentMatlab())
    else:
        print("Format not recognized. Please chose one of the following: Antimony or Matlab")

### Section 2 (Figure 1) ###

def simulate_model(model,time_start=0,time_stop=200,time_step=20000):
    '''
       Simulate a tellurium model
    '''

    model_sim = model.simulate(time_start, time_stop, time_step,['time','DayNight','C1','C2','C3','C4','DayNight','E','F','RPS6p'])
    return model_sim

def correct_time(time_values,scalar,offset):
    '''
       Convert model time to 24-hour cycle using a scalar and offset
    '''
    return [v*scalar+offset for v in time_values]

def PlotModels_Fig1(model_sim,data_code="WTLD"):
    '''
       Plot models and underlying data for Figure
    '''

    # Plot the Model
    plt.plot(correct_time(model_sim['time'][11400:14800],(24/9.5),-286), model_sim['RPS6p'][11400:14800], c='black',linewidth=3,zorder=3)

    # Plot underlying data
    raw_values = data_dict[data_code]["raw"]
    avg_values = data_dict[data_code]["avg"]
    std_values = data_dict[data_code]["std"]

    plt.plot(ZT_time, avg_values, "ro",linewidth=2,zorder=2)
    plt.plot(ZT_time, raw_values, linestyle="",color="blue",marker=".",markersize=15)
    plt.errorbar(ZT_time, avg_values, std_values, capsize=2, c= 'red',linewidth=2,zorder=2)

    # Shade Night/Subjective Night
    night_alpha =  0.5
    if data_code == "WTLL":
       night_alpha = 0.25
    plt.axvspan(16,24,alpha=night_alpha,color='grey',zorder=1)
    plt.axvspan(40,48,alpha=night_alpha,color='grey',zorder=1)
    plt.axvspan(64,72,alpha=night_alpha,color='grey',zorder=1)
    plt.xlabel("Time")
    plt.ylabel("eS6-P")
    plt.xlim(2,87.88)
    plt.xticks(ZT_time)

### Section 3 (Figure 2&4) ###

def update_model_woReset(model,parameter_dictionary):
    '''
       Update the parameters of Tellurium model using a dictionary of
       parameters (keys) and values (entries) without reseting the 
       the model (Needed to make small changes)
    '''
    
    for parameter in parameter_dictionary.keys():
        value  =  parameter_dictionary[parameter]
        model[parameter] = value

def update_and_sim_model(model,modify_model,time_start = 0,time_stop = 150, time_step = 15000):
    '''
        Simulate a tellurium model after running a function to update the paramers
    '''
    
    # Rest model
    model.resetToOrigin()
    
    # Run update function
    modify_model()
    
    # Simulate
    model_sim = model.simulate(time_start, time_stop, time_step,['time','DayNight','C1','C2','C3','C4','DayNight','E','F','RPS6p'])
    return model_sim

def GetTimeIdxs(model_out,start_time,stop_time):
    '''
        Get indexes for a range of time values
    '''
    
    time_idxs = [i for i in range(len(model_out['time'])) if model_out['time'][i] > start_time and model_out['time'][i] < stop_time]
    return time_idxs

def PlotModelsOverTime(base_model,shifts,parameter_dictionary):
    '''
        Shift time of dawn and plot the resulting eS6 curves over one day
    '''
    
    # Define the time span with base model parameters
    def base_newpars():
        base_model.model['shift_hour'] = 0.0
    base_model_out = update_and_sim_model(base_model,base_newpars)
    
    # Define an early day
    base_dawn_time = base_model.model['dawn_time']
    base_time_idxs = GetTimeIdxs(base_model_out,base_dawn_time-9.5/2,base_dawn_time+9.5/2)
    
    # For each shift
    color_num = 1.0
    for shift in shifts:
        
        # Generate shift model
        def shift_newpars():
            base_model.model['shift_hour'] = shift
            update_model_woReset(base_model,parameter_dictionary)
        shift_model_out = update_and_sim_model(base_model,shift_newpars)
        
        # Get the early day max
        shift_dawn = base_dawn_time-9.5*(shift/24.0)
        shift_dawn_time = shift_dawn*(24/9.5)-240
        idxs_early = GetTimeIdxs(shift_model_out,shift_dawn,shift_dawn+9.5/6)
        shift_max = max(shift_model_out['RPS6p'][idxs_early])
        shift_max_idx = list(shift_model_out['RPS6p'][idxs_early]).index(shift_max)
        shift_max_time = shift_model_out['time'][idxs_early][shift_max_idx]*(24/9.5)-240
    
        # Plot eS6-P
        plt.plot(correct_time(shift_model_out['time'][base_time_idxs],24/9.5,-240), shift_model_out['RPS6p'][base_time_idxs], c=cm.plasma(color_num),linewidth=2.5)
      
        # Mark eS6 max and dawn time
        plt.plot(shift_max_time, shift_max, c='cyan',linestyle="",marker = ".", markersize=15)  
        plt.axvline(shift_dawn_time,0.0,0.05, c=cm.plasma(color_num),linewidth=2.5)
        
        # Shift Color
        color_num = color_num - 1.0/len(shifts)
        
    # Labels
    plt.xlabel("Time")
    plt.ylabel("eS6-P")
    plt.xticks([-12.0,-6.0,0.0,6.0,12.0])
    plt.xlim(-12.0,12.0)
    
def PlotModelsOverTime_Long(base_model,shifts,parameter_dictionary):
    '''
        Shift time of dawn and plot the resulting eS6 curves over multiple days
    '''
    
    # Setup of the base model
    def base_newpars():
        base_model.model['shift_hour'] = 0.0
    base_model_out = update_and_sim_model(base_model,base_newpars)
    
    # Define the time span with base model parameters
    base_time_idxs_2day = GetTimeIdxs(base_model_out,base_model.model['prev_dusk'],base_model.model['prev_dusk']+9.5*2)
    
    # For each shift
    color_num = 1.0
    for shift in shifts:
        
        # Generate shift model
        def shift_newpars():
            base_model.model['shift_hour'] = shift
            update_model_woReset(base_model,parameter_dictionary)
        shift_model_out = update_and_sim_model(base_model,shift_newpars)
    
        # Plot eS6-P
        plt.plot(correct_time(shift_model_out['time'][base_time_idxs_2day],24/9.5,-240), shift_model_out['RPS6p'][base_time_idxs_2day], c=cm.plasma(color_num),linewidth=2.5)
        
        # Shift Color
        color_num = color_num - 1.0/len(shifts)
        
    # Labels
    plt.xlabel("Time")
    plt.ylabel("eS6-P")
    plt.xticks([-12.0,-6.0,0.0,6.0,12.0,18.0,24.0,30.0])
    plt.xlim(-14.0,34.0)
        
def PlotModelsOverTime_Long_Clock(base_model,shifts,parameter_dictionary):
    '''
        Shift time of dawn and plot the resulting clock component curves over multiple days
    '''
    
    # Setup of the base model
    def base_newpars():
        base_model.model['shift_hour'] = 0.0
    base_model_out = update_and_sim_model(base_model,base_newpars)
    
    # Define the time span with base model parameters
    base_time_idxs_2day = GetTimeIdxs(base_model_out,base_model.model['prev_dusk'],base_model.model['prev_dusk']+9.5*2)
    
    # For each shift
    color_num = 1.0
    for shift in shifts:
        
        # Generate shift model
        def shift_newpars():
            base_model.model['shift_hour'] = shift
            update_model_woReset(base_model,parameter_dictionary)
        shift_model_out = update_and_sim_model(base_model,shift_newpars)
    
        # Plot Clock Components
        plt.plot(correct_time(shift_model_out['time'][base_time_idxs_2day],24/9.5,-240), shift_model_out['C1'][base_time_idxs_2day], c='red',alpha=0.33,linewidth=2.5)
        plt.plot(correct_time(shift_model_out['time'][base_time_idxs_2day],24/9.5,-240), shift_model_out['C2'][base_time_idxs_2day], c='blue',alpha=0.33,linewidth=2.5)
        plt.plot(correct_time(shift_model_out['time'][base_time_idxs_2day],24/9.5,-240), shift_model_out['C3'][base_time_idxs_2day], c='orange',alpha=0.33,linewidth=2.5)
        plt.plot(correct_time(shift_model_out['time'][base_time_idxs_2day],24/9.5,-240), shift_model_out['C4'][base_time_idxs_2day], c='green',alpha=0.33,linewidth=2.5)
        
        # Shift Color
        color_num = color_num - 1.0/len(shifts)

    # Labels
    plt.xlabel("Time")
    plt.ylabel("Clock Components")
    plt.xticks([-12.0,-6.0,0.0,6.0,12.0,18.0,24.0,30.0])
    plt.xlim(-14.0,34.0)

def PlotModelsOverTime_DuskAndDawn(base_model,shifts,parameter_dict,null_model="None",reset=True):
    
    # Setup of the base model
    def base_newpars():
        base_model.model['shift_hour'] = 0.0
    base_model_out = update_and_sim_model(base_model,base_newpars)
    
    base_dawn_time = base_model.model['dawn_time']
    base_dusk_time = base_model.model['prev_dusk']
    
    base_time_idxs_middle = GetTimeIdxs(base_model_out,base_model.model['dawn_time']+(9.5*(3/8)),base_model.model['dawn_time']+(9.5*(11/8)))
    
    color_num = 1.0
    index=0
    for shift in shifts:
        
        index = index + 1
        
        #shift = 2.0 # TEMP
        def shift_newpars():
            base_model.model['shift_hour'] = shift
            if reset == True:
                update_model_woReset(base_model,parameter_dict)
        shift_model_out = update_and_sim_model(base_model,shift_newpars)
    
        # Find shifted dawn and dusk times
        shift_dawn = base_dawn_time -9.5*(shift/24.0)
        shift_dawn_time = shift_dawn*(24/9.5)-264
        
        shift_dusk = base_dusk_time +9.5*(shift/24.0)
        shift_dusk_time = shift_dusk*(24/9.5)-264
    
        # Base 
        plt.plot(correct_time(shift_model_out['time'][base_time_idxs_middle],24/9.5,-264), shift_model_out['RPS6p'][base_time_idxs_middle], c=cm.plasma(color_num),linewidth=2.5)

        plt.xticks([-12,-6,0,6,12])
        if null_model == "Dusk":
            plt.axvline(x=-12,ymin=0.0,ymax=0.075-shift*0.01,color=cm.plasma(color_num),alpha=0.1*index)
        else:
            plt.axvline(x=-12+shift,ymin=0.0,ymax=0.075-shift*0.01,color=cm.plasma(color_num),alpha=0.1*index)
        if null_model == "Dawn":
            plt.axvline(x=0,ymin=0.0,ymax=0.075-shift*0.01,color=cm.plasma(color_num),alpha=0.1*index)
        else:
            plt.axvline(x=0-shift,ymin=0.0,ymax=0.075-shift*0.01,color=cm.plasma(color_num),alpha=0.1*index)
        plt.ylim(0.05,0.30)
        
        # Shift Color
        color_num = color_num - 1.0/len(shifts)
    
    # Labels
    plt.xlabel("Time")
    plt.xlim([-15.0,9.0])
    plt.ylabel("eS6-P")

### Section 4 (Figures 5&6) ###

def make_day_dict_V2(model):
    '''
       Create a dictionary mapping differents days of the year to their assocaited indexes
    '''

    day = 0
    increment = 0
    all_dict = {}
    
    for i in range(len(model['DayNight'])-1):
    
        # Check if it is the start of the day
        if model['DayNight'][i-1] < 1 and model['DayNight'][i] > 0:
            day = day + 1
            
        increment = increment + 1 
        all_dict[i] = {'day': day, 'increment': increment, 'index': i, 'LD': "D"}
        
        if model['DayNight'][i] > 0:
            all_dict[i]['LD'] = 'L'
        
        # Check if it is end of the day
        if model['DayNight'][i+1] < 1:
            increment = 0
            
    all_day_dict = {}
    for k in all_dict.keys():
        day = all_dict[k]['day']
        LD = all_dict[k]['LD']
    
        if day > 2:
    
            if day in all_day_dict.keys():
                all_day_dict[day]['index'].append(k)
                if LD == "L":
                    all_day_dict[day]['light_index'].append(k)
                else:
                    all_day_dict[day]['dark_index'].append(k)
        
            else:
                all_day_dict[day]= {'index': [], 'light_index': [], 'dark_index': []}
                all_day_dict[day]['index'].append(k)
                if LD == "L":
                    all_day_dict[day]['light_index'].append(k)
                else:
                    all_day_dict[day]['dark_index'].append(k)
        
    return all_day_dict

def LongestConsecutiveStart(int_list):
    '''
       Find the start of the longest consecutive list of indexes
    '''

    longest_streak = 0
    num_set = set(int_list)
    
    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1
            start_num = num

            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1

            if current_streak > longest_streak:
                longest_streak = current_streak
                longest_start = start_num

    return longest_start

def get_daily_stats_V2(model,day_dict,scale=1):
    '''
       Get statistics for each day in dictionary of days and their indexes
    '''
    
    daily_stats_dict = {
        "DayIdxs": [],
        "LightIdxs": [],
        "NightIdxs": [],
        "DayLength": [],
        "MinRPS6p": [],
        "MaxRPS6p": [],
        "MinRPS6pEarly": [],
        "MaxRPS6pEarly": [],
        "EOD": [],
        "SteadyStateRPS6p": [],
        "AvgRPS6p": [],
        "AvgRPS6p_Light": [],
        "AvgRPS6p_Dark": [],
        "AvgRPS6p_Early": [],
        "AvgRPS6p_Late": [],
        "IntegralRPS6p": [],
        "IntegralRPS6p_Light": [],
        "IntegralRPS6p_Dark": [],
        "IntegralRPS6p_Early": [],
        "IntegralRPS6p_Late": []
    }
    
    for day in day_dict:
        
        # Get Indexes for the day
        day_idxs = day_dict[day]['index']
        light_idxs = day_dict[day]['light_index']
        night_idxs = day_dict[day]['dark_index']
        
        # Save those indexes
        daily_stats_dict["DayIdxs"].append(day_idxs)
        daily_stats_dict["LightIdxs"].append(light_idxs)
        daily_stats_dict["NightIdxs"].append(night_idxs)
        
        # Get time, light, and eS6-P values
        time_values = model['time'][day_idxs]
        daynight_values = model['DayNight'][day_idxs]
        rps6p_values = model['RPS6p'][day_idxs]*scale
        
        # Get time, light, and eS6-P values during the day
        light_time_values = model['time'][light_idxs]
        light_daynight_values = model['DayNight'][light_idxs]
        light_rps6p_values = model['RPS6p'][light_idxs]*scale
        
        # Get time, light, and eS6-P values during the night
        dark_time_values = model['time'][night_idxs]
        dark_daynight_values = model['DayNight'][night_idxs]
        dark_rps6p_values = model['RPS6p'][night_idxs]*scale
        
        # Get values at the beginning and end of the day
        early_day_rps6p_values = model['RPS6p'][light_idxs][0:17]*scale
        
        # Min and Max values
        min_rps6p = min(rps6p_values)
        max_rps6p = max(rps6p_values)
        min_rps6p_early = min(early_day_rps6p_values)
        max_rps6p_early = max(early_day_rps6p_values)
        
        # End of Day and Steady State
        End_of_day = LongestConsecutiveStart(night_idxs)-1
        steady_state_rps6p = model['RPS6p'][End_of_day]
        
        daily_stats_dict["DayLength"].append(len(light_idxs)*0.24)
        
        daily_stats_dict["MinRPS6p"].append(min_rps6p)
        daily_stats_dict["MaxRPS6p"].append(max_rps6p)
        daily_stats_dict["MinRPS6pEarly"].append(min_rps6p_early)
        daily_stats_dict["MaxRPS6pEarly"].append(max_rps6p_early)
        daily_stats_dict["EOD"].append(End_of_day)
        daily_stats_dict["SteadyStateRPS6p"].append(steady_state_rps6p)
        
    return(daily_stats_dict)
        

def PlotYearlyData(model,parameter_dict,location,time_start=0,time_stop=3467,time_step=34670):
    '''
       Plot all eS6 values from yearly data
    '''

    # Update eS6-P Model Parameters for Best Fit
    update_model(model,parameter_dict)

    # Update clock for Oslo
    if location == "Oslo":
        model['coef_a'] = 0.08
        model['coef_b'] = 0.775
    elif location == "Paria":
        model['coef_a'] = 0.1
        model['coef_b'] = 0.125
    elif location == "Boston":
        model['coef_a'] = 0.09
        model['coef_b'] = 0.40
    else:
        print("Unrecognized Location: Defaulting to Boston. Known locations are Oslo, Paria, and Boston.")
        model['coef_a'] = 0.09
        model['coef_b'] = 0.40

    # Simulate
    model_out = model.simulate(time_start,time_stop,time_step,['time','DayNight','RPS6p','C1','C2','C3','C4'])

    # Correct time for days, skip burn-in days, and plot
    plt.plot(correct_time(model_out['time'][190:],(1/9.5),0),model_out['RPS6p'][190:],color='lightgrey')
    plt.ylim(0.075,0.35)

    # Get Daily Stats
    days_model_V2 = make_day_dict_V2(model_out)
    model_daily_stats_V2 = get_daily_stats_V2(model_out,days_model_V2)

    days_list = range(2,len(model_daily_stats_V2["EOD"]))
    SteadyState_list = [model_daily_stats_V2["SteadyStateRPS6p"][d] for d in days_list]
    EarlyMin_list = [model_daily_stats_V2["MinRPS6pEarly"][d] for d in days_list]
    EarlyMax_list = [model_daily_stats_V2["MaxRPS6pEarly"][d] for d in days_list]
    peak_metric = [EarlyMax_list[i]/SteadyState_list[i] for i in range(len(SteadyState_list))]

    # Plot Daily Stats
    plt.plot(days_list,SteadyState_list,color='black',linewidth=3,label='Steady State')
    plt.plot(days_list,EarlyMax_list,color='turquoise',linewidth=3,label='Early Max')
    plt.plot(days_list,EarlyMin_list,color='orchid',linewidth=3,label='Early Min')

    plt.ylabel("eS6-P")
    plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0.)

    ax2 = plt.twinx()
    ax2.plot(days_list,peak_metric,color='goldenrod',linewidth=3,label='Peak Metric')

    plt.ylabel("Early Peak/Steady State")
    plt.xlabel("Time (days)")
    plt.xlim(0,365)
    plt.ylim(0.925,1.5)

    plt.legend(bbox_to_anchor=(1.2, 0.5), loc='upper left', borderaxespad=0.)

def PlotDailyData(model,parameter_dict,location,time_start=0,time_stop=3467,time_step=34670):
    '''
       Plot eS6 values from overlapping days from full year models
    '''

    # Update eS6-P Model Parameters for Best Fit
    update_model(model,parameter_dict)

    # Update clock for Oslo
    if location == "Oslo":
        model['coef_a'] = 0.08
        model['coef_b'] = 0.775
    elif location == "Paria":
        model['coef_a'] = 0.1
        model['coef_b'] = 0.125
    elif location == "Boston":
        model['coef_a'] = 0.09
        model['coef_b'] = 0.40
    else:
        print("Unrecognized Location: Defaulting to Boston. Known locations are Oslo, Paria, and Boston.")
        model['coef_a'] = 0.09
        model['coef_b'] = 0.40

    # Simulate
    model_out = model.simulate(time_start,time_stop,time_step,['time','DayNight','RPS6p','C1','C2','C3','C4'])

    # Get Daily Stats
    days_model_V2 = make_day_dict_V2(model_out)
    model_daily_stats_V2 = get_daily_stats_V2(model_out,days_model_V2)

    days =[7,21,35,49,63,77,91,105,119,133,147,161,175]

    color_num = 1.0
    for d in days:
    
        # Get indexes for the day
        day_idxs = model_daily_stats_V2["DayIdxs"][d]
    
        # Define end of day and early max vlaue
        EoD = model_daily_stats_V2["EOD"][d]
        EMax = model_daily_stats_V2["MaxRPS6pEarly"][d]
    
        # Get time values and eS6-P values
        times = model_out['time'][day_idxs]
        values = model_out['RPS6p'][day_idxs]
    
        # Adjust times so the day starts at zero
        start_time = times[0]
        times = [(t - start_time)*(24.0/9.5) % 24 for  t in times]
    
        # Get time of the early-day max
        values = list(values)
        EMax_idx = values.index(EMax)
        Emax_time = times[EMax_idx]
        Emax_value = values[EMax_idx]
    
        # Plot
        plt.plot(times, values, c=cm.plasma(color_num),linewidth=2.5)
        plt.plot(model_out['time'][EoD]*(24.0/9.5) % 24, 0.05, c=cm.plasma(color_num),marker="^",markersize=12,label='Steady State')
        plt.plot(Emax_time, Emax_value, c="turquoise",marker=".",markersize=12,label='Early Max')
    
        # Shift Color
        color_num = color_num - 1.0/len(days)

    plt.xticks([0,6,12,18,24])
    plt.xlim(0,24)
    plt.xlabel("Time (hours)")
    plt.ylabel("eS6-P")

def make_day_dict_HF(model,base_model):
    '''
       Get days for the Harvard Forest data by comparison to NOAA baseline
    '''

    day = 0
    increment = 0
    all_dict = {}
    
    for i in range(len(base_model['DayNight'])-41):
    
        # Check if it is the start of the day
        if base_model['DayNight'][i-1] < 1 and base_model['DayNight'][i] > 0:
            day = day + 1
            
        increment = increment + 1 
        all_dict[i+40] = {'day': day, 'half_hour': increment, 'index': i, 'LD': "D"}
        
        if model['DayNight'][i+40] > 0:
            all_dict[i+40]['LD'] = 'L'
        
        # Check if it is end of the day
        if base_model['DayNight'][i+1] < 1:
            increment = 0
            
    all_day_dict = {}
    for k in all_dict.keys():
        day = all_dict[k]['day']
        LD = all_dict[k]['LD']
    
        #print(day)
        if day > 2:
    
            if day in all_day_dict.keys():
                all_day_dict[day]['index'].append(k)
                if LD == "L":
                    all_day_dict[day]['light_index'].append(k)
                else:
                    all_day_dict[day]['dark_index'].append(k)
        
            else:
                all_day_dict[day]= {'index': [], 'light_index': [], 'dark_index': []}
                all_day_dict[day]['index'].append(k)
                if LD == "L":
                    all_day_dict[day]['light_index'].append(k)
                else:
                    all_day_dict[day]['dark_index'].append(k)
        
    return all_day_dict

def PlotYearlyData_HF(model,base_model,parameter_dict,time_start=0,time_stop=3467,time_step=34670):
    '''
       Plot eS6 values for yearly Harvard Forest data
    '''

    # Update eS6-P Model Parameters for Best Fit
    update_model(model,parameter_dict)
    update_model(base_model,parameter_dict)

    # Update clock for Boston
    model['coef_a'] = 0.09
    model['coef_b'] = 0.40

    base_model['coef_a'] = 0.09
    base_model['coef_b'] = 0.40

    # Simulate
    model_out = model.simulate(time_start,time_stop,time_step,['time','DayNight','RPS6p','C1','C2','C3','C4'])
    base_model_out = base_model.simulate(time_start,time_stop,time_step,['time','DayNight','RPS6p','C1','C2','C3','C4'])
    
    days_HF_model = make_day_dict_HF(model_out,base_model_out)
    HF_daily_stats_V2 = get_daily_stats_V2(model_out,days_HF_model)

    # Correct time for days, skip burn-in days, and plot
    plt.plot(model_out['time'][190:]*(1/9.5),model_out['RPS6p'][190:],color='lightgrey')
    plt.ylim(0.075,0.35)

    # Get Daily Stats
    days_list = range(2,len(HF_daily_stats_V2["EOD"]))
    SteadyState_list = [HF_daily_stats_V2["SteadyStateRPS6p"][d] for d in days_list]
    EarlyMin_list = [HF_daily_stats_V2["MinRPS6pEarly"][d] for d in days_list]
    EarlyMax_list = [HF_daily_stats_V2["MaxRPS6pEarly"][d] for d in days_list]
    peak_metric = [EarlyMax_list[i]/SteadyState_list[i] for i in range(len(SteadyState_list))]
    real_peak_metric = peak_metric

    # Plot Daily Stats
    plt.plot(days_list,SteadyState_list,color='black',linewidth=3,label='Steady State')
    plt.plot(days_list,EarlyMax_list,color='turquoise',linewidth=3,label='Early Max')
    plt.plot(days_list,EarlyMin_list,color='orchid',linewidth=3,label='Early Min')

    plt.ylabel("eS6-P")
    plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0.)

    ax2 = plt.twinx()
    ax2.plot(days_list,peak_metric,color='goldenrod',linewidth=3,label='Peak Metric')

    plt.ylabel("Early Peak/Steady State")
    plt.xlabel("Time (days)")
    plt.xlim(0,365)
    plt.ylim(0.925,1.5)

    plt.legend(bbox_to_anchor=(1.2, 0.5), loc='upper left', borderaxespad=0.)

def BinDays(model,base_model,HF_daily_stats_V2,model_daily_stats_V2):
    '''
       Bin days according to their day-length
    '''

    # Make holder dictionary
    daylength_dict_HF= {}
    for i in range(16,30,1):
        bin_start=(float(i)/2.0)
        bin_stop=(float(i+1)/2.0)
        bin_avg=(bin_start+bin_stop)/2.0
        daylength_dict_HF[bin_avg]={'start': bin_start,'stop': bin_stop, "days": []}


    daylength_dict_NOAA= {}
    for i in range(16,30,1):
        bin_start=(float(i)/2.0)
        bin_stop=(float(i+1)/2.0)
        bin_avg=(bin_start+bin_stop)/2.0
        daylength_dict_NOAA[bin_avg]={'start': bin_start,'stop': bin_stop, "days": []}

    # Bins days
    for d in range(len(HF_daily_stats_V2["DayLength"])):
        DayLength = HF_daily_stats_V2["DayLength"][d] 
        bin_key = -1
    
        for k in daylength_dict_HF.keys():
            bin_start = daylength_dict_HF[k]["start"]
            bin_stop = daylength_dict_HF[k]["stop"]
            if DayLength >= bin_start and DayLength < bin_stop:
                bin_key = k
    
        if not bin_key == -1:
            daylength_dict_HF[bin_key]['days'].append(d)
        
    for d in range(len(model_daily_stats_V2["DayLength"])):
        DayLength = model_daily_stats_V2["DayLength"][d]     
        bin_key = -1
    
        for k in daylength_dict_NOAA.keys():
            bin_start = daylength_dict_NOAA[k]["start"]
            bin_stop = daylength_dict_NOAA[k]["stop"]
            if DayLength >= bin_start and DayLength < bin_stop:
                bin_key = k
    
        if not bin_key == -1:
            daylength_dict_NOAA[bin_key]['days'].append(d)

    return(daylength_dict_HF,daylength_dict_NOAA)

def PlotBinnedPeakMetric(model,base_model,parameter_dict,time_start=0,time_stop=3467,time_step=34670):
    '''
       Plot peak matric for binned NOAA and Harvard Forest Days
    '''

    # Update eS6-P Model Parameters for Best Fit
    update_model(model,parameter_dict)
    update_model(base_model,parameter_dict)

    # Update clock for Boston
    model['coef_a'] = 0.09
    model['coef_b'] = 0.40

    base_model['coef_a'] = 0.09
    base_model['coef_b'] = 0.40

    # Simulate
    model_out = model.simulate(time_start,time_stop,time_step,['time','DayNight','RPS6p','C1','C2','C3','C4'])
    base_model_out = base_model.simulate(time_start,time_stop,time_step,['time','DayNight','RPS6p','C1','C2','C3','C4'])
    
    days_HF_model = make_day_dict_HF(model_out,base_model_out)
    HF_daily_stats_V2 = get_daily_stats_V2(model_out,days_HF_model)

    days_model_V2 = make_day_dict_V2(base_model_out)
    model_daily_stats_V2 = get_daily_stats_V2(base_model_out,days_model_V2)

    daylength_dict_HF, daylength_dict_NOAA = BinDays(model,base_model,HF_daily_stats_V2,model_daily_stats_V2)

    color_num = 1.0

    # All bins in the HF data
    for bin in daylength_dict_HF.keys():
        bin_start = daylength_dict_HF[bin]['start']
        bin_stop = daylength_dict_HF[bin]['stop']
        bin_days = daylength_dict_HF[bin]['days']
    
        SteadyState_list = [HF_daily_stats_V2["SteadyStateRPS6p"][d] for d in bin_days]
        EarlyMax_list = [HF_daily_stats_V2["MaxRPS6pEarly"][d] for d in bin_days]
        peak_metric = [EarlyMax_list[i]/SteadyState_list[i] for i in range(len(SteadyState_list))]
    
        peak_metric_avg = np.mean(peak_metric)
        peak_metric_std = np.std(peak_metric)
    
        plt.plot(bin, peak_metric_avg,color=cm.plasma(color_num),linestyle="",marker=".",markersize=10)
        plt.errorbar(bin, peak_metric_avg, yerr=peak_metric_std,color=cm.plasma(color_num))

        color_num = color_num - 1.0/len(daylength_dict_HF.keys())
    
        # If bin aslo exists in NOAA data, which has narrow range
        if bin in daylength_dict_NOAA.keys():
            bin_start = daylength_dict_NOAA[bin]['start']
            bin_stop = daylength_dict_NOAA[bin]['stop']
            bin_days = daylength_dict_NOAA[bin]['days']
    
            SteadyState_list = [model_daily_stats_V2["SteadyStateRPS6p"][d] for d in bin_days]
            EarlyMax_list = [model_daily_stats_V2["MaxRPS6pEarly"][d] for d in bin_days]
            peak_metric = [EarlyMax_list[i]/SteadyState_list[i] for i in range(len(SteadyState_list))]
    
            if len(peak_metric) > 0:
                peak_metric_avg = np.mean(peak_metric)
                peak_metric_std = np.std(peak_metric)
    
            plt.plot(bin, peak_metric_avg,color=cm.plasma(color_num),linestyle="",marker="o",markersize=10,markerfacecolor="none")

    plt.xlabel("Day Length (hours)")
    plt.ylabel("Peak Metric")    

### Section 5 (Figure 7) ###

def normalize_expr(values):
    '''
       Normalize an expression vector from 0 to 1 based on its min and max
    '''

    v_max = max(values)
    v_min = min(values)
    if v_max > v_min:
        norm_values = [ (v - v_min)/(v_max - v_min) for v in values]
    else: ### This would occur if the vector is flat, which it should not be for circdian genes
        norm_values = [ 0.5 for v in values]
    return norm_values

def PlotExpressionVectors_LD(selected_genes_file,all_genes_file,color,time_values):
    '''
       Plot expression vector of selected genes against a background of all genes
    '''

    # Read data from file
    select_gene_lines = [ln.strip() for ln in open(selected_genes_file,"r").readlines()]
    all_gene_lines = [ln.strip() for ln in open(all_genes_file,"r").readlines()]

    # Extract expression values from file
    select_gene_expr = [ln for ln in select_gene_lines if not ln.split("\t")[1].startswith('ct') and not ln.split("\t")[1].startswith('cos')]
    all_gene_expr = [ln for ln in all_gene_lines if not ln.split("\t")[1].startswith('ct') and not ln.split("\t")[1].startswith('cos')]

    ### SELECT GENES ###
    # Normalize and plot expression values
    norm_expr_vectors = []
    for ln in select_gene_expr[1:]:
        expr_values = [float(n) for n in ln.split("\t")[-1].split(",")]
        norm_expr = normalize_expr(expr_values)
        norm_expr_vectors.append(norm_expr)
        plt.plot(time_values,norm_expr,c=color,alpha=0.1, linewidth=3)
    
    # average across vectors in list
    avg_norm_vectors = []
    for i in range(len(norm_expr_vectors[0])):
        average_value = float(sum([norm_expr_vectors[n][i] for n in range(len(norm_expr_vectors))]))/float(len(norm_expr_vectors))
        avg_norm_vectors.append(average_value)
    plt.plot(time_values,avg_norm_vectors,c='black', linewidth=3)

    ### ALL GENES ###
    # Normal expression values
    all_norm_expr_vectors = []
    for ln in all_gene_expr[1:]:
        expr_values = [float(n) for n in ln.split("\t")[-1].split(",")]
        norm_expr = normalize_expr(expr_values)
        all_norm_expr_vectors.append(norm_expr)
    
    # average across vectors in list
    all_avg_norm_vectors = []
    for i in range(len(all_norm_expr_vectors[0])):
        average_value = float(sum([all_norm_expr_vectors[n][i] for n in range(len(all_norm_expr_vectors))]))/float(len(all_norm_expr_vectors))
        all_avg_norm_vectors.append(average_value)
    
    plt.plot(time_values,all_avg_norm_vectors,c='black', linewidth=3, linestyle=":")
    plt.xlabel("Time (hours)")
    plt.ylabel("Relative Expression")
    plt.xticks([0,6,12,18,24,30,36,42])
    plt.axvspan(16,24,alpha=0.5,color='grey')
    plt.axvspan(40,44,alpha=0.5,color='grey')
    plt.xlim(min(time_values),max(time_values))

def PlotExpressionVectors_LL(data_file,genes_file,all_genes_file,color,time_values):
    '''
       Plot expression vector of selected genes against a background of all genes
    '''

    # Read data file
    data_lines = [ln.strip() for ln in open(data_file,"r").readlines()]

    # Get genes
    gene_lines = [ln.strip() for ln in open(genes_file,"r").readlines()]
    genes = [ln.split("\t")[1] for ln in gene_lines[1:]]
    all_gene_lines = [ln.strip() for ln in open(all_genes_file,"r").readlines()]
    all_genes = [ln.split("\t")[1] for ln in all_gene_lines[1:]]

    # Filter data file by genes
    select_gene_expr = [ln for ln in data_lines if str.upper(ln.split("\t")[1]) in genes]
    all_gene_expr = [ln for ln in data_lines if str.upper(ln.split("\t")[1]) in all_genes]

    ### SELECT GENES ###
    # Normalize and plot expression values
    norm_expr_vectors = []
    for ln in select_gene_expr[1:]:
        expr_values = [float(n) for n in ln.split("\t")[-13:]]
        norm_expr = normalize_expr(expr_values)
        norm_expr_vectors.append(norm_expr)
        plt.plot(time_values,norm_expr,c=color,alpha=0.1, linewidth=3)
    
    # average across vectors in list
    avg_norm_vectors = []
    for i in range(len(norm_expr_vectors[0])):
        average_value = float(sum([norm_expr_vectors[n][i] for n in range(len(norm_expr_vectors))]))/float(len(norm_expr_vectors))
        avg_norm_vectors.append(average_value)
    plt.plot(time_values,avg_norm_vectors,c='black', linewidth=3)

    ### ALL GENES ###
    # Normal expression values
    all_norm_expr_vectors = []
    for ln in all_gene_expr[1:]:
        expr_values = [float(n) for n in ln.split("\t")[-13:]]
        norm_expr = normalize_expr(expr_values)
        all_norm_expr_vectors.append(norm_expr)
    
    # average across vectors in list
    all_avg_norm_vectors = []
    for i in range(len(all_norm_expr_vectors[0])):
        average_value = float(sum([all_norm_expr_vectors[n][i] for n in range(len(all_norm_expr_vectors))]))/float(len(all_norm_expr_vectors))
        all_avg_norm_vectors.append(average_value)
    
    plt.plot(time_values,all_avg_norm_vectors,c='black', linewidth=3, linestyle=":")
    plt.xlabel("Time (hours)")
    plt.ylabel("Relative Expression")
    plt.xticks([0,6,12,18,24,30,36,42])
    plt.axvspan(16,24,alpha=0.5,color='grey')
    plt.axvspan(40,44,alpha=0.5,color='grey')
    plt.xlim(min(time_values),max(time_values))

def PlotPhaseDifferences(select_diff_file,all_diff_file,color):
    '''
       Plot phase differences between selected all genes
    '''

    PhaseDiff = [float(ln.strip()) for ln in open(select_diff_file,"r").readlines() if not ln.strip()=="#N/A"]
    All_PhaseDiff = [float(ln.strip()) for ln in open(all_diff_file,"r").readlines() if not ln.strip()=="#N/A"]

    plt.hist(All_PhaseDiff,color='grey', bins = [-24, -18, -12, -6, 0, 6, 12, 18, 24], stacked=True,label="Background")
    plt.hist(PhaseDiff,color=color, bins = [-24, -18, -12, -6, 0, 6, 12, 18, 24], stacked=True,label="Day-to-Night",alpha=0.5)

    plt.xlabel("Phase Difference (hours)")
    plt.ylabel("Count")
    plt.xticks([-24,-18,-12,-6,0,6,12,18,24])
    plt.legend()

### MAIN ####
if __name__ == "__main__":
  run()
