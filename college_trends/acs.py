import pandas as pd

all_ages = pd.read_csv("all-ages.csv")
all_ages_totals  = all_ages.pivot_table(index="Major_category", aggfunc="sum")["Total"]
all_ages_major_categories = dict(all_ages_totals)
print all_ages_totals

print "\n****\n"

recent_grads = pd.read_csv("recent-grads.csv")
recent_totals = recent_grads.pivot_table(index="Major_category", aggfunc="sum")["Total"]
recent_grads_major_categories = dict(recent_totals)
print recent_totals


#Use the Low_wage_jobs and Total columns to calculate the proportion of recent college graduates that worked low wage jobs. Store the resulting Float object of the calculation as low_wage_percent.

recent_grads = pd.read_csv("recent-grads.csv")
low_wage_percent = 0.0

low_wage_sum = float(recent_grads["Low_wage_jobs"].sum())
recent_sum = float(recent_grads["Employed"].sum())


low_wage_percent = low_wage_sum / recent_sum


print low_wage_percent


#Both `all_ages` and `recent_grads` datasets have 173 rows, corresponding to the 173 college major codes. This enables us to do some comparisons between the two datasets and perform some initial calculations to see how similar or different the statistics of recent college graduates are from those of the entire population.


#Instructions 

#We want to know the number of majors where recent grads fare better than the overall population. For each major, determine if the Unemployment_rate is lower for `recent_grads` or for `all_ages` and increment either `recent_grads_lower_emp_count` or `all_ages_lower_emp_count` respectively.

# All majors, common to both DataFrames
majors = recent_grads['Major'].value_counts().index

recent_grads_lower_emp=[]
all_ages_lower_emp=[]

for major in majors:
    recent_unemply_rate = recent_grads[recent_grads["Major"]==major]["Unemployment_rate"].values[0]
    all_time_unemply_rate = all_ages[all_ages["Major"]==major]["Unemployment_rate"].values[0]
    diff = recent_unemply_rate - all_time_unemply_rate #comparator
    
    if diff < 0:
        recent_grads_lower_emp.append(major)
    elif diff >0:
        all_ages_lower_emp.append(major)
    else:
        pass #equal

recent_grads_lower_emp_count = len(recent_grads_lower_emp)
all_ages_lower_emp_count = len(all_ages_lower_emp)

print recent_grads_lower_emp_count
print all_ages_lower_emp_count

print recent_grads_lower_emp
print all_ages_lower_emp
        

