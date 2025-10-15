
"""
Queue based array position storage and adaptations for single peak finding
"""

# TODO: CONDITION: peaks exist before exponential increase
# peaks exist before exponential increase
def before_exponential_increase(position_vector, d1_data):
    last_position = d1_data.last().index()
    while last_position >= 0:
        last_position = last_position - 1
        if d1_data[last_position] < 0.2:
            for i in range(0, last_position+1):
                position_vector[i] = position_vector[i] + 1
            return  position_vector

# TODO: CONDITION: voltage range from 0.35 - 0.65
# sort out the data points that are in the voltage range
# peaks exist in between -0.35V and -0.65V range
def voltage_range(position_vector, list_x):
    for index, value in enumerate(list_x):
        if(value < -0.35 and value > -0.65):
            position_vector[index] = position_vector[index] + 1
    return position_vector


# TODO: CONDITION: derivative range between 0-0.5
def derivative_range(position_vector, d1_data):
    for index, value in enumerate(d1_data):
        if value < 1 and value > -1:
            position_vector[index] = position_vector[index] + 1
    return position_vector

# CONDITION: peaks need to exist after a rise, not a fall
def peak_after_rise(position_vector, d1_data):
    positive_flag = 0
    for index, value in enumerate(d1_data):
        if value > 0.5:
            positive_flag = 1
        if value < -0.5:
            positive_flag = 0
        if positive_flag == 1 and value > 0 and value < 0.2:
            position_vector[index] = position_vector[index] + 1
    return position_vector


# CONDITION: peaks after positive gradient
# after all functions have been found
def maximum_yvalue(position_vector, y_data):
    max_score = 0
    max_score_index = 0
    max_value = 0
    max_value_index = 0
    for index, value in enumerate(y_data):
        # within the highest score range
        if value > max_value:
            max_value = value
            max_index = index
    position_vector[max_index] = position_vector[max_index] + 1
    return max_index

# TODO: More conditions to be added