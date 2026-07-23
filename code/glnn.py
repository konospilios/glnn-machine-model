#######################################################################
# Initialization #

import jax
import jax.numpy as jnp
import numpy as np
import time
import pickle
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from functools import partial
import sys
import glob
import pickle
from tqdm import tqdm # Import tqdm for progress bars
import random


from jax.example_libraries import stax
from jax.example_libraries import optimizers

# visualization:
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from moviepy.editor import ImageSequenceClip
from functools import partial
import proglog
from scipy.integrate import cumulative_trapezoid
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from sklearn.preprocessing import StandardScaler


#######################################################################
# Initialize early functions #

def create_folder():
    # Check if the 'results' folder exists; if not, create it
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
        print(f"Folder '{parent_folder}' created successfully!")

    if Save_data == "yes":
        # Check if the target folder exists inside 'results'
        if os.path.exists(folder_path):
            print(f"Folder '{folder_path}' already exists. Exiting the script. Please rename the file_name , or load pickle data from 'Sprayer_Load_pickle'" )
            sys.exit() # Stop the script if the folder already exists
        else:
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully!")
    else:
        if os.path.exists(folder_path):
            print("Data not saved , but overwritten")
        else:
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully!")
    return parent_folder




def initialize_info(train_files):
    # Create a formatted string with all filenames
    filenames_text = "\n".join([f" - {file}" for file in train_files])

    # Construct the full text with the placeholder replaced by filenames
    text_data = f"""
        ############################### ITERATION {iteration_count}###########################################

        ##### GENERAL INFO #####
        Test name: {Test_name}

        Data used:
        {filenames_text}

        Start time = {start_time}
        End time = {end_time}
        Percentage_used_for training = {percentage_train}

        It is predicting on [{predict_on}] ( full data [Fu] / train data [Tr] / test_data [Te]).

        Number of layers for LNN = {number_of_layers_lnn}
        Number of neurons for LNN = {number_of_neurons_lnn}

        Number of layers dissipation = {number_of_layers_d}
        Number of neurons dissipation = {number_of_neurons_d}

        Is the data normalized?: {Normalize_data}
        Is the data shuffled?: {shuffle_data}

        Number of sensors used = {number_of_sensors}
        Number of cylinders used = {number_of_cylinders}

        Is gyroscope sensor used?: {pendulum_state}
        Is Pendulum cylinder used?: {pendulum_cylinder}

        Batch size used = {batch_size}
        Number of epochs = {num_epochs}

        Learning rates:
        First third = {first_third}
        Second third = {second_third}
        Third third = {third_third}

        Number of timesteps used for simulation = {number_of_timesteps}

        Tolerances used for simulation = {absolute_tolerance}


        ##################### LOG ######################
        """

    return text_data


# Write to the file
def write_data(text_data):
    with open(Info_text , 'a') as file:
        file.write(text_data)
        print(f"Data saved to {Info_text}")



def get_iteration_count(file_path):
    """Retrieve the current iteration count from the file."""
    if not os.path.exists(file_path):
        # Initialize the counter file if it doesn't exist
        with open(file_path , 'w') as file:
            file.write("0") # Start the counter at 0
        return 0
    else:
        # Read the current count from the file
        with open(file_path , 'r') as file:
            count = int(file.read().strip())
        return count


def increment_iteration_count(file_path):
    """Increment the iteration count and save it to the file."""
    count = get_iteration_count(file_path) + 1
    with open(file_path , 'w') as file:
        file.write(str(count))
    return count

def save_file_name(file_name , path):
    with open(path, 'w') as f:
        f.write(file_name)
#######################################################################
# User inputs #

# TO-DO: Set a name for the test: (No spaces!)
Test_name = "Test_name"

# TO-DO: Set a name for the folder that the test should end up in:
parent_folder = "Folder_name"

# TO-DO: Set folder for training data
train_files = glob.glob('Train_dataset_file.csv')

# TO-DO: Decide if the pendulum state and or the pendulum cylinder should be included:
pendulum_state = [] #("yes"/"no")
pendulum_cylinder = [] #("yes"/"no")


# TO-DO: Decide if the test should be saved: (If not, then it goes into the folder "garbage")
Save_data = [] #("yes"/"no")

# TO-DO: Decide of we want dissipation
inc_dissipation =[] #("yes"/"no")

#TO-DO: Decide if the data needs to be normalized
Normalize_data = [] #("yes"/"no")

# TO-DO: Decide if shuffle_data
shuffle_data = [] #("yes"/"no")

# TO-DO: Decide to make predictions on ( full data [Fu] / train data [Tr] / test_data [Te])
predict_on = [] #("Fu"/"Tr"/"Te")

# TO-DO: Decie on number of states and cylinders:
number_of_sensors = [] #(1/2/3)
number_of_cylinders = [] #(1/2/3)

# TO-DO: Decide on learning rates:
first_third = []
second_third = []
third_third = []

# TO-DO: Set the batch size:
batch_size = []

# TO-DO: Set the number of epochs:
num_epochs = []

# TO-DO: Define start and end times for the interval
start_time = []
end_time = []
percentage_train = []

# TO-DO: Choose how long we want to simulate the system 100 = 1 second
number_of_timesteps = []

# TO-DO: Decide on the tolerances for the solver
absolute_tolerance = []
relative_tolerance = []


# TO-DO: Build the LNN Network
number_of_layers_lnn = []
number_of_neurons_lnn = []

# TO-DO: Build the dissipation Network
number_of_layers_d = []
number_of_neurons_d = []

# TO-DO: Decide on dataset to make predictions on: (Should be 0 of there is only one dataset)
dataset_index = []


####### NOT USER INPUTS BUT NEED TO BE IN TH SAME CELL #######

if Save_data == "yes":
    print("Saving the data")
else:
    parent_folder = "Garbage"
    print("Data put in Garbage folder")

file_name = f"{Test_name}-Po{predict_on}_L{number_of_layers_lnn}x{number_of_neurons_lnn}-D{number_of_layers_d}x{number_of_neurons_d}-ND{Normalize_data}-SD{shuffle_data}-NS{number_of_sensors}-NC{number_of_cylinders}-G{pendulum_state}-P{pendulum_cylinder}-BS{batch_size}-NE{num_epochs}-LR_{first_third}_{second_third}_{third_third}_NT{number_of_timesteps}"

# Function to save the file_name to a text file



iteration_counter = f"{parent_folder}/{Test_name}/iteration_counter.txt"
# Define the path for the target folder inside the 'results' folder
folder_path = os.path.join(parent_folder , Test_name)
if not os.path.exists(folder_path):
    iteration_count = 0
    create_folder()

    #save_file_name(file_name , f"{parent_folder}/{file_name}/file_name.txt") # Save the file_name
    # SAVE "file_name" parameter
else:
    iteration_count = get_iteration_count(iteration_counter)
    iteration_count = increment_iteration_count(iteration_counter)
    #with open(f"{folder_path}/file_name.txt", 'r') as f:
    # file_name = f.read().strip()

pickle_file_name = f"{parent_folder}/{Test_name}/Pickle_it{iteration_count}_{file_name}.pkl"
full_data_img_file_name = f"{parent_folder}/{Test_name}/FullData_it{iteration_count}_{file_name}.png"
Predictions_data_img_file_name = f"{parent_folder}/{Test_name}/Simulations_target_it{iteration_count}_{file_name}.png"
loss_over_training_data = f"{parent_folder}/{Test_name}/LossOverTraining_it{iteration_count}_{file_name}.png"
Predictions = f"{parent_folder}/{Test_name}/Predictions_it{iteration_count}_{file_name}.png"

Info_text = f"{parent_folder}/{Test_name}/Info_and_log.txt"





ini_info = initialize_info(train_files)
write_data(ini_info)
#######################################################################
# More initializations #

if number_of_layers_lnn == 1:
    init_random_params_l , nn_forward_fn_l = stax.serial(
        stax.Dense(number_of_neurons_lnn),
        stax.Softplus ,
        stax.Dense(1),
        )
elif number_of_layers_lnn == 2:
    init_random_params_l , nn_forward_fn_l = stax.serial(
        stax.Dense(number_of_neurons_lnn),
        stax.Softplus ,
        stax.Dense(number_of_neurons_lnn),
        stax.Softplus ,
        stax.Dense(1),
        )
elif number_of_layers_lnn == 3:
    init_random_params_l , nn_forward_fn_l = stax.serial(
        stax.Dense(number_of_neurons_lnn),
        stax.Softplus ,
        stax.Dense(number_of_neurons_lnn),
        stax.Softplus ,
        stax.Dense(number_of_neurons_lnn),
        stax.Softplus ,
        stax.Dense(1),
        )
elif number_of_layers_lnn == 4:
    init_random_params_l , nn_forward_fn_l = stax.serial(
        stax.Dense(number_of_neurons_lnn),
        stax.Softplus ,
        stax.Dense(number_of_neurons_lnn),
        stax.Softplus ,
        stax.Dense(number_of_neurons_lnn),
        stax.Softplus ,
        stax.Dense(number_of_neurons_lnn),
        stax.Softplus ,
        stax.Dense(1),
        )
elif number_of_layers_lnn == 5:
    init_random_params_l , nn_forward_fn_l = stax.serial(
        stax.Dense(number_of_neurons_lnn),
        stax.Softplus ,
        stax.Dense(number_of_neurons_lnn),
        stax.Softplus ,
        stax.Dense(number_of_neurons_lnn),
        stax.Softplus ,
        stax.Dense(number_of_neurons_lnn),
        stax.Softplus ,
        stax.Dense(number_of_neurons_lnn),
        stax.Softplus ,
        stax.Dense(1),
        )

if number_of_layers_d == 1:
    init_random_params_d , nn_forward_d = stax.serial(
        stax.Dense(number_of_neurons_d),
        stax.Softplus ,
        stax.Dense(number_of_sensors),
        )
elif number_of_layers_d == 2:
    init_random_params_d , nn_forward_d = stax.serial(
        stax.Dense(number_of_neurons_d),
        stax.Softplus ,
        stax.Dense(number_of_neurons_d),
        stax.Softplus ,
        stax.Dense(number_of_sensors),
        )
elif number_of_layers_d == 3:
    init_random_params_d , nn_forward_d = stax.serial(
        stax.Dense(number_of_neurons_d),
        stax.Softplus ,
        stax.Dense(number_of_neurons_d),
        stax.Softplus ,
        stax.Dense(number_of_neurons_d),
        stax.Softplus ,
        stax.Dense(number_of_sensors),
        )
elif number_of_layers_d == 4:
    init_random_params_d , nn_forward_d = stax.serial(
        stax.Dense(number_of_neurons_d),
        stax.Softplus ,
        stax.Dense(number_of_neurons_d),
        stax.Softplus ,
        stax.Dense(number_of_neurons_d),
        stax.Softplus ,
        stax.Dense(number_of_neurons_d),
        stax.Softplus ,
        stax.Dense(number_of_sensors),
        )
elif number_of_layers_d == 5:
    init_random_params_d , nn_forward_d = stax.serial(
        stax.Dense(number_of_neurons_d),
        stax.Softplus ,
        stax.Dense(number_of_neurons_d),
        stax.Softplus ,
        stax.Dense(number_of_neurons_d),
        stax.Softplus ,
        stax.Dense(number_of_neurons_d),
        stax.Softplus ,
        stax.Dense(number_of_neurons_d),
        stax.Softplus ,
        stax.Dense(number_of_sensors),
        )



def normalize_data(data, epsilon=1e-8):
    """
        Normalizes data to the range [0, 1].
        Adds a small offset 'epsilon ' to avoid division by zero.
        Returns the normalized data and the parameters (min, max) used for normalization.
        """
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    # Avoid division by zero by adding a small offset
    range_ = data_max - data_min
    range_[range_ == 0] += epsilon
    normalized_data = (data - data_min) / range_
    return normalized_data , (data_min , data_max , epsilon)

def denormalize_data(normalized_data , params):
    """
        Denormalizes data from the range [0, 1] back to its original values.
        """
    data_min , data_max , epsilon = params
    range_ = data_max - data_min
    range_[range_ == 0] += epsilon
    return normalized_data * range_ + data_min

#######################################################################
# Import and process data #

# Create list of all the data:
train_data = [np.loadtxt(file, delimiter=',') for file in train_files] #Do not alter this.


# Concatenate all datasets into one
full_data = np.concatenate(train_data , axis=0)

# Initialize the network:


# Function to differentiate data
def central_difference(y, t):
    dy_dt = np.zeros_like(y)
    dt = t[1] - t[0] # Assuming uniform time step
    dy_dt[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
    dy_dt[0] = (y[1] - y[0]) / dt
    dy_dt[-1] = (y[-1] - y[-2]) / dt
    return dy_dt


# Import sensor data
FL_data = full_data[:, 0]
IL_data = full_data[:, 1]
IR_data = full_data[:, 2]
FR_data = full_data[:, 3]
gyroscope_data = full_data[:, 4]

# Import cylinder data
cylinder_left_data = full_data[:, 5]
cylinder_right_data = full_data[:, 6]
cylinder_pendulum_data = full_data[:, 7]

# Import time
t1 = full_data[:, 8]



# Calculate numerical derivatives
FL_deriv = central_difference(FL_data , t1)
IL_deriv = central_difference(IL_data , t1)
IR_deriv = central_difference(IR_data , t1)
FR_deriv = central_difference(FR_data , t1)

# Calculate the gyroscope position
gyroscope_int = cumulative_trapezoid(gyroscope_data , t1, initial=0)

# Calculate cylinder derivatives
cylinder_pendulum_deriv = central_difference(cylinder_pendulum_data , t1)
cylinder_left_deriv = central_difference(cylinder_left_data , t1)
cylinder_right_deriv = central_difference(cylinder_right_data , t1)

# One sensor:
if number_of_sensors == 1:
    print("One sensor")
    if pendulum_state == "yes":
        print("With gyroscope")
        state_vectors = np.column_stack([
            gyroscope_int ,
            gyroscope_data
            ])
    elif pendulum_state == "no":
        print("Without gyroscope")
        state_vectors = np.column_stack([
            FR_data ,
            FR_deriv ,
            ])
    else:
        print("Error in inputs: number_of_sensors need to be a number between 1 and 5, and pendulum_state needs to be either yes or no")
        print("Exiting system")
        sys.exit()


# Two sensors:
if number_of_sensors == 2:
    print("Two sensors")
    if pendulum_state == "yes":
        print("With Gyroscope")
        state_vectors = np.column_stack([
            FR_data ,
            gyroscope_int ,
            FR_deriv ,
            gyroscope_data
            ])
    elif pendulum_state == "no":
        print("Without gyroscope")
        state_vectors = np.column_stack([
            FR_data ,
            FL_data ,
            FR_deriv ,
            FL_deriv ,
            ])
    else:
        print("Error in inputs: number_of_sensors need to be a number between 1 and 5, and pendulum_state needs to be either yes or no")
        print("Exiting system")
        sys.exit()



# Three sensors:
if number_of_sensors == 3:
    print("Three sensors")
    if pendulum_state == "yes":
        print("With Gyroscope")
        state_vectors = np.column_stack([
            FR_data ,
            FL_data ,
            gyroscope_int ,
            FR_deriv ,
            FL_deriv ,
            gyroscope_data
            ])
    elif pendulum_state == "no":
        print("Without gyroscope")
        state_vectors = np.column_stack([
            FR_data ,
            IR_data ,
            FL_data ,
            FR_deriv ,
            IR_deriv ,
            FL_deriv ,
            ])
    else:
        print("Error in inputs: number_of_sensors need to be a number between 1 and 5, and pendulum_state needs to be either yes or no")
        print("Exiting system")
        sys.exit()


# Four sensors:
if number_of_sensors == 4:
    print("Three sensors")
    if pendulum_state == "yes":
        print("With Gyroscope")
        state_vectors = np.column_stack([
            FR_data ,
            IR_data ,
            FL_data ,
            gyroscope_int ,
            FR_deriv ,
            IR_deriv ,
            FL_deriv ,
            gyroscope_data
            ])
    elif pendulum_state == "no":
        print("Without gyroscope")
        state_vectors = np.column_stack([
            FR_data ,
            IR_data ,
            IL_data ,
            FL_data ,
            FR_deriv ,
            IR_deriv ,
            IL_deriv ,
            FL_deriv ,
            ])
    else:
        print("Error in inputs: number_of_sensors need to be a number between 1 and 5, and pendulum_state needs to be either yes or no")
        print("Exiting system")
        sys.exit()




# Five sensors:
if number_of_sensors == 5:
    print("Three sensors")
    if pendulum_state == "yes":
        print("With Gyroscope")
        state_vectors = np.column_stack([
            FR_data ,
            IR_data ,
            IL_data ,
            FL_data ,
            gyroscope_int ,
            FR_deriv ,
            IR_deriv ,
            IL_data ,
            FL_deriv ,
            gyroscope_data
            ])
    elif pendulum_state == "no":
        print("Not possible to have five sensors without the gyroscope ,will include it anyways.")
        state_vectors = np.column_stack([
            FR_data ,
            IR_data ,
            IL_data ,
            FL_data ,
            gyroscope_int ,
            FR_deriv ,
            IR_deriv ,
            IL_data ,
            FL_deriv ,
            gyroscope_data
            ])
    else:
        print("Error in inputs: number_of_sensors need to be a number between 1 and 5, and pendulum_state needs to be either yes or no")
        print("Exiting system")
        sys.exit()



# One cylinder
if number_of_cylinders == 1:
    if pendulum_cylinder == "yes":
        cylinder_vectors = np.column_stack([
            cylinder_pendulum_data
            ])
        cylinder_velocities = np.column_stack([
            cylinder_pendulum_deriv ,
            ])

    elif pendulum_cylinder == "no":
        cylinder_vectors = np.column_stack([
            cylinder_right_data ,
            ])
        cylinder_velocities = np.column_stack([
            cylinder_right_deriv ,
            ])
    else:
        print("Error in inputs: number_of_cylinders need to be a number between 1 and 5, and pendulum_state needs to be either yes or no")
        print("Exiting system")
        sys.exit()


# Two cylinders
if number_of_cylinders == 2:
    if pendulum_cylinder == "yes":
        cylinder_vectors = np.column_stack([
            cylinder_right_data ,
            cylinder_pendulum_data
            ])
        cylinder_velocities = np.column_stack([
            cylinder_right_data ,
            cylinder_pendulum_deriv
            ])

    elif pendulum_cylinder == "no":
        cylinder_vectors = np.column_stack([
            cylinder_right_data ,
            cylinder_left_data
            ])
        cylinder_velocities = np.column_stack([
            cylinder_right_deriv ,
            cylinder_left_deriv
            ])
    else:
        print("Error in inputs: number_of_cylinders need to be a number between 1 and 5, and pendulum_state needs to be either yes or no")
        print("Exiting system")
        sys.exit()





# Three cylinders:
if number_of_cylinders == 3:
    if pendulum_cylinder == "yes":
        cylinder_vectors = np.column_stack([
            cylinder_right_data ,
            cylinder_left_data ,
            cylinder_pendulum_data
            ])
        cylinder_velocities = np.column_stack([
            cylinder_right_data ,
            cylinder_left_data ,
            cylinder_pendulum_deriv
            ])

    elif pendulum_cylinder == "no":
        cylinder_vectors = np.column_stack([
            cylinder_right_data ,
            cylinder_left_data ,
            cylinder_pendulum_data
            ])
        cylinder_velocities = np.column_stack([
            cylinder_right_deriv ,
            cylinder_left_deriv ,
            cylinder_pendulum_deriv
            ])
    else:
        print("Error in inputs: number_of_cylinders need to be a number between 1 and 5, and pendulum_state needs to be either yes or no")
        print("Exiting system")
        sys.exit()



# Loop through each column to calculate the predicions:
state_derivatives = np.zeros_like(state_vectors) # Initialize array to hold derivatives
for i in range(state_vectors.shape[1]): # state_vectors.shape[1] gives the number of columns (variables)
    state_derivatives[:, i] = central_difference(state_vectors[:, i], t1)


x = state_vectors
xt = state_derivatives
x_cyl = cylinder_vectors
x_cyl_t = cylinder_velocities
# Time = t1

#######################################################################
# Initilize data and prepare for plot #

if Normalize_data == "yes":
    x, params_all_states = normalize_data(state_vectors)
    xt , params__targets = normalize_data(state_derivatives)
    x_cyl , params_cylinder_positions = normalize_data(cylinder_vectors)
    x_cyl_t , params_cylinder_velocities = normalize_data(cylinder_velocities)

def configure_data_plot():
    # Plotting the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))


    # One sensor:
    if number_of_sensors == 1:
        if pendulum_state == "yes":
            ax1.plot(t1, x[:, 0], label='Gyroscope')
        elif pendulum_state == "no":
            ax1.plot(t1, x[:, 0], label='FR sensor')


    # Two senors:
    if number_of_sensors == 2:
        if pendulum_state == "yes":
            ax1.plot(t1, x[:, 0], label='FR sensor')
            ax1.plot(t1, x[:, 1], label='Gyroscope')
        elif pendulum_state == "no":
            ax1.plot(t1, x[:, 0], label='FR sensor')
            ax1.plot(t1, x[:, 1], label='FL sensor')


    # Three sensors:
    if number_of_sensors == 3:
        if pendulum_state == "yes":
            ax1.plot(t1, x[:, 0], label='FR sensor')
            ax1.plot(t1, x[:, 1], label='FL sensor')
            ax1.plot(t1, x[:, 2], label='Gyroscope')
        elif pendulum_state == "no":
            ax1.plot(t1, x[:, 0], label='FR sensor')
            ax1.plot(t1, x[:, 1], label='IR sensor')
            ax1.plot(t1, x[:, 2], label='FL sensor')


    # Four sensors:
    if number_of_sensors == 4:
        if pendulum_state == "yes":
            ax1.plot(t1, x[:, 0], label='FR sensor')
            ax1.plot(t1, x[:, 1], label='IR sensor')
            ax1.plot(t1, x[:, 2], label='FL sensor')
            ax1.plot(t1, x[:, 3], label='Gyroscope')
        elif pendulum_state == "no":
            ax1.plot(t1, x[:, 0], label='FR sensor')
            ax1.plot(t1, x[:, 1], label='IR sensor')
            ax1.plot(t1, x[:, 2], label='IL sensor')
            ax1.plot(t1, x[:, 3], label='FL sensor')



    # Five sensors:
    if number_of_sensors == 5:
        if pendulum_state == "yes":
            ax1.plot(t1, x[:, 0], label='FR sensor')
            ax1.plot(t1, x[:, 1], label='IR sensor')
            ax1.plot(t1, x[:, 2], label='IL sensor')
            ax1.plot(t1, x[:, 3], label='FL sensor')
            ax1.plot(t1, x[:, 4], label='Gyroscope')
        elif pendulum_state == "no":
            ax1.plot(t1, x[:, 0], label='FR sensor')
            ax1.plot(t1, x[:, 1], label='IR sensor')
            ax1.plot(t1, x[:, 2], label='IL sensor')
            ax1.plot(t1, x[:, 3], label='FL sensor')
            ax1.plot(t1, x[:, 4], label='Gyroscope')



    ax1.set_title('Ultrasonic Sensor Data')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Sensor Reading')
    ax1.legend()
    ax1.grid(True)


    # One cylinder
    if number_of_cylinders == 1:
        if pendulum_cylinder == "yes":
            ax2.plot(t1, x_cyl[:, 0], label='Pendulum')
        elif pendulum_cylinder == "no":
            ax2.plot(t1, x_cyl[:, 0], label='Right')


    # Two cylinders:
    if number_of_cylinders == 2:
        if pendulum_cylinder == "yes":
            ax2.plot(t1, x_cyl[:, 0], label='Right')
            ax2.plot(t1, x_cyl[:, 1], label='Pendulum')
        elif pendulum_cylinder == "no":
            ax2.plot(t1, x_cyl[:, 0], label='Right')
            ax2.plot(t1, x_cyl[:, 1], label='Left')


        # Three cylinders:
        if number_of_cylinders == 3:
            ax2.plot(t1, x_cyl[:, 0], label='Right')
            ax2.plot(t1, x_cyl[:, 1], label='Left')
            ax2.plot(t1, x_cyl[:, 2], label='Pendulum')


        ax2.set_title('Cylinder positions')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Positions')
        ax2.legend()
        ax2.grid(True)


        # Adjust layout and show the plot
        plt.tight_layout()


        ## Plot

configure_data_plot()
plt.savefig(full_data_img_file_name , format="pdf")
plt.show()
#######################################################################
# More initializations , including cutting the datasets #

# Calculate train and test time intervals
train_time_start = start_time
train_time_end = (start_time + (end_time - start_time) * 0.01 *percentage_train)
test_time_start = train_time_end
test_time_end = end_time



# Initialize empty lists to store data from each dataset
x_sets_cutoff = []
xt_sets_cutoff = []
x_cyl_sets_cutoff = []
x_cyl_t_sets_cutoff = []

x_sets_train = []
xt_sets_train = []
x_cyl_sets_train = []
x_cyl_t_sets_train = []

x_sets_test = []
xt_sets_test = []
x_cyl_sets_test = []
x_cyl_t_sets_test = []

time_sets = []


# Iterate over each dataset in train_data
for file_index , data in enumerate(train_data):
    # Calculate start and end indices for each dataset within the combined arrays
    start_index = sum(len(d) for d in train_data[:file_index])
    end_index = start_index + len(data)

    # Extract data for the specific dataset
    x_single = x[start_index:end_index]
    xt_single = xt[start_index:end_index]
    x_cyl_single = x_cyl[start_index:end_index]
    x_cyl_t_single = x_cyl_t[start_index:end_index]
    time_single = t1[start_index:end_index]

    # Find indices for start , end, train , and test times
    start_index = np.where(t1 >= start_time)[0][0] # First index where time exceeds start_time
    end_index = np.where(t1 <= end_time)[0][-1] # Last index where time is less than or equal to end_time
    start_index_train = np.where(t1 >= train_time_start)[0][0]
    end_index_train = np.where(t1 <= train_time_end)[0][-1]
    start_index_test = np.where(t1 >= test_time_start)[0][0]
    end_index_test = np.where(t1 <= test_time_end)[0][-1]


    # Adjust time to start from 0
    time_adjusted = t1[start_index:end_index + 1] - t1[start_index]

    # Cut the data
    x_cutoff = x_single[start_index:end_index + 1] # Include end index
    xt_cutoff = xt_single[start_index:end_index + 1]
    x_cyl_cutoff = x_cyl_single[start_index:end_index + 1]
    x_cyl_t_cutoff = x_cyl_t_single[start_index:end_index + 1]

    x_train = x_single[start_index_train:end_index_train + 1]
    xt_train = xt_single[start_index_train:end_index_train + 1]
    x_cyl_train = x_cyl_single[start_index_train:end_index_train + 1]
    x_cyl_t_train = x_cyl_t_single[start_index_train:end_index_train +1]

    x_test = x_single[start_index_test:end_index_test + 1]
    xt_test = xt_single[start_index_test:end_index_test + 1]
    x_cyl_test = x_cyl_single[start_index_test:end_index_test + 1]
    x_cyl_t_test = x_cyl_t_single[start_index_test:end_index_test + 1]

    # Append each extracted and trimmed dataset to the corresponding list
    x_sets_cutoff.append(x_cutoff)
    xt_sets_cutoff.append(xt_cutoff)
    x_cyl_sets_cutoff.append(x_cyl_cutoff)
    x_cyl_t_sets_cutoff.append(x_cyl_t_cutoff)

    x_sets_train.append(x_train)
    xt_sets_train.append(xt_train)
    x_cyl_sets_train.append(x_cyl_train)
    x_cyl_t_sets_train.append(x_cyl_t_train)

    x_sets_test.append(x_test)
    xt_sets_test.append(xt_test)
    x_cyl_sets_test.append(x_cyl_test)
    x_cyl_t_sets_test.append(x_cyl_t_test)

    time_sets.append(time_adjusted)



# Concatenate the datasets Full
x_combined_cutoff = np.concatenate(x_sets_cutoff , axis=0) # Combine all x_sets (shape: sum of all x_single shapes)
xt_combined_cutoff = np.concatenate(xt_sets_cutoff , axis=0) # Combine all xt_sets
x_cyl_combined_cutoff = np.concatenate(x_cyl_sets_cutoff , axis=0) #Combine all x_cyl_sets
x_cyl_t_combined_cutoff = np.concatenate(x_cyl_t_sets_cutoff , axis=0) #Combine all x_cyl_sets


# Concatenate the datasets Train
x_combined_train = np.concatenate(x_sets_train , axis=0) # Combine all x_sets (shape: sum of all x_single shapes)
xt_combined_train = np.concatenate(xt_sets_train , axis=0) # Combine all xt_sets
x_cyl_combined_train = np.concatenate(x_cyl_sets_train , axis=0) #Combine all x_cyl_sets
x_cyl_t_combined_train = np.concatenate(x_cyl_t_sets_train , axis=0) #Combine all x_cyl_sets

# Concatenate the datasets Test
x_combined_test = np.concatenate(x_sets_test , axis=0) # Combine all x_sets (shape: sum of all x_single shapes)
xt_combined_test = np.concatenate(xt_sets_test , axis=0) # Combine all xt_sets
x_cyl_combined_test = np.concatenate(x_cyl_sets_test , axis=0) # Combine all x_cyl_sets
x_cyl_t_combined_test = np.concatenate(x_cyl_t_sets_test , axis=0) #Combine all x_cyl_sets

time_combined = np.concatenate(time_sets , axis=0)




# Shuffle the combined dataset
if shuffle_data == "yes":
    indices = np.random.permutation(x_combined_cutoff.shape[0]) #Generate shuffled indices
    x_combined_cutoff = x_combined_cutoff[indices] # Shuffle x_combined
    xt_combined_cutoff = xt_combined_cutoff[indices] # Shuffle xt_combined
    x_cyl_combined_cutoff = x_cyl_combined_cutoff[indices] # Shuffle x_cyl_combined
    x_cyl_t_combined_cutoff = x_cyl_t_combined_cutoff[indices] #Shuffle x_cyl_combined

    indices = np.random.permutation(x_combined_train.shape[0]) #Generate shuffled indices
    x_combined_train = x_combined_train[indices] # Shuffle x_combined
    xt_combined_train = xt_combined_train[indices] # Shuffle xt_combined
    x_cyl_combined_train = x_cyl_combined_train[indices] # Shuffle x_cyl_combined
    x_cyl_t_combined_train = x_cyl_t_combined_train[indices] # Shuffle x_cyl_combined

    indices = np.random.permutation(x_combined_test.shape[0]) #Generate shuffled indices
    x_combined_test = x_combined_test[indices] # Shuffle x_combined
    xt_combined_test = xt_combined_test[indices] # Shuffle xt_combined
    x_cyl_combined_test = x_cyl_combined_test[indices] # Shuffle x_cyl_combined
    x_cyl_t_combined_test = x_cyl_t_combined_test[indices] # Shuffle x_cyl_combined
    print("Data shuffled")

# Calculate the number of batches
num_batches = x_combined_train.shape[0] // batch_size # Total number of batches

#######################################################################
# Intilize GLNN critical functions #

# Define the Lagrangian model to include external inputs (cylinder positions)
def learned_lagrangian(params):
    def lagrangian(q, q_t, cyl_data):
        cyl_data = jnp.atleast_1d(cyl_data) # Ensure cyl_data is at least 1D
        state = jnp.concatenate([q, q_t, cyl_data]) # Concatenate q,q_t, and cylinder data
        print("State shape (for debugging):", state.shape) # Debugging line to check state shape
        return jnp.squeeze(nn_forward_fn(params , state), axis=-1)
    return lagrangian



if inc_dissipation == "yes":
    def equation_of_motion(lagrangian , dissipation , state , cyl_data ,cyl_velocities , t=None):
        # Split the state into positions (q) and velocities (q_t)
        print("EoM =", state.shape)
        q, q_t = jnp.split(state , 2)

        # Calculate the inverse of the Hessian of the Lagrangian with respect to velocities
        hessian_inv = jnp.linalg.pinv(jax.hessian(lambda q, q_t:lagrangian(q, q_t, cyl_data), argnums=1)(q, q_t))

        # Gradient of the Lagrangian with respect to positions
        grad_q = jax.grad(lambda q, q_t: lagrangian(q, q_t, cyl_data),argnums=0)(q, q_t)

        # Mixed partial derivative (Jacobian) of the Lagrangian with respect to q and q_t
        jacobian_qt_q = jax.jacobian(lambda q, q_t: jax.grad(lagrangian ,argnums=1)(q, q_t, cyl_data), argnums=0)(q, q_t)

        # Calculate the dissipation , possibly using q_t and cyl_data if relevant
        Fnc = dissipation(q, cyl_data , q_t,cyl_velocities)

        # Calculate accelerations (q_tt) with the Lagrange and dissipation terms
        q_tt = hessian_inv @ ((grad_q - jacobian_qt_q @ q_t) - Fnc)

        # Return the concatenated state derivative: [velocities ,accelerations]
        return jnp.concatenate([q_t, q_tt])

if inc_dissipation == "no":
    def equation_of_motion(lagrangian , dissipation , state , cyl_data ,cyl_velocities , t=None):
        # Split the state into positions (q) and velocities (q_t)
        print("EoM =", state.shape)
        q, q_t = jnp.split(state , 2)

        # Calculate the inverse of the Hessian of the Lagrangian with respect to velocities
        hessian_inv = jnp.linalg.pinv(jax.hessian(lambda q, q_t:lagrangian(q, q_t, cyl_data), argnums=1)(q, q_t))

        # Gradient of the Lagrangian with respect to positions
        grad_q = jax.grad(lambda q, q_t: lagrangian(q, q_t, cyl_data),argnums=0)(q, q_t)

        # Mixed partial derivative (Jacobian) of the Lagrangian with respect to q and q_t
        jacobian_qt_q = jax.jacobian(lambda q, q_t: jax.grad(lagrangian ,argnums=1)(q, q_t, cyl_data), argnums=0)(q, q_t)

        # Calculate the dissipation , possibly using q_t and cyl_data if relevant
        Fnc = dissipation(q, cyl_data , q_t,cyl_velocities)
        Fnc = 0

        # Calculate accelerations (q_tt) with the Lagrange and dissipation terms
        q_tt = (hessian_inv @ (grad_q - jacobian_qt_q @ q_t)) - Fnc

        # Return the concatenated state derivative: [velocities ,accelerations]
        return jnp.concatenate([q_t, q_tt])



def update_derivative(i, opt_state , batch , cyl_data , cyl_velocities):
    # Retrieve the current parameters
    prams = get_params(opt_state)
    params_lnn = get_params(opt_state)[0]
    params_d = get_params(opt_state)[1]

    # Compute gradients for both sets of parameters separately
    grads_lnn = jax.grad(loss, argnums=0)(params_lnn , params_d , batch ,cyl_data , cyl_velocities)
    grads_d = jax.grad(loss, argnums=1)(params_lnn , params_d , batch ,cyl_data , cyl_velocities)

    # Create a tuple of gradients in the same structure as the parameters
    grads = (grads_lnn , grads_d)

    # Update the optimizer state
    return opt_update(i, grads , opt_state)

@jax.jit
def loss(params_lnn , params_d , batch , cyl_data , cyl_velocities):
    state , targets = batch
    preds = jax.vmap(lambda s, c, v: equation_of_motion(learned_lagrangian(params_lnn), learned_dissipation(params_d), s,c, v))(state , cyl_data , cyl_velocities)
    return jnp.mean((preds - targets) ** 2)


def solve_lagrangian(lagrangian , dissipation , initial_state , cyl_data ,cyl_velocities , t, rtol, atol):
    # Function to interpolate 'cyl_data ' for each time step
    def interp_cyl_data(time_idx):
        return jnp.array([jnp.interp(time_idx , jnp.arange(len(cyl_data)), cyl_data[:, i]) for i in range(cyl_data.shape[1])])
    def interp_cyl_velocities(time_idx):
        return jnp.array([jnp.interp(time_idx , jnp.arange(len(cyl_velocities)), cyl_velocities[:, i]) for i in range(cyl_velocities.shape[1])])
        # Define the ODE function that uses interpolated 'cyl_data '
        def ode_func(state , time_idx):
            current_cyl_data = interp_cyl_data(time_idx)
            current_cyl_velocities = interp_cyl_velocities(time_idx)
            print(f"Time step {time_idx}, State: {state}, Cylinder data: {current_cyl_data}") # Diagnostic print
            return equation_of_motion(lagrangian , dissipation , state ,current_cyl_data , current_cyl_velocities)

        # Use 'odeint ' over the full time range
        return odeint(ode_func , initial_state , t, rtol=relative_tolerance ,atol=absolute_tolerance)



# Define the Lagrangian function based on learned parameters
def learned_lagrangian(params):
    def lagrangian(q, q_t,cyl_data):
        cyl_data = jnp.atleast_1d(cyl_data) # ensure cyl_Data is at least 1D
        if cyl_data.ndim < q.ndim:
            cyl_data = jnp.expand_dims(cyl_data ,axis=0)
        state = jnp.concatenate([q, q_t, cyl_data],axis=-1)
        return jnp.squeeze(nn_forward_fn_l(params , state), axis=-1)
    return lagrangian

# Update the learned_dissipation function to take inputs properly
def learned_dissipation(params):
    # Ensure you pass the input data (state) to the neural network as well
    def dissipation(q ,cyl_data ,q_t, cyl_velocities):
        state = jnp.concatenate([q ,cyl_data ,q_t, cyl_velocities])
        return jnp.squeeze(nn_forward_d(params , state))
    return dissipation



# Adam optimizer with learning rate decay
opt_init , opt_update , get_params = optimizers.adam(
    lambda t: jnp.select([t < batch_size * (num_batches // 3),
    t < batch_size * (2 * num_batches // 3),
    t >= batch_size * (2 * num_batches // 3)],
    [first_third , second_third , third_third]))

rng = jax.random.PRNGKey(0)

#######################################################################
# Information for debugging #

print("x_combined_cutoff =",x_combined_cutoff.shape)
print("xt_combined_cutoff =", xt_combined_cutoff.shape)
print("x_cyl_combined_cutoff =", x_cyl_combined_cutoff.shape)
print("x_cyl_t_combined_cutoff =", x_cyl_t_combined_cutoff.shape)

print("x_combined_train =",x_combined_train.shape)
print("xt_combined_train =", xt_combined_train.shape)
print("x_cyl_combined_train =", x_cyl_combined_train.shape)
print("x_cyl_t_combined_train =", x_cyl_t_combined_train.shape)

print("x_combined_test =",x_combined_test.shape)
print("xt_combined_test =", xt_combined_test.shape)
print("x_cyl_combined_test =", x_cyl_combined_test.shape)
print("x_cyl_t_combined_test =", x_cyl_t_combined_test.shape)

# Input size:
input_size_l = x_combined_cutoff.shape[1] + x_cyl_combined_cutoff.shape[1]
input_size_f = x_combined_cutoff.shape[1] + x_cyl_combined_cutoff.shape[1] + x_cyl_t_combined_cutoff.shape[1]

_, init_params_l = init_random_params_l(rng, (-1, input_size_l))
_, init_params_f = init_random_params_f(rng, (-1, input_size_f))

# Combine both sets of parameters into a single tuple
init_params = (init_params_l , init_params_f)
opt_state = opt_init(init_params)

# Initialize the optimizer state for combined parameters
opt_state = opt_init(init_params)
params = get_params(opt_state)
params_l = get_params(opt_state)[0]
params_f = get_params(opt_state)[1]

# Ensure the losses are global
train_losses = []
test_losses = []


print("input_size_l =", input_size_l)
print("input_size_fnc =", input_size_f)

#######################################################################
# Training loop #

write_data("\n Training started")
start_wall_time = time.time()
start_cpu_time = time.process_time()
for epoch in range(num_epochs): # Start from the next epoch
    print(f"Epoch {epoch + 1}/{num_epochs} starting...")

    for batch_idx in tqdm(range(num_batches)):
        # Training logic remains unchanged...
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        x_batch_train = x_combined_train[start_idx:end_idx]
        xt_batch_train = xt_combined_train[start_idx:end_idx]
        x_cyl_batch_train = x_cyl_combined_train[start_idx:end_idx]
        x_cyl_t_batch_train = x_cyl_t_combined_train[start_idx:end_idx]

        x_batch_test = x_combined_test[start_idx:end_idx]
        xt_batch_test = xt_combined_test[start_idx:end_idx]
        x_cyl_batch_test = x_cyl_combined_test[start_idx:end_idx]
        x_cyl_t_batch_test = x_cyl_t_combined_test[start_idx:end_idx]

        params_l = get_params(opt_state)[0]
        params_f = get_params(opt_state)[1]

        train_loss = loss(params_l , params_f , (x_batch_train ,xt_batch_train), x_cyl_batch_train , x_cyl_t_batch_train)
        test_loss = loss(params_l , params_f , (x_batch_test ,xt_batch_test), x_cyl_batch_test , x_cyl_t_batch_test)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if batch_idx % 1000 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{num_batches}, Training Loss: {train_loss:.6f}, Testing loss: {test_loss:.6f}")

        opt_state = update_derivative(batch_idx , opt_state , (x_batch_train , xt_batch_train), x_cyl_batch_train ,x_cyl_t_batch_train)

print(f"Epoch {epoch + 1} complete. Training loss: {train_loss:.6f},Testing loss: {test_loss:.6f}")


end_wall_time = time.time()
end_cpu_time = time.process_time()
# Get the final model parameters after all epochs
params = get_params(opt_state)
params_l = get_params(opt_state)[0]
params_f = get_params(opt_state)[1]
end_wall_time = time.time()
end_cpu_time = time.process_time()

wall_time = end_wall_time - start_wall_time
cpu_time = end_cpu_time - start_cpu_time

print("Training completed!")
write_data("\n Training completed!")
write_data(f"\n Wall time taken = {wall_time}")
write_data(f"\n CPU time taken = {cpu_time}")

# Save the data in the 'pickle' folder
with open(pickle_file_name , "wb") as f:
    pickle.dump({'params': params , 'train_losses': train_losses , 'test_losses': test_losses}, f)
print(f"Model parameters and losses saved in '{pickle_file_name}'!")

#######################################################################
# Prepare the losses over training plot #

# Plot and save trainnig process

# Check the lengths of train_losses and test_losses
print(f"Length of train_losses: {len(train_losses)}")
def configure_loss_plot():
    # Now proceed with the plot
    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(train_losses , label='Train loss')
    plt.plot(test_losses , label='Test loss')
    plt.yscale('log')
    plt.ylim(None, 200)
    plt.title('Losses over training')
    plt.xlabel("Train step")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.grid()

# Annotate the plot with time information
#time_info = f"Wall time: {wall_time_taken:.2f}s\nCPU time: {cpu_time_taken:.2f}s"
#plt.text(0.5, 0.9, time_info , ha='center', va='center', transform=plt.gca().transAxes , fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

#######################################################################
# Plot the losses over training #

# Save the plot and show it
write_data("\n Losses over trainig plotted")
configure_loss_plot()
plt.savefig(loss_over_training_data , format="png")
plt.show()


#######################################################################
# Prepare plot predictions #

## Prepare to make predictions
def configure_pred_plot():
    # Extract data for one specific dataset from the segmented lists
    if predict_on == "Fu":
        x_single_test = x_sets_cutoff[dataset_index]
        xt_single_test = xt_sets_cutoff[dataset_index]
        x_cyl_single_test = x_cyl_sets_cutoff[dataset_index]
        x_cyl_t_single_test = x_cyl_t_sets_cutoff[dataset_index]
    elif predict_on == "Tr":
        x_single_test = x_sets_train[dataset_index]
        xt_single_test = xt_sets_train[dataset_index]
        x_cyl_single_test = x_cyl_sets_train[dataset_index]
        x_cyl_t_single_test = x_cyl_t_sets_train[dataset_index]
    elif predict_on == "Te":
        x_single_test = x_sets_test[dataset_index]
        xt_single_test = x_sets_test[dataset_index]
        x_cyl_single_test = x_cyl_sets_test[dataset_index]
        x_cyl_t_single_test = x_cyl_t_sets_test[dataset_index]

    time_single_test = time_sets[dataset_index]

    # Initial state and cylinder data for simulation
    x1 = x_single_test[0] # Initial state from the test set
    cyl_data_for_sim = x_cyl_single_test[:number_of_timesteps]
    cyl_t_data_for_sim = x_cyl_t_single_test[:number_of_timesteps]

    # Define the time values for simulation
    t2 = time_single_test[:number_of_timesteps]

    import jax.numpy as jnp
    from jax.experimental.ode import odeint

    # Define the interpolation function for each cylinder data dimension
    def jax_linear_interp(t, t_vals , y_vals):
        """Perform JAX-compatible linear interpolation for each dimension of y_vals."""
        interpolated_vals = []
        for i in range(y_vals.shape[1]):
            interpolated_vals.append(jnp.interp(t, t_vals , y_vals[:, i]))
        return jnp.stack(interpolated_vals , axis=-1)

    # Define the ODE function with interpolated cylinder inputs
    def ode_func_with_cyl_input(state , t):
        cyl_data = jax_linear_interp(t, t2, cyl_data_for_sim)
        cyl_velocities = jax_linear_interp(t, t2, cyl_t_data_for_sim)
        return equation_of_motion(learned_lagrangian(params_l),learned_dissipation(params_f), state , cyl_data ,cyl_velocities)

    # Solve the Lagrangian using odeint
    x1_model = jax.device_get(odeint(ode_func_with_cyl_input , x1, t2,rtol=relative_tolerance , atol=absolute_tolerance))

    print("Simulation completed successfully.")
    print("Simulated time:", t2[-1])
    print("Model output shape:", x1_model.shape)

    # Define labels and colors dynamically based on the number of sensors
    labels = []
    colors = []
    if number_of_sensors >= 1:
        labels.append("FR Distance")
        colors.append("green")
    if number_of_sensors >= 2:
        labels.append("FL Distance")
        colors.append("blue")
    if number_of_sensors >= 3:
        labels.append("Pendulum Position")
        colors.append("red")

    # Calculate errors for reference
    mse_states = jnp.square(x1_model - x_single_test[:number_of_timesteps]).mean(axis=0)

    # First Graph: Predicted vs. Actual States
    plt.figure(figsize=(8.4, 5.6))
    for i in range(number_of_sensors):
        plt.plot(t2, x1_model[:, i], label=f'Predicted {labels[i]}',color=colors[i], linestyle='-')
        plt.plot(t2, x_single_test[:number_of_timesteps , i], label=f'Actual {labels[i]}', color=colors[i], linestyle='--')

    # Place error box inside the graph at the top-right corner
    error_text = "\n".join([f"Error (MSE) {labels[i]} = {mse_states[i]:.4f}" for i in range(number_of_sensors)])
    plt.text(0.98, 0.98, error_text , fontsize=9, color="black",
        bbox=dict(facecolor='white', alpha=0.5), ha='right', va='top', transform=plt.gca().transAxes)

    # Place the legend box in the bottom -right corner
    plt.legend(loc="lower right", facecolor='white', edgecolor='black')

    plt.title('Comparison of Predicted and Actual States')
    plt.xlabel('Time (s)')
    plt.ylabel('State Values')
    plt.grid()
    plt.savefig(f"{parent_folder}/{Test_name}/Predictions_it{iteration_count}_{file_name}.pdf", bbox_inches="tight") # Save as PDF
    plt.show()

    # Second Graphs: Predicted vs. Actual Accelerations with Drift Bias as Position Drift
    mse_accelerations = []
    for i in range(number_of_sensors):
        plt.figure(figsize=(8.4, 5.6))
        acceleration_model = jnp.gradient(x1_model[:, i], t2)
        acceleration_actual = jnp.gradient(x_single_test[:number_of_timesteps , i], t2)
        position_drift = x1_model[:, i] - x_single_test[:number_of_timesteps , i]
        mse_acceleration = jnp.square(acceleration_model -acceleration_actual).mean()
        mse_accelerations.append(mse_acceleration)
        bias = jnp.abs(position_drift).mean()

        plt.plot(t2, acceleration_model , label=f'Predicted Acceleration{labels[i]}', color=colors[i], linestyle='-')
        plt.plot(t2, acceleration_actual , label=f'Actual Acceleration {labels[i]}', color=colors[i], linestyle='--')
        plt.plot(t2, position_drift , label=f'Position Drift (Bias) {labels[i]}', color=colors[i], linestyle=':')

        # Place combined bias and error box inside the graph at the top-right corner
        combined_text = f"Bias (MAD) = {bias:.4f}\nError (MSE) = {mse_acceleration:.4f}"
        plt.text(0.98, 0.98, combined_text , fontsize=9, color="black",
            bbox=dict(facecolor='white', alpha=0.5), ha='right', va='top', transform=plt.gca().transAxes)

        # Place the legend in the bottom -right corner
        plt.legend(loc="lower right", facecolor='white', edgecolor='black')

        plt.title(f'Predicted vs. Actual Accelerations with Position Drift as Bias for {labels[i]}')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration')
        plt.grid()
        plt.savefig(f"{parent_folder}/{Test_name}/Acceleration_{labels[i]}_it{iteration_count}_{file_name}.pdf", bbox_inches="tight")# Save as PDF
        plt.show()

#######################################################################
# Plot predictions #

write_data("\n Starting making predictions")

try:
    # Call the prediction function (no need to pass number_of_states; it uses number_of_sensors)
    configure_pred_plot() # The function dynamically adjusts based on the number_of_sensors variable

    write_data("\n Simulation and plotting successful")

except Exception as e:
    # Log errors and provide useful debugging information
    write_data(f"\nError during simulation: {str(e)}")
    print(f"Simulation failed: {str(e)}")
