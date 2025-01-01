import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define the fuzzy logic controller
def setup_fuzzy_controller():
    # Define fuzzy variables
    error = ctrl.Antecedent(np.arange(-10, 11, 0.1), 'error')
    derivative = ctrl.Antecedent(np.arange(-10, 11, 0.1), 'derivative')
    control_signal = ctrl.Consequent(np.arange(-10, 11, 0.1), 'control_signal')
    
    # Define membership functions for error and derivative
    error['NB'] = fuzz.trimf(error.universe, [-1, -1, -0.5])   
    error['NS'] = fuzz.trimf(error.universe, [-1, -0.25, 0])    
    error['Z'] = fuzz.trimf(error.universe, [-0.15, 0, 0.15])  
    error['PS'] = fuzz.trimf(error.universe, [0, 0.25, 1])      
    error['PB'] = fuzz.trimf(error.universe, [0.5, 1, 1]) 
        
    derivative['NB'] = fuzz.trimf(derivative.universe, [-25, -25, -10])   
    derivative['NS'] = fuzz.trimf(derivative.universe, [-15, -5, 0])      
    derivative['Z'] = fuzz.trimf(derivative.universe, [-1.5, 0, 1.5])        
    derivative['PS'] = fuzz.trimf(derivative.universe, [0, 5, 15])        
    derivative['PB'] = fuzz.trimf(derivative.universe, [10, 25, 25])
    
    # Define membership functions for control signal
    control_signal['NB'] = fuzz.trimf(control_signal.universe, [-10, -10, 0])
    control_signal['NS'] = fuzz.trimf(control_signal.universe, [-10, 0, 10])
    control_signal['Z'] = fuzz.trimf(control_signal.universe, [-5, 0, 5])
    control_signal['PS'] = fuzz.trimf(control_signal.universe, [0, 10, 10])
    control_signal['PB'] = fuzz.trimf(control_signal.universe, [0, 10, 10])
    
    # Define fuzzy rules
    rule1 = ctrl.Rule(error['NB'] & derivative['NB'], control_signal['NB'])
    rule2 = ctrl.Rule(error['NB'] & derivative['NS'], control_signal['NB'])
    rule3 = ctrl.Rule(error['NB'] & derivative['Z'], control_signal['NS'])
    rule4 = ctrl.Rule(error['NB'] & derivative['PS'], control_signal['Z'])
    rule5 = ctrl.Rule(error['NB'] & derivative['PB'], control_signal['Z'])
    rule6 = ctrl.Rule(error['NS'] & derivative['NB'], control_signal['NB'])
    rule7 = ctrl.Rule(error['NS'] & derivative['NS'], control_signal['NB'])
    rule8 = ctrl.Rule(error['NS'] & derivative['Z'], control_signal['NS'])
    rule9 = ctrl.Rule(error['NS'] & derivative['PS'], control_signal['Z'])
    rule10 = ctrl.Rule(error['NS'] & derivative['PB'], control_signal['Z'])
    rule11 = ctrl.Rule(error['Z'] & derivative['NB'], control_signal['NS'])
    rule12 = ctrl.Rule(error['Z'] & derivative['NS'], control_signal['NS'])
    rule13 = ctrl.Rule(error['Z'] & derivative['Z'], control_signal['Z'])
    rule14 = ctrl.Rule(error['Z'] & derivative['PS'], control_signal['PS'])
    rule15 = ctrl.Rule(error['Z'] & derivative['PB'], control_signal['PS'])
    rule16 = ctrl.Rule(error['PS'] & derivative['NB'], control_signal['Z'])
    rule17 = ctrl.Rule(error['PS'] & derivative['NS'], control_signal['Z'])
    rule18 = ctrl.Rule(error['PS'] & derivative['Z'], control_signal['PS'])
    rule19 = ctrl.Rule(error['PS'] & derivative['PS'], control_signal['PB'])
    rule20 = ctrl.Rule(error['PS'] & derivative['PB'], control_signal['PB'])
    rule21 = ctrl.Rule(error['PB'] & derivative['NB'], control_signal['Z'])
    rule22 = ctrl.Rule(error['PB'] & derivative['NS'], control_signal['Z'])
    rule23 = ctrl.Rule(error['PB'] & derivative['Z'], control_signal['PS'])
    rule24 = ctrl.Rule(error['PB'] & derivative['PS'], control_signal['PB'])
    rule25 = ctrl.Rule(error['PB'] & derivative['PB'], control_signal['PB'])
    
    # Create the control system
    control_system = ctrl.ControlSystem([
        rule1, rule2, rule3, rule4, rule5,
        rule6, rule7, rule8, rule9, rule10,
        rule11, rule12, rule13, rule14, rule15,
        rule16, rule17, rule18, rule19, rule20,
        rule21, rule22, rule23, rule24, rule25])
    fuzzy_controller = ctrl.ControlSystemSimulation(control_system)
    return fuzzy_controller

# Fuzzy control simulation
def fuzzy_control(fuzzy_controller, error, derivative):
    print(error)
    print(derivative)
    fuzzy_controller.input['error'] = error
    fuzzy_controller.input['derivative'] = derivative
    fuzzy_controller.compute()
    return fuzzy_controller.output['control_signal']

# Simulation parameters
K = 1.0           # Gain
tau = 10.0        # Time constant
tau_DT = 1.0      # Dead time
dt = 0.1          # Simulation time step
time_end = 50     # End time for the simulation
t = np.arange(0, time_end, dt)

# Set-point for the system to follow
setpoint = np.ones_like(t)

# Discretized FOPTD process model with fuzzy control
def simulate_system_fuzzy(K, tau, tau_DT, t, setpoint):
    fuzzy_controller = setup_fuzzy_controller()
    n = len(t)
    error = np.zeros(n)
    derivative = 0
    plant_output = np.zeros(n)
    
    # Loop through each time step
    for i in range(1, n):
        error[i] = setpoint[i] - plant_output[i-1]
        if i > 1:
            derivative = (error[i] - error[i-1]) / dt      
        # Fuzzy Control
        control_signal = fuzzy_control(fuzzy_controller, error[i], derivative)
        # FOPTD Process with dead time
        if i * dt >= tau_DT:  # Apply delay
            plant_output[i] = (K / tau) * control_signal * dt + (1 - dt / tau) * plant_output[i-1]
        else:
            plant_output[i] = plant_output[i-1]  # No update during the delay period
    return plant_output

# Simulate and plot the response with the fuzzy controller
response = simulate_system_fuzzy(K, tau, tau_DT, t, setpoint)

# Plot the response
plt.plot(t, setpoint, 'r--', label="Setpoint")
plt.plot(t, response, 'b', label="Response")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("System Response with Fuzzy Controller")
plt.legend()
plt.grid()
plt.show()

