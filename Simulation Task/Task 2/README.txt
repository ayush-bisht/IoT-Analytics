REQUIREMENTS: 

Both the codes assume the following modules are installed:
collections module (for deque)
numpy (for random number generation)
pandas (for saving and displaying the information)

The code will run on python2.7 with these modules involved.


COMPILE AND RUN:

(This is the same for both task files)

Run the python file with the command: 
for task 2.1 - python task21.py
for task 2.2 - python task22.py

The following inputs are asked from the user:
1. Inter-arrival time for RT msgs
2. Inter-arrival time for non RT msgs
3. Service time for RT msgs
4. Service time for non RT msgs
5. Number of machine clock cycles you want to the run the simulation for

(eg for task 2.1 a the inputs would be 10, 5, 2, 4, 50)
(for task 2.2 the first 4 values are the mean values, for task 2.1 these are the absoulte values)


OUTPUT:

(This is same for both the task files as well)

Each event is printed on the terminal with the current values for MC, RTCL, nonRTCL, n RT, n nonRT, SCL, s, preempted service time

When the simulation is completed. All these values are saved in a csv file and the final table is printed as well.