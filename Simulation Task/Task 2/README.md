## Environment
The script has been testedon python2.7. It will not run on python>=3.6.
Install the packages in the `requirements.txt`.

## Compile and Run:

(This is the same for both task files)

Run the python file with the command: 

for task 2.1:
```python task21.py```
for task 2.2:
```python task22.py```

The following inputs are asked from the user:
1. Inter-arrival time for RT msgs
2. Inter-arrival time for non RT msgs
3. Service time for RT msgs
4. Service time for non RT msgs
5. Number of machine clock cycles you want to the run the simulation for

(eg for task 2.1 a the inputs would be 10, 5, 2, 4, 50)
(for task 2.2 the first 4 values are the mean values, for task 2.1 these are the absoulte values)


# Output:

(This is same for both the task files as well)

Each event is printed on the terminal with the current values for MC, RTCL, nonRTCL, n RT, n nonRT, SCL, s, preempted service time
When the simulation is completed. All these values are saved in a csv file and the final table is printed as well.
