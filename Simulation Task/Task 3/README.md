## Environment

They sript will run for python version >= 3.6.

Please install the required packages in the `requirements.txt`file by running the following command.

Linux/OS
```
pip3 install -r requirements.txt
```

Windows
```
pip install -r requirements.txt
```

## How to run the script: 

```
python3 simulation_task3.py
```

The code ask for the following inputs:

Mean inter-arrival time for RT messages, Mean service time for RT messages, Mean service time for nonRT messages, Batch size, Total number of batches


The code does not ask for MIAT for nonRT messages as it is a varying value and is hard-coded as a list


## Output

The script will output the observations (mean, 95th percentile, confidence interval, error) for all the RT and nonRT messages.
It does so for both mean batches and 95th percentile batches

The respective graphs are also printed.

NOTE: The code will not work on eos with its default setting (without the modules and python3)
