from collections import deque
import pandas as pd
import numpy as np

RT_lambda = int(input("Input inter-arrival time of RT messages: "))
nonRT_lambda = int(input("Input inter-arrival time of non RT messages: "))
RT_service = int(input("Input service time of an RT message: "))
nonRT_service = int(input("Input service time of a nonRT message: "))
max_MC = int(input("Input the maximum Machine clock cycles you want to run the simulation for: "))


class Simulator:
  def __init__(self, n_RT=0, n_nonRT=0, s=0, SCL=4, MC=0, RTCL=3, nonRTCL=5, preempted_ST = -1, \
               RT_lambda=10, nonRT_lambda=10, RT_service=4, nonRT_service=4, max_MC= 50):
    self.n_RT = n_RT #number of items in RT queue
    self.n_nonRT = n_nonRT #number of items in non RT queue
    self.s = s #sever status, 0: ideal, 1: servicing RT msg, 2: servicing nonRT msg
    self.SCL = SCL #service clock
    self.MC = MC  #master clock
    self.RTCL = RTCL #next RT packet arrival time
    self.nonRTCL = nonRTCL #next non RT packet arrival time
    self.preempted_ST = preempted_ST #pre-empted service time
    self.RT_lambda = RT_lambda #RT msg inter-arrival time
    self.nonRT_lambda = nonRT_lambda #nonRT msg inter-arrival time
    self.RT_service = RT_service #RT service time
    self.nonRT_service = nonRT_service #nonRT service time 
    self.RT_queue = deque([])  #store the arrival time of RT msg
    self.nonRT_queue = deque([])
    self.event_list = [[RTCL, 0], [nonRTCL, 1], [SCL, 2]]
    self.max_MC = max_MC
    self.df = pd.DataFrame(columns = ['MC', 'RTCL', 'nonRTCL', 'n_RT', 'n_nonRT', 'SCL', 's', 'preempted_ST'])

  def start_simulation(self):
    while self.MC <= self.max_MC:
      
      if any([self.n_RT, self.n_nonRT, self.SCL]):
        if self.preempted_ST == -1:
          self.preempted_ST = ""
        current_data = self.simulator_data()
        self.df = self.df.append(pd.Series(current_data, index=self.df.columns), ignore_index=True)
        print("MC: {}, RTCL: {}, nonRTCL: {}, nRT: {}, nnonRT: {}, SCL: {}, s: {}, pre-empted: {}".format(*current_data))
	
        if self.preempted_ST == "":
          self.preempted_ST = -1

      if self.SCL == 0:
        event = min(self.event_list[:2])
      else:
        event = min(self.event_list)

      self.MC = event[0]
      if event[1] == 0:
        self.RT_arrival()
      
      elif event[1] == 1:
        self.nonRT_arrival()
      
      elif event[1] == 2:
        self.service_completion()

  def RT_arrival(self):
    self.RT_queue.append(self.RTCL)
    self.n_RT += 1
    self.RTCL = self.MC + self.RT_lambda
    self.event_list[0][0] = self.RTCL 
    
    if self.n_RT == 1 and self.s!=1:
      self.RT_queue.popleft()
      if self.s == 2:
        self.preempted_ST = self.SCL - self.MC
        if self.preempted_ST > 0: 
          self.n_nonRT += 1
          self.nonRT_queue.appendleft(self.preempted_ST + self.MC)
        elif self.preempted_ST == 0:
          self.preempted_ST = -1
          
      self.SCL = self.MC + self.RT_service
      self.event_list[2][0] = self.SCL
      self.n_RT -= 1
      self.s = 1

 
  def nonRT_arrival(self):
    self.nonRT_queue.append(self.nonRTCL)
    self.n_nonRT += 1
    self.nonRTCL = self.MC + self.nonRT_lambda
    self.event_list[1][0] = self.nonRTCL 
    
    if self.n_nonRT == 1:
      if self.s == 0:
        self.nonRT_queue.popleft()
        self.SCL = self.MC + self.nonRT_service
        self.event_list[2][0] = self.SCL
        self.s = 2
        self.n_nonRT -= 1
    

  def service_completion(self):
    if len(self.RT_queue) > 0:
      self.SCL = self.MC + self.RT_service
      self.s = 1
      self.n_RT -= 1
      self.RT_queue.popleft()
      
      self.event_list[2][0] = self.SCL

    elif len(self.nonRT_queue) > 0:
      self.nonRT_queue.popleft()
      self.n_nonRT -= 1
      self.s = 2
      
      if self.preempted_ST > 0:
        self.SCL = self.MC + self.preempted_ST
        self.preempted_ST = -1
      else:
        self.SCL = self.MC + self.nonRT_service
      
      self.event_list[2][0] = self.SCL
    else:
      self.s = 0
      self.SCL = 0
      self.event_list[2][0] = 0

  def simulator_data(self):
    data = [self.MC, self.RTCL, self.nonRTCL, self.n_RT, self.n_nonRT, self.SCL, self.s, self.preempted_ST]
    return data
  
  def write_to_file(self, file_path):
    self.df.to_csv(file_path, index=False)


simulator1 = Simulator(n_RT=0, n_nonRT=0, s=2, SCL=4, MC=0, RTCL=3, nonRTCL=5, preempted_ST=-1, \
                       RT_lambda=RT_lambda, nonRT_lambda=nonRT_lambda, RT_service=RT_service, 
                       nonRT_service=nonRT_service, max_MC=max_MC)

file_path1 = 'task2.1_output.csv'
simulator1.start_simulation()
simulator1.write_to_file(file_path1)

data = pd.read_csv(file_path1)
print("\n")
print("OUTPUT TABLE:")
print(data)
