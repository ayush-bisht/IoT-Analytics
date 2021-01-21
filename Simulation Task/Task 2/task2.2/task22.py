from collections import deque
import numpy as np
import pandas as pd

mRT_lambda = int(input("Input mean inter-arrival time of RT messages: "))
mnonRT_lambda = int(input("Input mean inter-arrival time of non RT messages: "))
mRT_service = int(input("Input mean service time of an RT message: "))
mnonRT_service = int(input("Input mean service time of a nonRT message: "))
max_MC = int(input("Input the maximum Machine clock cycles you want to run the simulation for: "))


np.random.seed(0)


class Simulator:
  def __init__(self, n_RT=0, n_nonRT=0, s=0, SCL=4, MC=0, RTCL=3, nonRTCL=5, preempted_ST = -1, \
               mRT_lambda=7, mnonRT_lambda=6, mRT_service=3, mnonRT_service=3, max_MC=50):
    self.n_RT = n_RT #number of items in RT queue
    self.n_nonRT = n_nonRT #number of items in non RT queue
    self.s = s #sever status, 0: ideal, 1: servicing RT msg, 2: servicing nonRT msg
    self.SCL = SCL #service clock
    self.MC = MC  #master clock
    self.RTCL = RTCL #next RT packet arrival time
    self.nonRTCL = nonRTCL #next non RT packet arrival time
    self.preempted_ST = preempted_ST #pre-empted service time
    self.mRT_lambda = mRT_lambda #RT msg inter-arrival time mean
    self.mnonRT_lambda = mnonRT_lambda #nonRT msg inter-arrival time mean
    self.mRT_service = mRT_service #RT service time mean
    self.mnonRT_service = mnonRT_service #nonRT service time mean 
    self.RT_queue = deque([])  #store the arrival time of RT msg
    self.nonRT_queue = deque([])
    self.event_list = [[RTCL, 0], [nonRTCL, 1], [SCL, 2]]
    self.max_MC = max_MC
    self.df = pd.DataFrame(columns = ['MC', 'RTCL', 'nonRTCL', 'n_RT', 'n_nonRT', 'SCL', 's', 'preempted_ST'])

  def RT_IA_time(self): return -self.mRT_lambda*np.log(np.random.uniform())
  def nonRT_IA_time(self):  return -self.mnonRT_lambda*np.log(np.random.uniform())
  def RT_S_time(self): return -self.mRT_service*np.log(np.random.uniform()) 
  def nonRT_S_time(self): return -self.mnonRT_service*np.log(np.random.uniform())

  def start_simulation(self):

    while self.MC < self.max_MC:
      
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

    self.RTCL = self.MC + self.RT_IA_time()

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
          
      self.SCL = self.MC + self.RT_S_time()

      self.event_list[2][0] = self.SCL
      self.n_RT -= 1
      self.s = 1


 
  def nonRT_arrival(self):
    self.nonRT_queue.append(self.nonRTCL)
    self.n_nonRT += 1
    self.nonRTCL = self.MC + self.nonRT_IA_time()

    self.event_list[1][0] = self.nonRTCL 
    
    if self.n_nonRT == 1:
      if self.s == 0:
        self.nonRT_queue.popleft()
        self.SCL = self.MC + self.nonRT_S_time()

        self.event_list[2][0] = self.SCL
        self.s = 2
        self.n_nonRT -= 1
    
  def service_completion(self):
    if len(self.RT_queue) > 0:
      self.SCL = self.MC + self.RT_S_time()

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
        
        self.SCL = self.MC + self.nonRT_S_time()

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

simulator3 = Simulator(n_RT=0, n_nonRT=0, s=2, SCL=4, MC=0, RTCL=3, nonRTCL=5, preempted_ST=-1, \
                       mRT_lambda=mRT_lambda, mnonRT_lambda=mnonRT_lambda, mRT_service=mRT_service, \
                       mnonRT_service=mnonRT_service, max_MC=max_MC)

file_path = 'task2.2_output.csv'
simulator3.start_simulation()
simulator3.write_to_file(file_path)

data = pd.read_csv(file_path)
print("\n")
print("OUTPUT TABLE:")
print(data)
