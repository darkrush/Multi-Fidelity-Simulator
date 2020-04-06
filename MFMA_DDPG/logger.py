import pickle
import os

class Logger(object):
    def __init__(self):
        self.log_data_dict = {}
        self.output_dir = None

    def setup(self,output_dir):
        self.output_dir = output_dir
            
    def get_dir(self):
        return self.output_dir
        
    def add_scalar(self,name,y,x):
        if name not in self.log_data_dict:
            self.log_data_dict[name] = []
        self.log_data_dict[name].append([y,x])
        
    def save_dict(self):
        with open(os.path.join(self.output_dir,'log_data_dict.pkl'),'wb') as f:
            pickle.dump(self.log_data_dict,f)

Singleton_logger = Logger()