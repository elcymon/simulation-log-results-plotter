import pandas as pd

def get_robot_data(filename,header):

    robot_data = pd.read_csv(filename,
                             header=None, sep=':|,',engine='python',
                             names=header)
    return robot_data