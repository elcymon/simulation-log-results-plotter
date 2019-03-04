# -*- coding: utf-8 -*-

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import os

class Turtlebot_Results:
    def __init__(self,folderPath):
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rc('image',cmap='inferno')
        self.cm = mpl.cm.inferno
        self.cm.N=220
        self.cm.colors = self.cm.colors[0:220]
        self.osSep = '{}'.format(os.sep)
        self.folderPath = folderPath
        
        self.resultFolder = self.folderPath + ['Results']
        pathlib.Path(self.osSep.join(self.resultFolder)).\
            mkdir(parents=False,exist_ok=True)#create folder for saving results
    
    def import_data(self,folderPath):
        '''
        Get data of nest and robots
        return them as separate variables
        '''
        filesPath = self.osSep.join(self.folderPath + ['*.txt'])
        nestHeader = ['t','ros_t','goal_d','nest_x','nest_y','yaw']
        robotHeader = ['t','ros_t','x','y','yaw','prev_sound','curr_sound','turn_p','action']
        nestLog = pd.DataFrame()
        robotLog = []
        for filename in glob(filesPath):
            simData = pd.read_csv(filename,sep=':|,',engine='python')
            
            #nest has nest_x in column names
            if len(simData.columns) == 6:
                nestLog = simData
                nestLog.columns = nestHeader
            else:
                simData.columns = robotHeader
                robotLog.append(simData)
        
        return nestLog,robotLog
    
    def plot_nest_robot_locs(self,nestData,robotList):
        '''
        take in robot list and nest data
        plot nest and robot locations based on x,y data
        '''
        #y values will be distance from nest per time
        #x values will be distance travelled by robot
        #account for offset by subtracting diameter of turtlebot on x value
        #also include t column
        robot_nest_distanceList = []
        for r in robotList:
            robot_nest_distance = pd.DataFrame(columns=['t','nest_dst','robot_dst','curr_sound'])
        
            #loop over shorter df
            rows = len(r.index) if len(r.index) <= len(nestData.index) else len(nestData.index)
            
            for i in range(rows):
                nestxy = np.array(nestData.loc[:,('nest_x','nest_y')].iloc[i])
                
                #robot x,y data
                rxy = np.array(r.loc[:,('x','y')].iloc[i])
                #180 degrees tranformation of xy data
                rxy = -1*rxy - [0.354,0]
#                print(nestxy,rxy)
                robot_dst = np.linalg.norm(nestxy-rxy)
                robot_nest_distance.loc[i,:] = \
                nestData['t'].iloc[i],10-nestData['goal_d'].iloc[i],robot_dst,r['curr_sound'].iloc[i]
                
                robot_nest_distanceList.append(robot_nest_distance)
        
        robot_nest_distanceList[0].plot(x='nest_dst',y='robot_dst',cmap=self.cm)
        return robot_nest_distance
    def plot_dfData(self,robot_nest_distance,colx,coly):
        '''
        plot the data in format for publication
        '''
        f = plt.figure()
        ax = f.gca()
        robot_nest_distance.plot(x=colx,y=coly,cmap=self.cm,ax=ax)
        if colx == 'nest_dst':
            ax.set_xlabel('Nest Travel Distance in metres',fontsize=14,fontweight='bold')
        else:
            ax.set_xlabel(colx,fontsize=14,fontweight='bold')
        
        if coly == 'curr_sound':
            ax.set_ylabel('Sound Intensity',fontsize=14,fontweight='bold')
        elif coly == 'robot_dst':
            ax.set_ylabel('Robot distance in metres',fontsize=14,fontweight='bold')
        else:
            ax.set_ylabel(coly,fontsize=14,fontweight='bold')
        ax.get_legend().remove()
        ax.tick_params(axis='both',which='major',labelsize=18)
        
        ax.set_ylim([0, robot_nest_distance[coly].max()*1.2])
        figName = self.osSep.join(self.resultFolder\
                                  + [self.folderPath[-1] + '_' + coly + '_vs_' + colx +'.pdf'])
        
        f.savefig(figName,bbox_inches='tight')
        
        
    def trim_nest_robot_data(self,nestData,robotList):
        '''
        trim data so that both will be within the same range of
        time
        '''
        
        #extract ranges where x is not zero for robot
        #let last time of nest be last data for now
        maxrobotX = nestData['t'].iloc[-1]
        
        minrobotX = nestData['t'].iloc[0]
        
        for r in robotList:
            #extract where this data has robots starts listing to sound
            timeR = r['t']
            #update minimum only if it is greater than previous
            minrobotX = timeR.iloc[0] if timeR.iloc[0] > minrobotX else minrobotX
            maxrobotX = timeR.iloc[-1] if timeR.iloc[-1] < maxrobotX else maxrobotX
        
        #trim nest and robots to within same time range
        nestData = nestData.loc[(nestData['t'] >= minrobotX) & 
                                (nestData['t'] <= maxrobotX), :]
        for i,r in enumerate(robotList):
            robotList[i] = r.loc[(r['t'] >= minrobotX) & 
                     (r['t'] <= maxrobotX),:]
        return nestData,robotList
        
        
            