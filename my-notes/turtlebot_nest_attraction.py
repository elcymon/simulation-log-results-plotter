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
#        self.cm.N=220
#        self.cm.colors = self.cm.colors[0:220]
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
        filesPath = self.osSep.join(folderPath + ['*.txt'])
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
    
    def plot_dfData(self,robot_nest_distance,colx,coly,iStr,exp):
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
                                  + [exp + '_' + iStr + '_' + coly + '_vs_' + colx +'.pdf'])
        
        f.savefig(figName,bbox_inches='tight')
        
    def plot_dfRobotDistRange(self,meansDF,stdDF,exp):#robot_dist_distribution):
        '''
        plot bar figure of robot distance per range of distance traveled by nest
        '''
#        meansDF = robot_dist_distribution.mean(axis=0,skipna=True)
#        stdDF = robot_dist_distribution.std(axis=0,skipna=True)
        
        yerr = []
        for err in stdDF.columns: #if considering multiple types
            yerr.append([stdDF[err].values,stdDF[err].values])
#            yerr.append([stdDF[err],stdDF[err]])
        
        f = plt.figure()
        ax = f.gca()
        meansDF.plot(kind='bar',cmap='plasma',ax=ax,yerr=yerr,rot=0)#cmap=self.cm,
        ax.set_ylabel('Distance in metres',fontsize=18,fontweight='bold')
        ax.set_xlabel('Nest travel distance in metres',fontsize=18,fontweight='bold')
        ax.tick_params(axis='both',which='major',labelsize=18)
        legend = plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.2),fontsize=16,ncol=2)
        figName = self.osSep.join(self.resultFolder\
                                  + [exp + '-nest-following-turtlebot.pdf'])
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
        
        
    
    
    def nest_robot_dist_relation(self,robot_nest_distance):
        '''
        robot_nest_distance is a list of DF, where nest travel distance has been mapped to
        robot's distance from the nest's location.
        '''
        distCols =  {'0 - 2':[], '2 - 4':[],'4 - 6':[], '6 - 8':[], '8 - 10':[]}
        robot_dist_distribution = pd.DataFrame(columns = list(distCols.keys()))
        for rn in robot_nest_distance:
    #        rn = robot_nest_distance
            for drange in distCols:
                dList = [int(i) for i in drange.split(' - ')]#split into [min,max]
                #extract rows where nest distance travelled is within dList range
                distValues = rn.loc[(rn['nest_dst'] >= dList[0]) & (rn['nest_dst'] <= dList[1]),:]
                
                distCols[drange] = distCols[drange] + list(distValues['robot_dst'])
        for drange in distCols:
            robot_dist_distribution[drange] = pd.Series(distCols[drange])
        return robot_dist_distribution
    
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
        
#        robot_nest_distanceList[0].plot(x='nest_dst',y='robot_dst',cmap=self.cm)
        return robot_nest_distanceList
    def analyse_experiment(self):
        nestLog,robotLog = self.import_data(self.folderPath)
        nestData,robotList = self.trim_nest_robot_data(nestLog,robotLog)
        robot_nest_distance = self.plot_nest_robot_locs(nestData,robotList)
#        return robot_nest_distance
        robot_dist_distribution = self.nest_robot_dist_relation(robot_nest_distance)
        
        return robot_dist_distribution
    def merge_nest_robot_info(self,experiments):
        '''
        merges nest and robot logged data in the respective experiments folders
        '''
        
        for i,exp in enumerate(experiments):
            #print progress
            print('\r{}/{}: {}'.format(i+1,len(experiments),exp),end='')
            
            myPath = self.folderPath + [exp]
            self.resultFolder = myPath + ['Results']
            pathlib.Path(self.osSep.join(self.resultFolder)).\
            mkdir(parents=False,exist_ok=True)#create folder for saving results
            
            
            nestLog,robotLog = self.import_data(myPath)
#            return nestLog,robotLog
            nestData,robotList = self.trim_nest_robot_data(nestLog,robotLog)
            robot_nest_distanceList = self.plot_nest_robot_locs(nestData,robotList)
            
            colx = 'nest_dst'
            coly = 'robot_dst'
            
            for i,data in enumerate(robot_nest_distanceList):
                iStr = '{}'.format(i)
                resultName = self.osSep.join(self.resultFolder + [exp + '-' + iStr + '-result-df.csv'])
                self.plot_dfData(data,colx,coly,iStr,exp)
                data.to_csv(resultName)
    def import_nest_robot_info(self,experiments):
        '''
        reads csv data of merged robot nest info and creates a new df from it
        that calls nest_robot_relation function
        '''
        nest_robot_relation = [] #initally set to empty list
        for exp in experiments:
            #path to csv formatted result
            csvFilePath = self.osSep.join(self.folderPath + \
                                      [exp,'Results',exp + '*' + 'result-df.csv'])
            for csvFile in glob(csvFilePath):
                nest_robot_df = pd.read_csv(csvFile,sep=':|,',engine='python')
                robot_dist_distribution = self.nest_robot_dist_relation([nest_robot_df])
                if len(nest_robot_relation) == 0:
                    nest_robot_relation = robot_dist_distribution
                else:
                    nest_robot_relation = pd.concat([nest_robot_relation,\
                                                     robot_dist_distribution],
                            ignore_index=True)
        return nest_robot_relation
    def rw_chemotaxis_df_plot(self,rw=[],chemotaxis=[]):
        '''
        create a df for random walk and chemotaxis then plot a bar graph
        of the two approaches combined
        '''
        meanDF = pd.DataFrame(columns=['Random Walk','Chemotaxis'])
        stdDF = pd.DataFrame(columns=['Random Walk','Chemotaxis'])
        
        if len(chemotaxis) > 0:
            chemotaxisDF = self.import_nest_robot_info(chemotaxis)
            meanDF['Chemotaxis'] = chemotaxisDF.mean()
            stdDF['Chemotaxis'] = chemotaxisDF.std()
            
        if len(rw) > 0:
            rwDF = self.import_nest_robot_info(rw)
            meanDF['Random Walk'] = rwDF.mean()
            stdDF['Random Walk'] = rwDF.std()
        
        exp = self.folderPath[-1]
        return chemotaxisDF
        self.plot_dfRobotDistRange(meanDF,stdDF,exp)
        
            