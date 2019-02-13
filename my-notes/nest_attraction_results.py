# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 08:40:12 2019

@author: elcymon
@description: This script is plots a number of results for tne nest attraction simulation and robot experiments.
"""
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
import numpy as np
import xlwt
import matplotlib.pyplot as plt
import pathlib
import sys
import pandas as pd
import os


class NA_Results:
    def __init__(self,folderPath,
                 minBounds = [0, 10, 13, 15, 20],
                 maxBounds = [10, 13, 15, 20, np.Inf]):
        '''
        The constructor takes in two lists
        
        folderPath is a list of folders to the path of a file
        minBounds for minimum distances analysis
        maxBounds for list of maximum distance analysis points
        '''
        #set font type so that it does not use font type 3
        
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        
        #SET COLORS DICTIONARY
        self.colors_dict = {'robots': 'C0', 'nest': 'C1',
                       'bound10': 'C2', 'bound13': 'C9',
                       'bound15': 'C6', 'bound20': 'C3'}

        self.osSep = '{}'.format(os.sep)
        self.folderPath = folderPath
        
        self.minBounds = minBounds
        self.maxBounds = maxBounds
        
        #setup distance analalysis
        (self.allIDdists,self.boundRange) = \
        self.setup_distance_analysis_dict()
        
        #create readme files
        self.uniqueIDs = self.create_readme()
        
        #setup results folder
        self.resultFolder = self.folderPath + ['Results']
        pathlib.Path(self.osSep.join(self.resultFolder)).\
            mkdir(parents=False,exist_ok=True)#create folder for saving results
            
        
        
    def create_readme(self):
        '''
        creates a readme2.md file for all simulations in folderPath
        It also creates a pdSeries of unique simulation IDs
        which represent the different algorithms
        '''
        folderPath = self.folderPath
        IDList = []
        readmeFile = self.osSep.join(folderPath + ['readme2.md'])
        
        with open(readmeFile,'w+') as f:
            for name in glob(self.osSep.join(folderPath + ['*litter_count*'])):
                name = name.split(os.sep)[-1] #last element is filename
                name = name.split('_')#split into [simulaiton_ID/prefix, result_type]
                prefix = name[0]
                algID = prefix.split('-')
                algID = '-'.join(algID[1:-7])
                lineData = 'pefix:{},ID:{}\n'.format(prefix,algID)
                
                algID = algID.replace('p','.')# replace p in nest velocity with .
                IDList.append(algID)
                f.write(lineData)#write simulation data to readme2 file
        
        return pd.Series(IDList).unique()#extract all unique simulation algorithms
    
    def setup_distance_analysis_dict(self):
        '''
        set up dictionary for storing means and CI95 analysis if distances from nest
        '''
        mins = self.minBounds
        maxs = self.maxBounds
        
        boundRange = []
        for mn,mx in zip(mins,maxs):
            if mx != np.Inf:
                boundRange.append('{} - {}'.format(mn,mx))
            else:
                boundRange.append('> {}'.format(mn))
        return ({'Mean': pd.DataFrame(index=boundRange),
                'CI95': pd.DataFrame(index=boundRange)},boundRange)
    def import_nest_data(self,ID):
        '''
        function to import nest simulation data for analysis
        done for specific algorithm ID
        '''
        filesPath = self.osSep.join(self.folderPath + ['*'+ID + '*litter_count*'])
        IDnestData = []
        for filename in glob(filesPath):
            simData = pd.read_csv(filename,sep=':|,',engine='python')
            #all data for a particular algorithm
            IDnestData.append(simData)
            
        return IDnestData
    
    def nest_t_n_dsts(self,ID,col,IDnestData):
        '''
        Extracts information of distance relationships between nest and mobile robots
        for all simulations for specific algorithm ID
        '''
        allDistData = pd.DataFrame()
        
        for data in IDnestData:
            #filter dist columns
            dstCols = data.filter(like='_dst').columns
            dstCols = ['t'] + list(dstCols)
            dstData = data[dstCols]
#            print(dstData)
#            input('>')
            #create long df for all nest dist and robot dists for all simulations
            allDistData = allDistData.append(dstData,ignore_index=True)
        
        #number of robots within specific ranges of distance from nest
        
        return allDistData.sort_values(col,axis=0)
    
    def robots_per_range(self,col,dists_df):
        '''
        This function accepts a dataframe and returns a dataframe
        whose columns represent the number of robots within 
        distance ranges from the nest location
        '''
        mins = self.minBounds
        maxs = self.maxBounds
        
        nrobots = pd.DataFrame(columns=[col] + self.boundRange)
        
        #initialize first column to distance travelled by nest
        nrobots[col] = dists_df[col]
        df = dists_df.iloc[:,2:]
        for mn,mx,rge in zip(mins,maxs,self.boundRange):
            mnData = df[df > mn]
            mxData = mnData[mnData <= mx]
            nrobots[rge] = mxData.count(axis=1,numeric_only=True)
        
        return  nrobots.sort_values(col,axis=0)
    
    def squish_nest_dsts(self,col,dists_df):
        '''
        make the col column values have unique values
        replace duplicates with mean values of all the repetitions
        '''
        if col == 'nest_dst':
            #if using nest_dst, round to nearest metres
            dists_df = dists_df.round({col:0})
        squished_dsts = pd.DataFrame(columns=dists_df.columns)
        
        #extract unique nest dists and make 
        squished_dsts[col] = pd.Series(dists_df[col]).unique()
        
        for d in squished_dsts[col]:
#            print(squished_dsts[squished_dsts['nest_dst'] == d])
#            print(dists_df[dists_df['nest_dst'] == d].mean(axis=0))
            squished_dsts[squished_dsts[col] == d] = \
            dists_df[dists_df[col] == d].mean(axis=0).values
        return squished_dsts
    def nest_t_x_y_data(self,ID,col,simData):
        '''
        Extract all xy positions of robots and nest for specific instance of
        an algorithm's simulation. Returns a tuple of tx,ty for all robots and nest
        ideally, col should be t.
        '''
#        alltX = pd.DataFrame()
        # prepend x columns with t
        txCols = [col] + list(simData.filter(like='_x').columns)
        alltX = simData[txCols]
        
        # prepend y columns with t
#        alltY = pd.DataFrame()
        tyCols = [col] + list(simData.filter(like='_y').columns)
        alltY = simData[tyCols]
        
        
        return alltX,alltY
    
    
            
    def nest_dist_analysis(self,col='t',IDList=[],showFig=False):
        if len(IDList) == 0:
            IDList = self.uniqueIDs
        
        mean_df = pd.DataFrame(index=self.boundRange,columns=IDList)
        ci95_df = pd.DataFrame(index=self.boundRange,columns=IDList)
        for i,ID in enumerate(IDList):
            print('\r{}/{}: {}'.format(i+1,len(IDList),ID),end='')
            #read simulations for specific algorithm
            #remember to change all '.' in ID to 'p'
            IDnestData = self.import_nest_data(ID.replace('.','p'))
            
            simIndx = 5#np.random.randint(len(IDnestData))#randomly select simulation
            #get t,x,y information for all robots for the simulation
            alltX,alltY = self.nest_t_x_y_data(ID,'t',
                                    IDnestData[simIndx])
            
#            self.plot_heatmap(ID,alltX,alltY)
#            
#            self.plot_robots_loc(ID,alltX,alltY)
#            continue
            #get dists columns and t column
            allDistData = self.nest_t_n_dsts(ID.replace('.','p'),col,IDnestData)
            
            
            #extract data of swarm population per distance range
            nrobots = self.robots_per_range(col,allDistData)
            
            #filter duplicate row values based on 'col' column
            squished_dsts = self.squish_nest_dsts(col,nrobots)
            
            #plot squished_dsts data
            self.plot_col_vs_rnge(ID,col,squished_dsts,showFig)
            
            #find mean
            mean_df[ID] = squished_dsts.mean(axis=0)
            
            #compute 95% confidence interval
            ci95_df[ID] = squished_dsts.std(axis=0) * \
                        1.96 / np.sqrt(squished_dsts.shape[0])
        
        cols = mean_df.columns
        #filter out only noisy and random walk results
        cols = [i for i in cols if 'N100' in i or 'M1-D1' in i]
        #plot stacked bar plot of average number of robots per range
        #through out all experiments
        self.plot_stacked(mean_df[cols].T,col)
        
        return mean_df,ci95_df
        
    def plot_col_vs_rnge(self,ID,col,squished_dsts,showFig=False):
        '''
        Function plots number of robots within specific range
        as a function of the col i.e. nest_dst or t (time)
        '''
        f = plt.figure()
        squished_dsts.plot(x=col,ax=f.gca(),figsize=(10,5))
        if col == 'nest_dst':
            plt.xlabel('Distance in metres',fontsize=18,fontweight='bold')
        elif col == 't':
            plt.xlabel('Time in seconds',fontsize=18,fontweight='bold')
        else:
            plt.xlabel(col,fontsize=18)
        
        plt.ylabel('Number of robots',fontsize=18,fontweight='bold')
        legend = plt.legend(loc='center left',bbox_to_anchor=(1,0.5),
                   title='Distance',fontsize=18)#.set_visible=(True)
        
        plt.setp(legend.get_title(),fontsize=18,fontweight='bold')
        plt.tick_params(axis='both',which='major',labelsize=18)
        figName = self.osSep.join(self.resultFolder\
                                  + [ID + '_' + col +'.pdf'])
        
        f.savefig(figName,bbox_inches='tight')
        plt.title(ID)
        if not showFig:
            plt.close()
        
        
    def plot_stacked(self,mean_df,col):
        '''
        plot stacked bar chart of different algorithms
        '''
        f = plt.figure()
        mean_df.plot(kind='bar',stacked=True,ax=f.gca(),rot=90,figsize=(10,5))
        plt.xlabel('Algorithms',fontsize=18,fontweight='bold')
        plt.ylabel('Number of Robots',fontsize=18,fontweight='bold')
        
        legend = plt.legend(loc='center left',bbox_to_anchor=(1,0.5),
                   title='Distance',fontsize=18)#.set_visible=(True)
        
        plt.setp(legend.get_title(),fontsize=18,fontweight='bold')
        plt.tick_params(axis='both',which='major',labelsize=18)
        
        figName = self.osSep.join(self.resultFolder\
                                  + [self.folderPath[-1] + '_' + col +'.pdf'])
        
        f.savefig(figName,bbox_inches='tight')
        
        plt.title(self.folderPath[-1])
        
        
    def plot_robots_loc(self,ID,alltX,alltY,showFig=False):
        '''
        create 10 scatter plots of robot xy locations.
        '''
        #get number of rows/timesteps recorded
        nrows = alltX.shape[0]
        
        #time steps to plot : 10 plots from 0 to sim duration
        steps = np.linspace(0,nrows-1,num=10,endpoint=True,dtype=int)
        
        #loop through steps to plot and create scatter plot
        for step in steps:
            tx = alltX.iloc[step,:]
            ty = alltY.iloc[step,:]
#            print(tx)
#            print(ty)
            t = tx[0]
            nest_x = tx[1]
            nest_y = ty[1]
            f = plt.figure()
            #plot boundaries of world
            ax = plt.gca()
            bound10 = plt.Circle((nest_x,nest_y),10,linewidth=2,label='10m',
                                 color=self.colors_dict['bound10'],fill=False)
            bound15 = plt.Circle((nest_x,nest_y),15,linewidth=2,label='15m',
                                 color=self.colors_dict['bound15'],fill=False)
            bound20 = plt.Circle((nest_x,nest_y),20,linewidth=2,label='20m',
                                 color=self.colors_dict['bound20'],fill=False)
            
            ax.add_patch(bound10)
            ax.add_patch(bound15)
            ax.add_patch(bound20)
            
            ax.axis('square')
            
            #change axis based on experiment type
            if 'move' in self.folderPath[-1]:
                ax.axis([-40, 110, -40, 40])
            else:
                ax.axis([-40, 40, -40, 40])
            # ty starts from location 3 because yaw of nest in included.:(
            plt.plot(tx[2:],ty[3:],'o',color=self.colors_dict['robots'],
                     label='robots',markerfacecolor=self.colors_dict['robots'])#,markersize=1)
            plt.plot(nest_x,nest_y,'o',color=self.colors_dict['nest'],
                     label='nest',markerfacecolor=self.colors_dict['nest'])#,markersize=1)
            
#            #restrict to 25 by 25
#            plt.xlim([-25, 25])
#            plt.ylim([-25, 25])
            plt.tick_params(axis='both',which='major',labelsize=18)
            plt.legend(loc='center left',bbox_to_anchor=(1,0.5),
                   fontsize=18)
            figName = self.osSep.join(self.resultFolder\
                                  + [ID + '_' + str(t) +'.pdf'])
        
            f.savefig(figName,bbox_inches='tight')
            plt.title(ID + '_' + str(t))
            if not showFig:
                plt.close()
                
    def plot_heatmap(self,ID,alltX,alltY,showFig=False):
        '''
        create heatmap of the simulation instance, showing 
        over the duration of the simulation
        '''
        
        #set plot area boundaries
        ymin = -40
        ymax = 40
        xmin = -40
        xmax = 40
        
        if 'move' in self.folderPath[-1]:
            xmax = 110
            
        y = alltY.iloc[:,3:].to_numpy() #extract values
        y = list(y.ravel()) # convert to 1d
        
        x = alltX.iloc[:,2:].to_numpy() #extract values
        x = list(x.ravel()) # convert to 1d
        print(len(x))
        print(len(y))
        heatmap, xedges, yedges = \
        np.histogram2d(x + [xmax,xmax,xmin,xmin],y + [ymax,ymin,ymax,ymin],
                       bins=(10,10))
        extent = [xmin, xmax, ymin, ymax]
        f = plt.figure()
        plt.clf()
        ax = plt.gca()
        plt.tick_params(axis='both',which='major',labelsize=18)
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
        
        im = plt.imshow(heatmap.T, extent=extent, origin='lower',
                        vmin=0, vmax=1000, interpolation='gaussian',
                        cmap='gist_heat')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right',size='5%',pad=0.05)
        cbar = plt.colorbar(im,cax=cax)
        cbar.ax.tick_params(labelsize=18)
        
        figName = self.osSep.join(self.resultFolder\
                                  + [ID + '_heatmap' +'.pdf'])
        
        f.savefig(figName,bbox_inches='tight')
        plt.title(ID)
        if not showFig:
            plt.close()
        
        
            