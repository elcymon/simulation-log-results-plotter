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
import seaborn as sns;
from copy import deepcopy
class NA_Results:
    def __init__(self,folderPath,
                 minBounds = [0, 10, 12, 14, 16],
                 maxBounds = [10, 12, 14, 16, np.Inf]):
        '''
        The constructor takes in two lists
        
        folderPath is a list of folders to the path of a file
        minBounds for minimum distances analysis
        maxBounds for list of maximum distance analysis points
        '''
        #set font type so that it does not use font type 3
        sns.set(context='paper',style = 'ticks',palette='colorblind',font_scale=1.5)

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rc('image',cmap='inferno')
        cm1 = deepcopy(mpl.cm.inferno)
        cm1.N=220
        cm1.colors = cm1.colors[0:220]
        self.cm = mpl.colors.LinearSegmentedColormap.from_list('my_inferno',cm1.colors,N=cm1.N)
        #get six equally spaced colours from cmap so that they will be perceptually 
        #different if possible.
#        self.cNum = np.linspace(0,self.cm.N,100,endpoint=True,dtype=np.int)
        #SET COLORS DICTIONARY
        self.colors_dict = {'robots': self.cm(70), 'nest': self.cm(0),
                       'litter':self.cm(200),'bound10':self.cm(130),
                       'bound15': self.cm(170), 'bound20': self.cm(219)}

        self.osSep = '{}'.format(os.sep)
        self.folderPath = folderPath
        
        self.minBounds = minBounds
        self.maxBounds = maxBounds
        
        #setup distance analalysis
        (self.allIDdists,self.boundRange) = \
        self.setup_distance_analysis_dict(minBounds,maxBounds)
        
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
    
    def setup_distance_analysis_dict(self,mins,maxs):
        '''
        set up dictionary for storing means and CI95 analysis if distances from nest
        '''
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
    
    def nest_t_n_dsts(self,ID,col,IDnestData,like='_dst'):
        '''
        Extracts information of distance relationships between nest and mobile robots
        for all simulations for specific algorithm ID
        '''
        allDistData = pd.DataFrame()
        
        for data in IDnestData:
            #filter dist columns
            dstCols = data.filter(like=like).columns
            dstCols = ['t','nest_dst'] + list(dstCols)
            dstCols = list(set(dstCols))
            dstData = data[dstCols]
#            print(dstData)
#            input('>')
            #create long df for all nest dist and robot dists for all simulations
            allDistData = allDistData.append(dstData,ignore_index=True)
        
        #number of robots within specific ranges of distance from nest
        
        return allDistData.sort_values(col,axis=0)
    
    def robots_per_range(self,col,dists_df,mins,maxs,boundRange):
        '''
        This function accepts a dataframe and returns a dataframe
        whose columns represent the number of robots within 
        distance ranges from the nest location
        '''
        
        
        nrobots = pd.DataFrame(columns=[col] + boundRange)
        
        #initialize first column to distance travelled by nest
        nrobots[col] = dists_df[col]
        #get robots dist data
        df = dists_df.filter(like='m_4wrobot')#dists_df.iloc[:,2:]
        for mn,mx,rge in zip(mins,maxs,boundRange):
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
        tyCols.remove('nest_yaw')
        
        alltY = simData[tyCols]
        
        
        return alltX,alltY
    
    
    def prepare_range_header(self,ID):        
        NAtD = ID.split('-')[1] #get AtD parameter
        NAtD = int(NAtD[3:])#get int value
        minBounds = [0, NAtD, NAtD + 2, NAtD + 4, NAtD + 6] # go in steps of two metres after NAtD
        maxBounds = minBounds[1:] + [np.Inf]
        matrixsetup,boundRange = self.setup_distance_analysis_dict(minBounds,maxBounds)
        
        return minBounds,maxBounds,boundRange
    def noTargetsForaged(self,IDnestData,col,t=1000):
        ''' 
        Function to return number of targets foraged after a specific time
        for all simulation data of specific ID
        '''
        noTargetsData = []
        for data in IDnestData:
            if t == 'last':
                d = data['litter_count'].iloc[-1]
                noTargetsData.append(d)
            else:
                #get only column for litcount i.e. foraged targets and when time >= t
                d = data.loc[data['t'] >= t,'litter_count']
    #            print(d.iloc[0])
                noTargetsData.append(d.iloc[0])
        return noTargetsData
         
    def exploration_analysis(self,col='t',IDList=[],showFig=False,t=1000,d=None):
        '''
        This function performs analysis of amount of targets foraged by the swarm
        as a means of testing the swarms exploration ability in unbounded world
        in comparison to bounded world case.
        '''
        if len(IDList) == 0:
            IDList = self.uniqueIDs
        
        mean_df = pd.DataFrame()
        ci95_df = pd.DataFrame()
        mean_robsDF = pd.DataFrame()
        ci95_robsDF = pd.DataFrame()
        
        targetPerTime = {}
        IDList.sort()
        
        for i,  ID in enumerate(IDList):
            print('\r{}/{}: {}'.format(i+1,len(IDList),ID),end='')
            
            #import nest data
            IDnestData = self.import_nest_data(ID)
            
#            
            simIndx = 0#np.random.randint(len(IDnestData))#randomly select simulation
            #get t,x,y information for all robots for the simulation
            alltX,alltY = self.nest_t_x_y_data(ID,'t',
                                    IDnestData[simIndx])
            self.plot_robots_loc(ID,alltX,alltY,showFig=False,
                             lims=[-15,15,-15,15],bounds=[14],
                             boundstyle='-.',nest_path=False,boundtype='circle')
            continue
#            return alltX,alltY
#            self.plot_robots_loc(ID,alltX,alltY,showFig=False,lims=[0,100,0,100],
#                        bounds=[16],boundstyle='-.')
            
#            heatmap,xedges,yedges = self.analyse_exploration_heat(ID,IDnestData)#self.exploration_frequency(ID,alltX,alltY,(-20,120),(-20,120),(5,5))
#            self.exploration_heatmap(ID,heatmap,xedges,yedges)
            
            #get t vs forage count
            #reuse the implementation for t n dsts
            tlitCount = self.nest_t_n_dsts(ID, col,IDnestData.copy(),
                                           like='litter_count')
            allDistData = self.nest_t_n_dsts(ID,col,IDnestData)
            robotsD = allDistData.filter(like='m_4wrobot')
            #attraction trheshold distance from ID
            if d== None:
                AtD = int(ID.split('-')[1][3:]) + 4
            else:
                AtD = d
            dData = robotsD[robotsD <= AtD]
            nrobots = dData.count(axis=1,numeric_only=True)
            
            #squish data
            tlitCount = tlitCount.round({col:0})
            tlitCount[AtD] = nrobots
            squishedData = self.squish_nest_dsts(col,tlitCount)
            
            #append to list for plotting later
            targetPerTime[ID] = squishedData
            
            #robots till time t
            if t == 'last':
                nrobotsT = squishedData[AtD] 
            else:
                nrobotsT = squishedData.loc[squishedData['t'] <= t, AtD]
#            return nrobotsT,squishedData
            mean_robsDF.loc[t,ID] = np.mean(nrobotsT)
            ci95_robsDF.loc[t,ID] = np.std(nrobotsT) * 1.96 / np.sqrt(len(nrobotsT))
            #get number of targets/litter picked after t seconds
            try:
                noTargetsData = self.noTargetsForaged(IDnestData,col,t)
            except:
                continue
#            print(noTargetsData)
            #compute mean and ci95
            mean_df.loc[t,ID] = np.mean(noTargetsData)
            ci95_df.loc[t,ID] = np.std(noTargetsData) * \
                        1.96 / np.sqrt(len(noTargetsData))
        
        return mean_df,ci95_df,mean_robsDF,ci95_robsDF,targetPerTime
    def analyse_exploration_heat(self,ID,IDnestData):
        '''
        goes through all the simulations of a specific ID and creates a sns heatmap
        plot that represents the mean frequencies of visiting particular rangs in 
        the world.
        '''
        
        xedges = np.nan
        yedges = np.nan
        heatmap = np.array([])
        
        for i in range(len(IDnestData)):
        #get t,x,y information for all robots for the simulation
            alltX,alltY = self.nest_t_x_y_data(ID,'t',
                                IDnestData[i])
            
            h,xedge,yedge = self.exploration_frequency(ID,alltX,alltY,(-20,120),(-20,120),(7,7))
            if len(heatmap) == 0:
                xedges = xedge
                yedges = yedge
                heatmap = h
            else:
                heatmap = heatmap + h
        heatmap = np.around(heatmap / float(len(IDnestData)),1)
        
        return heatmap,xedges,yedges
        
        
        
        
    def exploration_heatmap(self,ID,heatmap,xedges,yedges):
        '''
        '''
        heatmap = pd.DataFrame(np.flipud(heatmap),columns=xedges[1:],index=np.flipud(yedges[1:]))
        xranges = ['{:.0f} - {:.0f}'.format(xedges[i],xedges[i+1]) for i in range(len(xedges)-1)]
        yranges = ['{:.0f} - {:.0f}'.format(yedges[i],yedges[i+1]) for i in range(len(yedges)-1)]
        f = plt.figure(figsize=(7,7))#figsize=(4,4))
        ax = sns.heatmap(heatmap,vmin=0,vmax=1000,cmap=self.cm,annot=True,fmt='g',
                         linewidth=0.5,cbar=False,xticklabels=xranges,yticklabels=np.flipud(yranges))
        ax.set_xlabel('x in metres',fontweight='bold')
        ax.set_ylabel('y in metres',fontweight='bold')
        

#        ax.set_xticks(xedges*ax.get_xlim()[1]/(2*np.pi))
#        ax.set_yticks(yedges*ax.get_ylim()[1])
        figName = self.osSep.join(self.resultFolder\
                                  + ['-' +ID + '-exploration-expDuration.pdf'])
        f.savefig(figName,bbox_inches='tight')
        plt.show()
#        f.savefig(figName[0:-4] + '-no-legend.pdf',bbox_inches = 'tight')
                
    def exploration_frequency(self,ID,alltX,alltY,xlimits,ylimits,bins):
        '''
        returns a grid of frequency of robots visiting different ranges in the
        explored world.
        alltX is all x locs of all robots
        alltY is all y locs of all robots
        xlimits is the limits on x axis (xmin,xmax)
        ylimits is the limits on y axis (ymin,ymax)
        bins is the division on x and y axis (xbin,ybin)
        '''
        xmin,xmax = xlimits
        ymin,ymax = ylimits
        
        y = np.array(alltY.iloc[:,2:]) #extract values
        y = list(y.ravel()) # convert to 1d
        
        x = np.array(alltX.iloc[:,2:]) #extract values
        x = list(x.ravel()) # convert to 1d
        
        heatmap, xedges, yedges = \
        np.histogram2d(x + [xmax,xmax,xmin,xmin],y + [ymax,ymin,ymax,ymin],
                       range=[[xmin,xmax],[ymin,ymax]],bins=bins)
        return heatmap, xedges, yedges
#    def prepare_txy_robots_data(self,IDnestData,ID):
#        '''
#        Extract all x and y data for all robots in all simulations of specific ID
#        '''
        
#    def plot_exploration_sns_heatmap(self,ID,IDnestData,edges=(-10,110,-10,110),showFig=False):
#        '''
#        plot heatmap of the exploration behaviour as sns heatmap
#        the exploration is a
#        '''
#        xmin,xmax,ymin,ymax = edges
#        y = alltY.iloc[:,]
    def plot_targetPerTime2(self,targetPerTimeDict,col='t',coly='litter_count',d=None):
        '''
        loops through the list containing targets per time to view how each
        algorithm performs its foraging activit
        This version takes in which column to plot as function of time
        '''
        
        f,ax1 = plt.subplots()
#        ax1.set_facecolor('#FFFFFF')
#        ax1.grid(False)
        colors = np.linspace(0,mpl.cm.inferno.N-1,len(targetPerTimeDict),endpoint=True,dtype=np.int)
        for i,ID in enumerate(targetPerTimeDict):
            data = targetPerTimeDict[ID]
            ax1.plot(data[col],data[coly],label=ID,linewidth=3,color=mpl.cm.inferno(colors[i]))#,cmap='inferno')
#            data.plot(x=col,y=coly,label=ID,linewidth=3,
#                      cmap='inferno',ax=ax1)
#            ax1.plot(data[col],data['litter_count'],label=IDlabel,c=self.cm)
        
        
        legend = ax1.legend(loc='center left',bbox_to_anchor=(1.05,0.5),
            title='Nest Vel',ncol=1,fontsize=18)
        plt.setp(legend.get_title(),fontsize=18,fontweight='bold')
        
        if coly == 'litter_count':
            ax1.set_ylabel('Found Targets',fontsize=18,fontweight='bold')
        elif coly == d:
            ax1.set_ylabel('Robots within ${}m$'.format(d),fontsize=18,fontweight='bold')
        else:
            ax1.set_ylabel(coly,fontsize=18,fontweight='bold')
        
        if col == 't':
            ax1.set_xlabel('Time in seconds',fontsize=18,fontweight='bold')
        elif col == 'nest_dst':
            ax1.set_xlabel('Distance in metres',fontsize=18,fontweight='bold')
        else:
            ax1.set_xlabel(col,fontsize=18)
        ax1.tick_params(axis='both',which='major',labelsize=18)
        figName = self.osSep.join(self.resultFolder\
                                  + ['{}-vs-{}.pdf'.format(coly,col)])
        f.savefig(figName,bbox_inches='tight') 
        
    def plot_targetPerTime(self,targetPerTimeDict,col='t',d=16):
        '''
        loops through the list containing targets per time to view how each
        algorithm performs its foraging activity
        '''
        
        f,ax1 = plt.subplots()
        ax1.set_facecolor('#FFFFFF')
        ax1.grid(False)
        
        if not d:
            ax2 = ax1.twinx()
            ax2.grid(False)
        
            for ID in targetPerTimeDict:
                data = targetPerTimeDict[ID]
                ax2.plot(data[col],data[d],linestyle=':',linewidth=2,cmap=self.cm)
            ax2.set_ylabel('Number of Robots',fontsize=18)
            ax2.tick_params(axis='both',which='major',labelsize=18)
        colors = [0,100,170,219]#np.linspace(0,self.cm.N,len(targetPerTimeDict),endpoint=False,dtype=np.int)
        for i,ID in enumerate(targetPerTimeDict):
            data = targetPerTimeDict[ID]
            dmci95=''
            if d==16:
                dm = np.mean(data[16])
                ci95 = np.std(data[16]) * 1.96 / np.sqrt(len(data[16]))
                dmci95 = ' (${:.1f}\pm{:.2f}$)'.format(dm,ci95)
            if 'NA0-' in ID:
                IDlabel = ID.split('-')[1][3:]
            else:
                IDlabel = ID.split('-')[0][2:]
            if 'M1-D1' in ID:#IF NO CHEMOTAXIS, BOUNDED WORLD
                IDlabel = 'Bounded'
            data.plot(x=col,y='litter_count',label=IDlabel+dmci95,linewidth=3,
                      color=self.cm(colors[i]),ax=ax1)
#            ax1.plot(data[col],data['litter_count'],label=IDlabel,c=self.cm)
        
        
        legend = ax1.legend(loc='center left',bbox_to_anchor=(1.05,0.5),
            title='Nest Vel (< 16m robots)',ncol=1,fontsize=18)
        plt.setp(legend.get_title(),fontsize=18,fontweight='bold')
        ax1.set_ylabel('Found Targets',fontsize=18,fontweight='bold')
        if col == 't':
            ax1.set_xlabel('Time in seconds',fontsize=18,fontweight='bold')
        elif col == 'nest_dst':
            ax1.set_xlabel('Distance in metres',fontsize=18,fontweight='bold')
        else:
            ax1.set_xlabel(col,fontsize=18)
        ax1.tick_params(axis='both',which='major',labelsize=18)
        figName = self.osSep.join(self.resultFolder\
                                  + [col +'-' + 'targets-foraged-per-time.pdf'])
        f.savefig(figName,bbox_inches='tight')  
    def plot_exploration_data(self,mean_df,ci95_df,forageType='NA0',t=1000):
        '''
        This function plots bar chart showing mean and confidence interval for 
        different attraction thresholds in comparison to the bounded world case
        '''
        if type(t) != 'str':
            t = str(t)
            
        mean_df2,ci95_df2 = mean_df.copy(deep=True),ci95_df.copy(deep=True)
        cols = list(mean_df2.columns)
        
        for i,j in enumerate(cols):
            if 'M1-D1' in j:
                #bounded world
                cols[i] = 'Bounded'
            elif forageType == 'NA0':
                # get AtD (attraction threshold distance) part then extract number representing attraction threshold
                j = j.split('-')[1]
#                print(j)
                cols[i] = j[3:]
        
        #replace columns names
        mean_df2.columns = cols
        ci95_df2.columns = cols
        
        mean_df2.sort_index(axis=1,inplace=True)
        ci95_df2.sort_index(axis=1,inplace=True)
        
        yerr = []
        for i in ci95_df2.columns:
            yerr.append([ci95_df2[i].values,ci95_df2[i].values])
#        print(yerr)
#        print(ci95_df2)
        f = plt.figure()
        ax= f.gca()
        ax.set_facecolor('#FFFFFF')
        plt.rcParams['axes.edgecolor'] = "0.15"
        plt.rcParams['axes.linewidth'] = 1.25
        mean_df2.plot(kind='bar',ax=ax,figsize=(4,5),yerr=yerr,rot=0)
        if 'sweep' not in t:
            ax.set_xticks([])
        if 'sweep' in t:
            ax.set_xlabel('Time in seconds',fontsize=18)
        ax.tick_params(axis='both',which='major',labelsize=18)
        ax.set_ylabel('Number of Targets',fontsize=18,fontweight='bold')
        legend = plt.legend(loc='center left',bbox_to_anchor=(1,0.5),
                   title='Nest Vel',fontsize=18,ncol=1)
        plt.setp(legend.get_title(),fontsize=18,fontweight='bold')
        figName = self.osSep.join(self.resultFolder\
                                  + [t+'-'+forageType + '-targets-foraged.pdf'])
        f.savefig(figName,bbox_inches='tight')
        
    def nest_dist_analysis(self,col='t',IDList=[],showFig=False):
        if len(IDList) == 0:
            IDList = self.uniqueIDs
        
        mean_df = pd.DataFrame()#index=self.boundRange,columns=IDList)
        ci95_df = pd.DataFrame()#index=self.boundRange,columns=IDList)
        
        for i,ID in enumerate(IDList):
            print('\r{}/{}: {}'.format(i+1,len(IDList),ID),end='')
            if 'NA0-' in ID:
                col = 't'
            else:
                col = 'nest_dst'
            #read simulations for specific algorithm
            #remember to change all '.' in ID to 'p'
            IDnestData = self.import_nest_data(ID)
            
            simIndx = 5#np.random.randint(len(IDnestData))#randomly select simulation
            #get t,x,y information for all robots for the simulation
            alltX,alltY = self.nest_t_x_y_data(ID,'t',
                                    IDnestData[simIndx])
            
#            self.plot_heatmap(ID,alltX,alltY)
            
#            
            self.plot_robots_loc(ID,alltX,alltY)
#            plot_robots_loc(self,ID,alltX,alltY,showFig=False,lims=[],
#                        bounds=[10,15,20],boundstyle='-.')
            #get dists columns and t column
            allDistData = self.nest_t_n_dsts(ID,col,IDnestData)
#            return allDistData
            #create minBounds, maxBounds and boundRange for ID
            #if ID has AtD parameter in name
            if 'AtD' in ID:
                minBound,maxBound,boundRange = self.prepare_range_header(ID)
            else:
                minBound,maxBound,boundRange = \
                self.minBounds,self.maxBounds,self.boundRange
            
            #extract data of swarm population per distance range
            nrobots = self.robots_per_range(col,allDistData,minBound,maxBound,boundRange)
            
            #filter duplicate row values based on 'col' column
            squished_dsts = self.squish_nest_dsts(col,nrobots)
#            return nrobots, squished_dsts
            #plot squished_dsts data
            self.plot_col_vs_rnge(ID,col,squished_dsts,showFig)
            
            #find mean
            dfmean = pd.DataFrame()
            dfmean[ID] = squished_dsts.mean(axis=0)
            mean_df = pd.concat([mean_df,dfmean],axis=1,sort=False)
            
            #compute 95% confidence interval
            dfci95 = pd.DataFrame()
            dfci95[ID] = squished_dsts.std(axis=0) * \
                        1.96 / np.sqrt(squished_dsts.shape[0])
            ci95_df = pd.concat([ci95_df,dfci95],axis=1,sort=False)
        
#        cols = mean_df.columns
        #filter out only noisy and random walk results
#        cols = [i for i in cols if 'N100' in i or 'M1-D1' in i or 'AtD' in i]
        #plot stacked bar plot of average number of robots per range
        #through out all experiments
#        self.plot_stacked(mean_df[cols].T,col)
        
        sns_heatmap_data = []#self.prepare_sns_heatmap_data(mean_df)
        
        return mean_df,ci95_df,sns_heatmap_data
    def threshold_vs_distance(self,col='t',IDList=[],tex=False):
        '''
        This creates a latex table of threshold vs distance of robots from tne
        nest location.
        '''
        if len(IDList) == 0:
            IDList = self.uniqueIDs
        #distance from nest information.
        mean_dst = pd.DataFrame(index=self.minBounds[1:],columns=IDList)
        ci95_dst = pd.DataFrame(index=self.minBounds[1:],columns=IDList)
        
        for i,ID in enumerate(IDList):
            print('\r{}/{}: {}'.format(i+1,len(IDList),ID),end='')
            if 'NA0-' in ID:
                col = 't'
            else:
                col = 'nest_dst'
            #read simulations for specific algorithm
            #remember to change all '.' in ID to 'p'
            IDnestData = self.import_nest_data(ID)
            
            #get dists columns and t column
            allDistData = self.nest_t_n_dsts(ID,col,IDnestData)
            
            nrobots = pd.DataFrame(columns = [col] + list(mean_dst.index))
            nrobots[col] = allDistData[col] # initialize 'col' column i.e. 't' by default
            #extract robot distances
            df = allDistData.filter(like='m_4wrobot')
            for dst in mean_dst.index:#loop through desired distances
                dData = df[df<=dst]
                nrobots[dst] = dData.count(axis=1,numeric_only=True)
            
            nrobots.sort_values(col,axis=0,inplace=True)
#            return nrobots
            #filter duplicate row values based on 'col' column
            squished_dsts = self.squish_nest_dsts(col,nrobots)
#            return squished_dsts
            squished_dsts.drop(col,inplace=True,axis=1)
            mean_dst[ID] = squished_dsts.mean(axis=0)
            ci95_dst[ID] = squished_dsts.std(axis=0) * \
                        1.96 / np.sqrt(squished_dsts.shape[0])
        
        if tex:
            latexFmt = pd.DataFrame(index=self.minBounds[1:],columns = IDList)
            for c in mean_dst.columns:
                for i in mean_dst.index:
                    latexFmt.loc[i,c] = \
                    '${:.1f} \pm {:.2f}$'.format(mean_dst.loc[i,c],ci95_dst.loc[i,c])
            
            fileName = self.osSep.join(self.resultFolder\
                                      + [col+'-'+'thresholdVSdist.tex'])
            
            #extract only attraction threshold values
            cols = [int(i.split('-')[1][3:]) for i in latexFmt.columns]
            latexFmt.columns = cols#replace columns
            #save as latex
            latexFmt.sort_index(axis=1).to_latex(fileName,encoding='utf-8',escape=False)
        return mean_dst,ci95_dst
        
        
    def df_to_tex(self,filename,mean_df,ci95_df=pd.DataFrame()):
        '''
        converts a dataframe to latex table.
        '''
        latexFmt = pd.DataFrame(index=mean_df.index,columns=mean_df.columns)
        for c in mean_df.columns:
            for i in mean_df.index:
                latexFmt.loc[i,c] = \
                '${:.1f} \pm {:.2f}$'.format(mean_df.loc[i,c],ci95_df.loc[i,c]) \
                if not ci95_df.empty else '${}$'.format(mean_df.loc[i,c])
        #save in result path
        filename =  self.osSep.join(self.resultFolder\
                                      + [filename +'.tex'])
        latexFmt.sort_index(axis=0,inplace=True)
        latexFmt.sort_index(axis=1,inplace=True)
        latexFmt.to_latex(filename,encoding='utf-8',escape=False)
    
    def mean_ci95_tStep(self,analysis,filename='',t=[200,400,600,800,1000]):
        '''
        loop through time steps to create a mean and ci95 df
        '''
        mean_dfs,ci95_dfs,mean_nrobDF,ci95_nrobDF = \
        pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        
        for i in t:
            try:
                m,c,mr,cr,_ = analysis(t=i)   
            except:
                continue
            mean_dfs = pd.concat([mean_dfs,m])
            ci95_dfs = pd.concat([ci95_dfs,c])
            
            mean_nrobDF = pd.concat([mean_nrobDF,mr])
            ci95_nrobDF = pd.concat([ci95_nrobDF,cr])
            
        if len(filename) > 0:
            self.df_to_tex(filename,mean_dfs,ci95_dfs)
            
            self.df_to_tex(filename+'-nrob',mean_nrobDF,ci95_nrobDF)
        
        return mean_dfs,ci95_dfs,mean_nrobDF,ci95_nrobDF
    
    def plot_col_vs_rnge(self,ID,col,squished_dsts,showFig=False):
        '''
        Function plots number of robots within specific range
        as a function of the col i.e. nest_dst or t (time)
        '''
        f = plt.figure()
        ax = f.add_axes([0.1,0.1,0.5,0.7])
        ax.set_ylim([0,10])
        ax.set_xlim([0, squished_dsts[col].max()*1.05])
        axStack = f.add_axes([0.61,0.1,0.1,0.7])
        axStack.set_ylim([0,10])
        cols = [i for i in squished_dsts.columns if 't' not in i]
        
        dataMean = squished_dsts[cols].mean(axis=0)
        
        pd.DataFrame(dataMean).T.plot(kind='bar',stacked=True,ax=axStack,cmap=self.cm)
        
        squished_dsts.plot(x=col,ax=ax,figsize=(10,5),cmap=self.cm)
        if col == 'nest_dst':
            ax.set_xlabel('Distance in metres',fontsize=18,fontweight='bold')
        elif col == 't':
            ax.set_xlabel('Time in seconds',fontsize=18,fontweight='bold')
        else:
            ax.set_xlabel(col,fontsize=18)
        
        ax.set_ylabel('Number of robots',fontsize=18,fontweight='bold')
        axStack.set_xticks([])
        axStack.set_yticks([])
        ax.legend().set_visible(False)
        
        legend = plt.legend(loc='center left',bbox_to_anchor=(1,0.5),
                   title='Distance (m)',fontsize=18)#.set_visible=(True)
        
        plt.setp(legend.get_title(),fontsize=18,fontweight='bold')
        ax.tick_params(axis='both',which='major',labelsize=18)
        figName = self.osSep.join(self.resultFolder\
                                  + [ID + '_' + col +'.pdf'])
        
        f.savefig(figName,bbox_inches='tight')
        
        #no legend version
        axStack.legend().set_visible(False)
        f.savefig(figName[0:-4] + '-no-legend.pdf',bbox_inches='tight')
        
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
    def prepare_sns_heatmap_data(self,mean_df_in):
        '''
        Plot the annotated heatmap for the mean number of robots
        based on their distance from the nest. This is different
        plot type from stacked, because it considers distance
        range starting from nest to desired point. i.e. 0 - x
        
        columns are in format: NAx-Ny-Mi-Dj
        WHERE: x = nest velocity
               y =  noise value in percentage of modelled noise
               i = turn probability multiplier
               j = turn probability divisor
        The function will return dataframe with columns that correspond
        to different nest speeds and rows that correspond to different distances
        from the nest.
        
        The content of each cell in the dataframe is itself a dataframe where
        where columns represent Divisors and rows represent Multipliers.
        Each cell will be the mean number of robots within each of these ranges,
        based on the multiplier and divisor.
        '''
        
        #filter out rows that contain range of distance from nest
        mean_df = mean_df_in.filter(like='-',axis=0)
        
        #make use of only results that are for Noise of 100% modelled value
        n100cols = [i for i in mean_df.columns if 'N100' in i]
        N100_mean_df = mean_df[n100cols]
        
        #prepare means into grid of M by D. i.e. M rows and D cols
        
        #columns of mean_df represent the different simulation experiments
        NAs = [i.split('-')[0] for i in N100_mean_df.columns] # extract nest speeds tested
        
        NAs = list(set(NAs)) # remove duplicates
        
        #prepare dataframe to hold data for different speed and distances
        #use maxBounds as rows (exclude Inf i.e. last maxbound data)
        
        #extract multipliers
        Ms = [i.split('-')[2] for i in N100_mean_df.columns] # extract multipliers tested
        
        Ms = list(set(Ms)) # remove duplicates
        Ms = [int(i[1:]) for i in Ms]
        
        #extract divisors
        Ds = [i.split('-')[3] for i in N100_mean_df.columns] # extract divisors tested
        
        Ds = list(set(Ds)) # remove duplicates
        Ds = [int(i[1:]) for i in Ds]
        
        
        #CREATE MULTIINDEX DF to store the values
        #ignore last boundrange index used for mean_df
        dists = [int(i.split(' - ')[1]) for i in list(mean_df.index)]
        col_index = pd.MultiIndex.from_product([NAs,Ds])
        idx = pd.MultiIndex.from_product([dists,Ms])
        
        sns_heatmap_df = pd.DataFrame(index=idx,columns=col_index)
        
        #do cumulative some of the different ranges in order to get total robots per range from nest
        #ignore last index which represents lost robots
        mean_df_cumsum = mean_df.cumsum(axis=0)
        mean_df_cumsum.index = dists
        
        for col in mean_df_cumsum.columns:
            col_list = col.split('-')
            NA = col_list[0]
            M = int(col_list[2][1:])
            D = int(col_list[3][1:])
            for ind in mean_df_cumsum.index:
#                print(mean_df_cumsum.loc[ind,col])
#                input('>')
                sns_heatmap_df.loc[(ind,M),(NA,D)] = mean_df_cumsum.loc[ind,col]
        
        #plots the heatmap data of frequency of visits
        grid_kws = {"height_ratios":(0.05,0.9),"hspace":0.3}
        for na in sns_heatmap_df.columns.levels[0]:
            for dst in sns_heatmap_df.index.levels[0]:
                heatData = sns_heatmap_df.loc[dst,na]
                heatData = heatData[heatData.columns].astype(float)
                heatData.sort_index(axis=0,inplace=True,ascending=False)
                heatData.sort_index(axis=1,inplace=True,ascending=True)
                
                f, (cbar_ax,ax) = plt.subplots(2,gridspec_kw=grid_kws,figsize=(4,4))
                ax = sns.heatmap(heatData,linewidths=0.5,vmin=0,
                    ax=ax,cbar_ax=cbar_ax,cbar_kws={'orientation':'horizontal'},
                    annot=True,cmap=self.cm,vmax=10)
                
                ax.set_xlabel('Divisors',fontweight='bold')
                ax.set_ylabel('Multipliers',fontweight='bold')
                figName = self.osSep.join(self.resultFolder \
                                          + [na +'-'+ str(dst) + '.pdf'])
                f.savefig(figName,bbox_inches='tight')
                
                #no colorbar version
                f = plt.figure(figsize=(4,4))
                ax = sns.heatmap(heatData,vmin=0,vmax=10,cmap=self.cm,annot=True,
                                 linewidth=0.5,cbar=False)
                ax.set_xlabel('Divisors',fontweight='bold')
                ax.set_ylabel('Multipliers',fontweight='bold')
                f.savefig(figName[0:-4] + '-no-legend.pdf',bbox_inches = 'tight')
                plt.close()
        return sns_heatmap_df
    
#        for nest_vel in sns_heatmap_df.columns:
#            
        
        #rows represent different ranges of distances.
        
    def plot_robots_loc(self,ID,alltX,alltY,showFig=False,lims=[],
                        bounds=[10,15,20],boundtype=None,boundstyle='-.',nest_path=True):
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
            t = tx.filter(regex='^t$')
            nest_x = tx.filter(regex='nest_x').iloc[0]
            nest_y = ty.filter(regex='nest_y').iloc[0]
            f = plt.figure()
            #plot boundaries of world
            ax = plt.gca()
            if len(bounds) == 3 and boundtype == 'circle':
                bound10 = plt.Circle((nest_x.iloc[0],nest_y.iloc[0]),bounds[0],linewidth=3,
                                     label='${}m$'.format(bounds[0]),linestyle='-',
                                     color=self.colors_dict['bound10'],fill=False)
                bound15 = plt.Circle((nest_x.iloc[0],nest_y.iloc[0]),bounds[1],linewidth=3,
                                     label='${}m$'.format(bounds[1]),linestyle='-.',
                                     color=self.colors_dict['bound15'],fill=False)
                bound20 = plt.Circle((nest_x.iloc[0],nest_y).iloc[0],bounds[2],linewidth=3,
                                     label='${}m$'.format(bounds[2]),linestyle=':',
                                     color=self.colors_dict['bound20'],fill=False)
                
                ax.add_patch(bound10)
                ax.add_patch(bound15)
                ax.add_patch(bound20)
            elif len(bounds) == 1  and boundtype == 'circle':
                bound15 = plt.Circle((nest_x,nest_y),bounds[0],linewidth=3,label='${}m$'.format(bounds[0]),
                                     linestyle=boundstyle,
                                     color=self.colors_dict['bound15'],fill=False)
                ax.add_patch(bound15)
            elif boundtype == 'square':
                xy1,xy2,width,height = bounds
                rectWall = plt.Rectangle((xy1,xy2), width, height,fill=False,color='k',linewidth=3,
                              label='${}m$'.format(width),linestyle='-')
                
                ax.add_patch(rectWall)
            
            ax.axis('square')
            
            #change axis based on experiment type
            if 'move' in self.folderPath[-1] or 'NA0-' not in ID:
                ax.axis([-40, 110, -40, 40])
            else:
                ax.axis([-40, 40, -40, 40])
            
            if len(lims) > 0:
                ax.axis(lims)
            #plot path taken by nest robot
            hist = 130#history of nest locations to include
            if nest_path:
                if step < hist:
                    plt.plot(alltX.iloc[0:step,1],alltY.iloc[0:step,1],
                         linestyle=':',color=self.colors_dict['nest'],label='nest path')
                else:
                    plt.plot(alltX.iloc[step-hist:step,1],alltY.iloc[step-hist:step,1],
                         linestyle=':',color=self.colors_dict['nest'],label='nest path')
            
            # ty starts from location 3 because yaw of nest in included.:(
            plt.plot(tx.filter(regex='litter*'),ty.filter(regex='litter*'),marker='*',linestyle='',
                     label='litter',color=self.colors_dict['litter'],markersize=7)
            plt.plot(tx.filter(regex='m_4wrobot*'),ty.filter(regex='m_4wrobot*'),marker='o',linestyle='',color=self.colors_dict['robots'],
                     label='robots',markerfacecolor=self.colors_dict['robots'],markersize=7)
            plt.plot(nest_x,nest_y,marker='X',markersize=15,linestyle='',color=self.colors_dict['nest'],
                     label='nest',markerfacecolor=self.colors_dict['nest'])#,markersize=1)
            
#            #restrict to 25 by 25
#            plt.xlim([-25, 25])
#            plt.ylim([-25, 25])
            plt.tick_params(axis='both',which='major',labelsize=18)
            plt.legend(loc='center left',bbox_to_anchor=(1,0.5),
                   fontsize=18)
            figName = self.osSep.join(self.resultFolder\
                                  + [ID + '_' + str(t.iloc[0]) +'.pdf'])
        
            
            if not showFig:
                f.savefig(figName,bbox_inches='tight')
                #no legend version
                plt.legend().set_visible(False)
                f.savefig(figName[0:-4] + '-no-legend.pdf',bbox_inches='tight')
                plt.close()
                
            
            if showFig:
                plt.title(ID + '_' + str(t.iloc[0]))
            
    
    
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
        
        if ID.split('-')[0] != 'NA0':
            xmax = 110
            
        y = alltY.iloc[:,2:].to_numpy() #extract values
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
        
if __name__ == '__main__':
    folderPath = ['..','results','2019-02-16-NoBound-NA_Vel_M_D_Test_10']
#    '2019-02-18-NoBound-NA_att_threshold_vs_robDist']
    na = NA_Results(folderPath=folderPath)