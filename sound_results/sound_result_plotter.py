from copy import deepcopy
import tkinter as tk
from tkinter import filedialog
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import numpy as np
import xlwt
import matplotlib.pyplot as plt
import pathlib #for creating directories
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
class Algorithm:
    '''This class maps an algorithm ID to experiment time, and sets all algorithm parameters
        data logged in the format:
        "robName:t,x,y,theta,turnProb,signalDistance,signalStrength,linearDist,rotDist:state"'''
    def __init__(self,id):
        self.id = id
        self.name = ''
        self.simList = []
        self.params = {}
        self.myColor = ''
        


class MyPlotter:
    '''Class for plotting/visualizing my simulation results'''
    def __init__(self,valid_result):
        self.algorithmList = {}
        self.algorithmNames = {'0':'-Random-Walk','2':'','3':'', '4':'-Linear',
                                '5':'','6':'',
                                '7': '','8':'','9':''}
        self.file_path = ''
        self.plotsNdata = ''
        self.readme = ''
        self.valid_result = valid_result #row to start reading data from
        self.maxLitter = 200
        self.litStep = self.maxLitter * 10 / 100.0

    def initAlgorithm(self,id,d):
        self.algorithmList[id] = Algorithm(id)
        
        self.appendSimTime(id,d)
        
        for i in d:
            param = i.split(':')
            if len(param) < 2:
                continue
            if id not in ['0','4'] and param[0] in ['A0','noise_mean','noise_std','queue_size']:
                self.algorithmNames[id] += i + ' '
            
            self.algorithmList[id].params[param[0]] = param[1]
    
    def appendSimTime(self,id,d):
        stat = d[-2]
        stat = stat.split(':')
        start_time = stat[1]
        self.algorithmList[id].simList.append(start_time)
    
    def initSimulations(self):
        '''This function selects the folder of the results and matches each 
        simulation times to their respective algorithms'''
        root = tk.Tk()
        root.withdraw()
        self.file_path = filedialog.askdirectory()
        self.file_path +=  '/'
        self.plotsNdata = self.file_path + 'plotsNdata/'
        pathlib.Path(self.plotsNdata).mkdir(parents=False,exist_ok=True)
        readmeFile = self.file_path + 'readme.md'

        with open(readmeFile) as f:
            for data in f:
                data = data.rstrip('\n')
                d = data.split(',')
                if len(d) > 1:
                    exp_id = d[0]
                    exp_id = exp_id.split(':')
                    if exp_id[1] in self.algorithmList:
                        self.appendSimTime(exp_id[1],d)
                    else:
                        #new algorithm found
                        self.initAlgorithm(exp_id[1],d[1:])
    
    def plotRobotTrajectory(self):
        '''Goes through each algorithm and plot trajectory of one simulation'''
        for i in self.algorithmList:
            
            dist_fig,dist_ax =  plt.subplots()
            dist_ax.set_xlabel('Time in seconds')
            dist_ax.set_ylabel('Distance in metres')
            plotTrajectory = True
            simCount = 0
            for sampleSim in self.algorithmList[i].simList:
                allSimTimes = glob.glob(self.file_path + sampleSim + '_m_4wrobot*')
                allX = []
                allY = []
                allDist = []#distance from sound source
                allT = []
                
                # print(allSimTimes)
                for j in allSimTimes:
                    with open(j) as f:
                        data = f.read().rstrip('\n').split('\n')
                        for rowData in data[self.valid_result:]:
                            rowData = rowData.split(':')
                            d = rowData[1].split(',')
                            if len(allT) > 1 and allT[-1] > float(d[0]):
                                continue
                            allX.append(float(d[1]))
                            allY.append(float(d[2]))
                            
                            allDist.append(float(d[5]))
                            allT.append(float(d[0]))
                            
                            # input('>')
                # ptestfig,ptest = plt.subplots()
                if len(allT) !=0:
                    # ptest.plot(allT,allDist)
                    # ptest.set_title(sampleSim)
                    # plt.show()
                    # input('>')
                    # plt.close(ptestfig)
                    # print(sampleSim,len(allT))
                    # if allT[-1] < 1.5:
                    #     print(sampleSim,allT[-1])
                        
                        
                    simCount += 1
                    dist_ax.plot(allT,allDist)#plot distance as function of time
                
                f_robNax = plt.subplots()
                f_rob,ax_rob = f_robNax
                if plotTrajectory:
                    plotTrajectory = False
                    nb = np.linspace(-5,5,4)
                    ax_rob.set_xlim(-5,5)
                    ax_rob.set_ylim(-5,5)
                    ax_rob.tick_params(labelsize=30)
                    nbins = (nb,nb)
                    print(len(allY))
                    H,xedges,yedges = np.histogram2d(allX,allY,bins=nbins)
                    # H = np.rot90(H)
                    # H = np.flipud(H)
                    Hmasked = np.ma.masked_where(H==0,H)
                    im = mpl.image.NonUniformImage(ax_rob,interpolation='bilinear')
                    xcenters = (xedges[:-1] + xedges[1:]) / 2
                    ycenters = (yedges[:-1] + yedges[1:]) / 2
                    im.set_data(xcenters,ycenters,H)
                    pcm = ax_rob.imshow(H,interpolation='bilinear')
                    
                    a2=ax_rob.images.append(im)
                    divider = make_axes_locatable(ax_rob)
                    cax = divider.append_axes("right",size='5%',pad=0.1)
                    cax.tick_params(labelsize=30)
                    a1=f_rob.colorbar(pcm,cax=cax,ticks=[0,400,800,1200,1600,2000])
                    # freqTicks = a1.ax.get_yticks()
                    # print(freqTicks)
                    # # newTics = np.arange(freqTicks[0],freqTicks[-1],400)
                    # a1.ax.set_yticks([0,0.5,1])
                    # a1.ax.set_yticklabels(['Low','Medium','High'])
                    # start,end = a1.get_ylim()
                    # a1.yaxis.set_ticks(np.arange(start,end,400))
                    f_rob.tight_layout()
                    f_rob.savefig(self.plotsNdata + 'robot_trajectory' + self.algorithmNames[i].replace(':','=') +
                                    '_' + sampleSim + '.pdf')
            dist_fig.savefig(self.plotsNdata+'distPlot_'+i+'_simCount='+str(simCount)+'-'+ self.algorithmNames[i].replace(':','=') +'.pdf',bbox_inches='tight')
            plt.close(dist_fig)



if __name__ == '__main__':
    valid_data = 2
    plotObj = MyPlotter(valid_data)
    plotObj.initSimulations()
    print(plotObj.algorithmList.keys())
    
    plotObj.plotRobotTrajectory()