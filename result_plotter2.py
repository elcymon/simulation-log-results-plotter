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

class Algorithm:
    '''This class maps an algorithm ID to experiment time, and sets all algorithm parameters'''
    def __init__(self,id):
        self.id = id
        self.name = ''
        self.simList = []
        self.params = {}
        self.myColor = ''

    def setup_litterMatrix(self):
        names = ['Simulation times']
        names.extend([str(i) for i in range(0,101,10)])

        formats = ['f4']*11
        formats.insert(0,'|U20')
        
        tHeader = ['Sim Time']
        tHeader.extend(range(0,101,10))
        
        self.dat_dtype = np.dtype({'names':names,'formats':formats})
        self.litterCounts = np.zeros(1,self.dat_dtype)#30 to represent iteration counts, but can change automatically.
        self.litterCounts[0] = np.array(tuple(tHeader),self.dat_dtype)

    def append_litterCounts(self,sim_lit_data):
        #print(len(self.litterCounts[0]),len(sim_lit_data))
        # print(sim_lit_data)
        # print(self.dat_dtype)
        while len(sim_lit_data) < len(self.dat_dtype):
            '''padding with invalid data if data incomplete'''
            sim_lit_data.append(-99.0)
        np_sim_lit_data = np.array(tuple(sim_lit_data),self.dat_dtype)
        self.litterCounts = np.r_[self.litterCounts,[np_sim_lit_data]]
        


class MyPlotter:
    '''Class for plotting/visualizing my simulation results'''
    def __init__(self,valid_result):
        self.algorithmList = {}
        self.algorithmNames = {'0':'Random Walk', '10':'Selective Attraction',
                                '104':'Selective Attraction (4 Att threshold)','102':'Selective Attraction (2 Att threshold)',
                                '20':'Selective Repulsion','30':'Rep-Att',
                                '31': 'Rep-Att (pure)','50':'Greedy','60':'Optimal',
                                '305': 'Rep-Att (5Hz)','301':'Rep-Att (1Hz)',
                                '304': 'Rep-Att (4 Att threshold)', '302': 'Rep-Att (2 Att threshold'}
        self.file_path = ''
        self.plotsNdata = ''
        self.readme = ''
        self.valid_result = valid_result #row to start reading data from
        self.maxLitter = 200
        self.litStep = self.maxLitter * 10 / 100.0
        self.colorList = {'0':(0.50196078431,0,0), '10':(0.66666666666,0.43137254902,0.15686274509),
                                '104':(0,0,1),'102':(0,0,0.5),
                                '20':(0.50196078431,0.50196078431,0),'30':(0.23529411764,0.70588235294,0.29411764705),
                                '31': (0.27450980392,0.94117647058,0.94117647058),'50':(0,0.50980392156,0.78431372549),'60':(0,0,0.50196078431),
                                '305': (0.94117647058,0.19607843137,0.90196078431),'301':(0.98039215686,0.74509803921,0.74509803921),
                                '304': (0,0,0), '302': (0.7,0.7,0.7)}
        # mpl.style.use('seaborn-colorblind')

    def initAlgorithm(self,id,d):
        self.algorithmList[id] = Algorithm(id)
        self.algorithmList[id].setup_litterMatrix()#initialize setup matrix
        # print('length',len(self.algorithmList))
        # self.algorithmList[id].myColor = self.colorList[id]
        self.appendSimTime(id,d)
        
        for i in d:
            param = i.split(':')
            self.algorithmList[id].params[param[0]] = param[1]
    
    def appendSimTime(self,id,d):
        stat = d[-2]
        stat = stat.split(':')
        start_time = stat[1]
        self.algorithmList[id].simList.append(start_time)
        
        sim_lit_data = self.getLitterCount(start_time)
        self.algorithmList[id].append_litterCounts(sim_lit_data)

        
    def getLitterCount(self,start_time):
        '''returns the time and litter count per time as an iteratortool'''
        fname = self.file_path + start_time + '_litter_count.txt'
        with open(fname,'r') as f:
            data = f.read().rstrip('\n').split('\n')

            t = [start_time]
            lt = np.arange(0,self.maxLitter+1,self.litStep)

            lt_pct = 0 # start from 0 pct at location in lt

            for x in data[self.valid_result:]:
                a = x.split(',')
                #check for percentage before appending.
                a_float = [float(i) for i in a]
                if lt_pct >= len(lt):
                    break
                elif lt_pct == len(lt)-1 and a_float[1] >= lt[lt_pct]:
                    t.append(a_float[0])
                    break
                elif a_float[1] >= lt[lt_pct] and a_float[1] < lt[lt_pct+1]:
                    t.append(a_float[0])
                    lt_pct += 1

                # t.append(float(a[0]))
                # lt.append(float(a[1]))

        # return zip(t,lt)
        return t

    
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

    def processLitterData(self):
        for i in self.algorithmList:
            pctNames = self.algorithmList[i].litterCounts.dtype.names
            pctMean = ['mean']
            confidenceInterval = ['95% Confidence Interval']
            for j in pctNames[1:]:
                # print(self.algorithmList[i].litterCounts[j][1:])
                litData = self.algorithmList[i].litterCounts[j][1:]
                pctMean.append(np.mean(litData))
                stDev = np.std(litData)
                confidenceInterval.append(1.96*stDev/np.sqrt(len(litData)))
            self.algorithmList[i].append_litterCounts(pctMean)
            self.algorithmList[i].append_litterCounts(confidenceInterval)

    def saveLitterDataXLS(self):
        wb = xlwt.Workbook()
        ws = wb.add_sheet('Litter Data')
        row = 0
        ws.write(row,0,'MEAN')
        row += 1
        algkeys = list(self.algorithmList.keys())
        algkeys.sort()
        for i in algkeys:
            pctNames = list(deepcopy(self.algorithmList[i].litterCounts.dtype.names))
            
            if row == 1:
                for ii,pct in enumerate(pctNames):
                    if ii == 0:
                        ws.write(row,ii,'ID')
                    else:
                        ws.write(row,ii,pct)
                row += 1
            for jj,pct in enumerate(pctNames):
                if jj == 0:
                    ws.write(row,jj,int(i))
                else:
                    ws.write(row,jj,float(self.algorithmList[i].litterCounts[pct][-2]))
            row += 1
        row += 3
        ws.write(row,0,'95% Confidence Interval')
        row +=1
        for i in algkeys:
            for jj,pct in enumerate(pctNames):
                if jj == 0:
                    ws.write(row,jj,int(i))
                else:
                    ws.write(row,jj,float(self.algorithmList[i].litterCounts[pct][-1]))
            row += 1
        wb.save(self.plotsNdata+'Results.xls')
    def plotLitterData(self):
        fig = plt.figure()
        ax = plt.axes()
        algkeys = list(self.algorithmList.keys())
        algkeys.sort()
        for i in algkeys:
            pcts = list(self.algorithmList[i].litterCounts[0])
            pcts = np.array(pcts[1:])
            pctTimes = list(self.algorithmList[i].litterCounts[-2])
            
            pctErrors = list(self.algorithmList[i].litterCounts[-1])
            pctErrors = np.array(pctErrors[1:])
            
            pctTimes = pctTimes[1:]

            # print(pctTimes)
            # print(pcts)
            ax.fill_between(pcts,pctTimes-pctErrors,pctTimes+pctErrors,alpha=0.3,color=self.colorList[i])
            ax.plot(pcts,pctTimes,label=self.algorithmNames[i],color=self.colorList[i])
            ax.scatter(pcts,pctTimes,color=self.colorList[i])
        # xmin,xmax = ax.get_xlim()
        ax.set_xlim(0,110)
        ax.set_ylim(ymin=0)
        ax.set_xlabel('Percentage of Litter Collected')
        ax.set_ylabel('Time in Seconds')
        llg = ax.legend(loc=9,bbox_to_anchor=(0.5,1.2),ncol=2)
        fig.savefig(self.plotsNdata+'litter_collected_pcts.pdf',bbox_extra_artists=(llg,),bbox_inches='tight')
        # plt.show()
            
    def plotSwarmTrajectory(self):
        '''Goes through each algorithm and plot trajectory of one simulation'''
        for i in self.algorithmList:
            sampleSim = self.algorithmList[i].simList[1]
            allSimTimes = glob.glob(self.file_path + sampleSim + '_m_4wrobot*')
            allX = []
            allY = []
            # print(allSimTimes)
            for j in allSimTimes:
                with open(j) as f:
                    data = f.read().rstrip('\n').split('\n')
                    for rowData in data[self.valid_result:]:
                        rowData = rowData.split(':')
                        d = rowData[1].split(',')
                        allX.append(float(d[1]))
                        allY.append(float(d[2]))
                        # print(d)
                        # input('>')
            f_robNax = plt.subplots()
            f_rob,ax_rob = f_robNax
            
            nb = np.linspace(-25,25,4)
            ax_rob.set_xlim(-25,25)
            ax_rob.set_ylim(-25,25)
            nbins = (nb,nb)
            print(len(allY))
            H,xedges,yedges = np.histogram2d(allX,allY,bins=nbins)
            H = np.rot90(H)
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
            a1=f_rob.colorbar(pcm,cax=cax)
            f_rob.tight_layout()
            f_rob.savefig(self.plotsNdata + 'robot_trajectory' + self.algorithmNames[i] +
                            '_' + sampleSim + '.pdf')



if __name__ == '__main__':
    valid_data = 0
    plotObj = MyPlotter(valid_data)
    plotObj.initSimulations()
    print(plotObj.algorithmList.keys())
    
    # drdID = input('which ID: ')
    # print(plotObj.algorithmList[drdID].simList)
    # drdSim = input('Which Sim time: ')
    # print(plotObj.algorithmList['0'].litterCounts)
    # ltnames = plotObj.algorithmList['50'].litterCounts.dtype.names

    # print(ltnames[1:])
    plotObj.processLitterData()
    plotObj.saveLitterDataXLS()
    plotObj.plotLitterData()
    plotObj.plotSwarmTrajectory()

    # print('\n\n\n\n')
    # print(plotObj.algorithmList['0'].litterCounts)
    # print(plotObj.algorithmList['50'].litterCounts['10'][1:])
    # # drdAlg = plotObj.algorithmList[drdID]
    # print(drdAlg.id)
    # print(drdAlg.simList)
    
    # paramNames = drdAlg.params.keys()

    # for p in paramNames:
    #     print(p,':',drdAlg.params[p])
    # litterFiles = plotObj.processLitterCount('2018-07-06--21-09-21')
    # for a,b in litterFiles:
    #     print(a,b)