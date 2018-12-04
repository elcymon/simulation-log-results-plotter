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
import sys
import pandas as pd

class Algorithm:
    '''This class maps an algorithm ID to experiment time, and sets all algorithm parameters'''
    def __init__(self,id):
        self.id = id
        self.name = ''
        self.simList = []
        self.params = {}
        self.myColor = ''
        names = ['Simulation times']
        names.extend([str(i) for i in range(0,101,10)])

        formats = ['f4']*11
        formats.insert(0,'|U20')
        
        self.dat_dtype = np.dtype({'names':names,'formats':formats})
        
        self.litterCounts = pd.DataFrame(columns=range(0,101,10)) # each column should represent the time taken to forage 10pct of objects
        self.wall_bounces = pd.DataFrame(columns=range(0,101,10))#self.setup_Matrix('wall_bounces')
        

    def setup_Matrix(self,record_type):
        tHeader = [record_type]
        tHeader.extend(range(0,101,10))
        paramVar = pd.DataFrame(columns=tHeader)
        # self.dat_dtype = np.dtype({'names':names,'formats':formats})
        paramVar = np.zeros(1,self.dat_dtype)#30 to represent iteration counts, but can change automatically.
        paramVar[0] = np.array(tuple(tHeader),self.dat_dtype)
        return paramVar

    def append_Counts(self,sim_lit_data,paramVar):
        #print(len(self.litterCounts[0]),len(sim_lit_data))
        # print(sim_lit_data)
        # print(self.dat_dtype)
        while len(sim_lit_data) < len(self.dat_dtype):
            '''padding with invalid data if data incomplete'''
            sim_lit_data = np.append(sim_lit_data,np.nan)
        np_sim_lit_data = np.array(tuple(sim_lit_data),self.dat_dtype)
        paramVar = np.r_[paramVar,[np_sim_lit_data]]
        return paramVar
        


class MyPlotter:
    '''Class for plotting/visualizing my simulation results'''
    def __init__(self,valid_result,createReadme):
        self.createReadme = createReadme
        self.algorithmList = {}
        self.file_path = ''
        self.plotsNdata = ''
        self.readme = ''
        self.valid_result = valid_result #row to start reading data from
        self.maxLitter = 200
        self.litStep = self.maxLitter * 10 / 100.0
        self.colorList = {(0.50196078431,0,0), (0.66666666666,0.43137254902,0.15686274509),(0,0,1),(0,0,0.5),
        (0.50196078431,0.50196078431,0),(0.23529411764,0.70588235294,0.29411764705),(0.27450980392,0.94117647058,0.94117647058),
        (0,0.50980392156,0.78431372549),(0,0,0.50196078431),(0.94117647058,0.19607843137,0.90196078431),
        (0.98039215686,0.74509803921,0.74509803921),(0,0,0), (0.7,0.7,0.7)}

        


        
        self.count = 0

        # print('plotter created')
        # mpl.style.use('seaborn-colorblind')

    def initAlgorithm(self,id,d):
        
        self.algorithmList[id] = Algorithm(id)
        # self.algorithmList[id].setup_Matrix(self.algorithmList[id].litterCounts)#initialize setup matrix
        # print('length',len(self.algorithmList))
        # self.algorithmList[id].myColor = self.colorList[id]
        self.appendSimTime(id,d)
        
        for i in d:
            param = i.split(':')
            self.algorithmList[id].params[param[0]] = param[1]
    
    def appendSimTime(self,id,d):
        stat = d[0]
        # print(stat)
        stat = stat.split(':')
        start_time = stat[1]
        self.algorithmList[id].simList.append(start_time)
        
        sim_lit_data = self.getLitterCount(start_time)
        while len(sim_lit_data[1:]) < len(self.algorithmList[id].litterCounts.columns):
            sim_lit_data.append(np.nan)
        self.algorithmList[id].litterCounts.loc[sim_lit_data[0]] = sim_lit_data[1:]

        # self.algorithmList[id].litterCounts = self.algorithmList[id].append_Counts(sim_lit_data,self.algorithmList[id].litterCounts)

        

        
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

    
    def initSimulations(self,file_path=''):
        '''This function selects the folder of the results and matches each 
        simulation times to their respective algorithms'''
        if file_path == '':
            root = tk.Tk()
            root.withdraw()
            self.file_path = filedialog.askdirectory()
        else:
            self.file_path = file_path

        self.file_path +=  '/'
        self.plotsNdata = self.file_path + 'plotsNdata/'
        pathlib.Path(self.plotsNdata).mkdir(parents=False,exist_ok=True)
        readmeFile = self.file_path + 'readme.md'
        if self.createReadme == 'createReadme':
            readmeFile = self.file_path + 'readme2.md'
            #if readme file does not exist, create a readme.md file
            with open(readmeFile,'w+') as f:
                for name in glob.glob(self.file_path + '*litter*'):
                    name = name.replace('\\','/')
                    name = name.split('/')
                    name = name[-1].split('_')
                    prefix = name[0]
                    algID = prefix.split('-')
                    algID = '-'.join(algID[1:-7])
                    lineData = 'prefix:{},ID:{}\n'.format(prefix,algID)
                    f.write(lineData)

            
                
        
        with open(readmeFile) as f:
            for data in f:
                data = data.rstrip('\n')
                d = data.split(',')
                if len(d) > 1:
                    exp_id = d[1]
                    exp_id = exp_id.split(':')
                    if exp_id[1] in self.algorithmList:
                        self.appendSimTime(exp_id[1],d)
                    else:
                        #new algorithm found
                        self.initAlgorithm(exp_id[1],d)
        
    def processParamData(self):
        ## Call function that will process information for particular logged information from the robot files
        countAlg = len(self.algorithmList)
        progAlg = 0
        for i in self.algorithmList:
            progAlg += 1
            # print(i)
            litCounts = self.algorithmList[i].litterCounts
            # litCounts = pd.DataFrame(litCounts)

            wall_bounces_all = pd.DataFrame(columns=self.algorithmList[i].wall_bounces.columns)
            countSmList = len(self.algorithmList[i].simList)
            progSim = 0
            for start_time in self.algorithmList[i].simList:
                progSim += 1
                sim_data = litCounts.loc[start_time]
                # print(start_time)
                # print(sim_data)
                # print(sim_data.iloc[0,1:])
                # print(sim_data.loc[:,1:].values())
                # input('>')
                # # 
                wall_bounces_data = self.get_param_values(start_time,sim_data,'wall_bounces')
                wall_bounces_all.loc[start_time] = wall_bounces_data
                # wall_bounces_all = self.algorithmList[i].append_Counts(wall_bounces_data,wall_bounces_all)
                
                print('\r{}/{}: {}/{}'.format(progAlg,countAlg,progSim,countSmList),end='')
            # bounce_data = wall_bounces_all[1:]
            # bounces_mean = ['mean']
            # print(bounce_data)
            # bounces_mean.append(np.nanmean(bounce_data))
            bounces_mean = np.nanmean(wall_bounces_all,axis=0)
            bounces_std  = np.nanstd(wall_bounces_all,axis=0)
            bounces_CI95 = 1.96*bounces_std/np.sqrt(wall_bounces_all.count(axis=0))
            # print(len(bounces_mean),len(self.algorithmList[i].wall_bounces.columns))
            self.algorithmList[i].wall_bounces.loc['mean'] = bounces_mean
            self.algorithmList[i].wall_bounces.loc['CI95'] = bounces_CI95
            # print()
            # print('bouncesMean\n',bounces_mean)
            # print('stored value in algirthm list\n',self.algorithmList[i].wall_bounces)

        
            
        
    def get_param_values(self,start_time,sim_lit_data,param_column):
        sim_lit_data = pd.Series(sim_lit_data)
        # print(sim_lit_data)
        total_param_value = np.zeros_like(sim_lit_data)
        # print(total_param_value)
        # input('>')
        for name in glob.glob(self.file_path + start_time + '*robot*'):
            robot_data = self.get_robot_data(name)
            param_list = robot_data.loc[robot_data['time'].isin(sim_lit_data)]
            # print(total_param_value)
            # print(param_list[param_column])
            # print()
            total_param_value += param_list[param_column]
        return total_param_value



    def processLitterData(self):
        for i in self.algorithmList:
            pctNames = self.algorithmList[i].litterCounts.columns
            pctMean = ['mean']
            confidenceInterval = ['95% Confidence Interval']
            for j in pctNames:
                # print(self.algorithmList[i].litterCounts[j][1:])
                litData = self.algorithmList[i].litterCounts[j]
                pctMean.append(np.nanmean(litData))
                stDev = np.nanstd(litData)
                confidenceInterval.append(1.96*stDev/np.sqrt(litData.count()))
            # if pctMean[0] not in self.algorithmList[i].litterCounts.index:
            self.algorithmList[i].litterCounts.loc[pctMean[0]] = pctMean[1:]
            self.algorithmList[i].litterCounts.loc[confidenceInterval[0]] = confidenceInterval[1:]

    def get_robot_data(self,filename):

        robot_data = pd.read_csv(filename,
                             header=None, sep=':|,',engine='python',
                             names=['robot','time','x','y','theta','turnP',
                                    'seen_litter','rep_neigh','rep_sig',
                                   'call_neigh','call_sig','collected_lit',
                                   'linear_dist','rot_dist','litter_collected','litter_deposited',
                                    'wall_bounces', 'neighbour_bounces','t_obstacle_avoidance',
                                    't_searching','t_oa_searching','t_go4litter','t_oa_go4litter',
                                    't_litter_processing','t_homing','t_oa_homing','action','state'])
        return robot_data

    def saveLitterDataXLS(self):
        wb = xlwt.Workbook()
        ws = wb.add_sheet('Litter Data')
        row = 0
        ws.write(row,0,'MEAN')
        row += 1
        algkeys = list(self.algorithmList.keys())
        algkeys.sort()
        for i in algkeys:
            pctNames = self.algorithmList[i].litterCounts.columns
            
            if row == 1:
                for ii,pct in enumerate(pctNames):
                    if ii == 0:
                        ws.write(row,ii,'ID')
                    else:
                        ws.write(row,ii,pct)
                row += 1
            for jj,pct in enumerate(pctNames):
                if jj == 0:
                    ws.write(row,jj,i)
                else:
                    ws.write(row,jj,str(self.algorithmList[i].litterCounts[pct][-2]))
            row += 1
        row += 3
        ws.write(row,0,'95% Confidence Interval')
        row +=1
        for i in algkeys:
            for jj,pct in enumerate(pctNames):
                if jj == 0:
                    ws.write(row,jj,i)
                else:
                    ws.write(row,jj,str(self.algorithmList[i].litterCounts[pct][-1]))
            row += 1
        wb.save(self.plotsNdata+'Results.xls')

    def plotLitterData(self):
        fig = plt.figure()
        ax = plt.axes()
        algkeys = list(self.algorithmList.keys())
        algkeys.sort()
        for i in algkeys:
            pcts = self.algorithmList[i].litterCounts.columns
            
            pcts = np.array(pcts)
            pctTimes = list(self.algorithmList[i].litterCounts.loc[-2])
            
            pctErrors = list(self.algorithmList[i].litterCounts.loc[-1])
            pctErrors = np.array(pctErrors)
            
            pctTimes = pctTimes

            # print(pctTimes)
            # print(pcts)
            ax.fill_between(pcts,pctTimes-pctErrors,pctTimes+pctErrors,alpha=0.3)#,color=self.colorList[i])
            ax.plot(pcts,pctTimes,label=i)#,color=self.colorList[i])
            ax.scatter(pcts,pctTimes)#,color=self.colorList[i])
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
            f_rob.savefig(self.plotsNdata + 'robot_trajectory'
                            '_' + sampleSim + '.pdf')



if __name__ == '__main__':
    valid_data = 0
    if len(sys.argv) == 2:
        plotObj = MyPlotter(valid_data,sys.argv[1])
    elif len(sys.argv) == 1:
        plotObj = MyPlotter(valid_data,'oldReadme')
    else:
        print('invalid call')
        exit(0)
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