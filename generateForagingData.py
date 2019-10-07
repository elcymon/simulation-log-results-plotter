# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from glob import glob
import ntpath

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def variedQsize(xcelFile):
    #investigate random walk vs repAtt qSize
    sheet='2Clstr-RepAt-Nois-v-q-Gud'
    resultMean = pd.read_excel(xcelFile,sheet_name=sheet,usecols="A:F")
    resultCI95 = pd.read_excel(xcelFile,sheet_name=sheet,usecols="I:M")
    resultCI95.index = resultMean.index
#    print(resultMean)
#    return resultMean
    # resultMean.plot(kind='bar')
    names = [ 'Rep-Att-Q1', 'Rep-Att-Q8'
            , 'Rep-Att-Q20', 'Rep-Att-Q40', 'Rep-Att-Q80', 'Rep-Att (Pure)','Random Walk']
    df = pd.DataFrame(columns=names)
    df.loc[0,'Rep-Att-Q1':'Rep-Att-Q80'] = list(resultMean[100][1:].values)
    df.loc[0,'Rep-Att (Pure)'] = resultMean.loc[1,0]
    df.loc[0,'Random Walk'] = resultMean.loc['RW',0]
    normalizeWith = df.loc[0,'Random Walk']
    df = df/ normalizeWith# normalize values
    
    dfCI = pd.DataFrame(columns=names)
    dfCI.loc[0,'Rep-Att-Q1':'Rep-Att-Q80'] = list(resultCI95[100][1:].values)
    dfCI.loc[0,'Rep-Att (Pure)'] = resultCI95.loc[1,0]
    dfCI.loc[0,'Random Walk'] = resultCI95.loc['RW',0]
    dfCI = dfCI/normalizeWith
    # df.plot(kind='bar')
    plotVariedQsize(df,dfCI,'2Clstr-RepAt-Nois-v-q-Gud')
    return df
def plotVariedQsize(df,dfCI,figName,yMax=0):
    colors = plt.cm.inferno(np.linspace(0,0.9,len(df)))#did not find tab20 cm on my windows workstation
        
        
    f = plt.figure()
    
    yerr=[]
    for i in dfCI.index:
        yerr.append([dfCI[i],dfCI[i]])
    print(len(yerr))
    
#    df.to_frame().T.plot(kind='bar',ax = f.gca(),color=colors,
#                      rot=0,width=1,figsize=(5,5))
    
    df.to_frame().T.plot(kind='bar',ax = f.gca(),color=colors,
                      rot=0,width=4,figsize=(5,5),yerr=dfCI)
    
#    legend = plt.legend(loc='center left',bbox_to_anchor=(1,0.5),title='Algorithms',fontsize=18)#.set_visible(True)
    #     legend.()
#    plt.setp(legend.get_title(),fontsize=18)
    plt.legend(fontsize=13)
    plt.ylabel('Normalized Foraging Time',fontsize=13,fontweight='bold')
    #     plt.xlabel('Litter Distributions',fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.xticks([])
    if yMax > 0:
        plt.ylim([0,yMax])
    #     plt.ylim([0, robot_data['turnP'].max() * 1.2])
    f.savefig(figName + '.pdf', bbox_inches='tight')#save before showing the title on notebook
    # plt.title(' '.join(title))
    # df['Rep-Att-Q1'] = resultMean
    # resultMean[100].T.plot(kind='bar')
    

def annotateBars(row,ax,yMax):
    for p in ax.patches:
        
        x,y = p.get_x() + p.get_width() / 2.0,p.get_height()
        ax.annotate('%.2f' % p.get_height(),(x,y),
                    ha='center',va='top',fontsize=16,fontweight='normal',color=p.get_facecolor(),rotation=90,xytext=(x,yMax), textcoords='data')
    
def plotNormalizedLinearComm(aMean,aCI95,figName,yMax,ncol=0):
    
    colors = plt.cm.inferno(np.linspace(0.2,0.9,len(aMean.columns)))#did not find tab20 cm on my windows workstation
    
    
    f = plt.figure()
    
    yerr=[]
    for i in aCI95.columns:
        yerr.append([aCI95[i].values,aCI95[i].values])
#    aMean.columns = [x.replace('ialization','').replace('antaneous','').\
#                     replace('0p','0.').replace('-1','-1.0') for x in aMean.columns]
    aMean.plot(kind='bar',ax = f.gca(),color=colors,
                      rot=0,width=0.9,figsize=(8,5),yerr=yerr,logy=False)
    
#    legend = plt.legend(loc='center left',bbox_to_anchor=(1,0.5),title='Algorithms',fontsize=13)
#     legend.()
#    plt.setp(legend.get_title(),fontsize=13,fontweight='bold')
    if ncol == 0:
        ncol = len(aMean.columns)
        loc = 'center left'
        anchor = (-0.03,1.05)
    else:
        loc = 'center right'
        anchor = (1.5,.50)
    plt.legend(fontsize=16,ncol=ncol,loc=loc,bbox_to_anchor=anchor)#.remove()
    plt.ylabel('Foraging Time',fontsize=16,fontweight='bold')
    plt.xlabel('')
    plt.ylim([0,yMax])
#     plt.xlabel('Litter Distributions',fontsize=18)
    plt.tick_params(axis='y', which='major', labelsize=17)
    plt.tick_params(axis='x',which='major',labelsize=17)
    aMean.apply(annotateBars,ax=f.gca(),yMax=yMax,axis=1)
    
#     plt.ylim([0, robot_data['turnP'].max() * 1.2])
    f.savefig(figName + '.pdf', bbox_inches='tight')#save before showing the title on notebook

def readNplotForagingData(xcelFile,sheet,meanCols='A:E',ci95cols='G:J',yMax=2000):
    forageMean = pd.read_excel(xcelFile,sheet_name=sheet,usecols=meanCols)
    forageCI95 = pd.read_excel(xcelFile,sheet_name=sheet,usecols=ci95cols)
    
    
    #normalize
    forageMean,forageCI95 = forageMean.div(forageMean['Random Walk'],axis=0),forageCI95.div(forageMean['Random Walk'].values,axis=0)
    plotNormalizedLinearComm(forageMean,forageCI95,'imported-rss19-results/values-' + sheet,yMax=yMax)

def populateDataFrame(df,data,simNo):
    for n in df.index:
        if n in data.iloc[:,-1].values:
            subData = data.loc[data.iloc[:,-1] == n,'time']
#            print(subData.shape,n)
            df.loc[n,simNo] = subData.iloc[0]
        elif data.iloc[0,-1] >= n:
            df.loc[n,simNo] = np.nan
        elif data.iloc[-1,-1] >= n:
            #value does not exist, so find mean of pevious and next 
            df.loc[n,simNo] = (data.loc[data.iloc[:,-1] < n,'time'].iloc[-1] + data.loc[data.iloc[:,-1] > n,'time'].iloc[0]) / 2.0
    return df
def createDefaultForagingDF(start=0,stop=201,step=1,ncols=30):
    df = pd.DataFrame(index=np.arange(start,stop,step,dtype=np.int),columns= ['%03d'% i for i in np.arange(1,ncols + 1,1)])
    df.index.name = 'NoOfLitter'
    df.columns.name = 'simNo'
    return df

def getALNS(world,resultType='pickupTime'):
    if resultType == 'pickupTime':
        category = 'pickedLitter'
        step = 1
    else:
        category = 'litterCount'
        step = 5
    meanDF = createDefaultForagingDF(step = step, ncols = 1)
    
    worldCSV = glob('alns/nestData/*' + world.lower() + '*.csv')[0]
    worldDF = pd.read_csv(worldCSV)
    worldDF = worldDF.astype(np.int)
#    print(worldDF.head(5))
    meanDF = populateDataFrame(meanDF,worldDF[['time',category]],meanDF.columns[0])
    meanDF.columns = ['ALNS']
    ci95DF = meanDF.mul(0)
    return meanDF,ci95DF
     
def readWorldData(resultsPath,worldName,algorithms = ['RepAtt-N0-Q1','RepAtt-N100-Q40','RW-0p0025P']):
    meanPickup,ci95Pickup = pd.DataFrame(columns=algorithms),pd.DataFrame(columns=algorithms)
    meanDeposit = meanPickup.copy()
    ci95Deposit = ci95Pickup.copy()
    
    with pd.ExcelWriter(f'{resultsPath}/{worldName}.xls', mode='a') as writer:
        for alg in algorithms:
            print('    ',alg)
            
            pickupTime = createDefaultForagingDF(step = 1, ncols = 30)
            
            depositTime = createDefaultForagingDF(step = 5, ncols = 30)
            
            for sim in glob(resultsPath + '/*' + worldName + '-*/*' + alg + '_*nestFile.csv'):
                simDF = pd.read_csv(sim)
                simFileName = ntpath.basename(sim)
                simNo = simFileName.split('_')[1]
               
                pickupTime = populateDataFrame(pickupTime,simDF[['time','pickedLitter']],simNo)
                depositTime = populateDataFrame(depositTime,simDF[['time','litterCount']],simNo)
                
                
            
            alg = alg.replace('RW-0p0025P-','')
            
            pickupTime.to_excel(writer,sheet_name=alg+'_pick')
            
            depositTime.to_excel(writer,sheet_name=alg+'_dep')
            
            pickupTime = pickupTime.dropna(how='all',axis=1)
            depositTime = depositTime.dropna(how='all',axis=1)
            
            meanPickup[alg] = pickupTime.mean(axis=1)
            meanDeposit[alg] = depositTime.mean(axis=1)
            
            
            ci95Pickup[alg] = pickupTime.std(axis=1).div(np.sqrt(pickupTime.shape[1])) * 1.96
            ci95Deposit[alg] = depositTime.std(axis=1).div(np.sqrt(depositTime.shape[1])) * 1.96
        return meanDeposit,ci95Deposit,meanPickup,ci95Pickup
def countDetectableBy(detectableBy):
    
    if detectableBy != None:
        return len(detectableBy.split(';'))
    else:
        return 0

     
def processLitterDetectionFiles(resultsPath,worlds,algorithms):
    summaryFileName = f'{resultsPath}/litterDetectionDetails'
    with pd.ExcelWriter(f'{summaryFileName}.xls', mode = 'a') as summary:
        for worldName in worlds:
            print(worldName)
            meanDets = pd.DataFrame(columns=algorithms,index=np.arange(1,201,1,dtype=np.int))
            ci95Dets = meanDets.copy()
            meanDetBy = meanDets.copy()
            ci95DetBy = meanDets.copy()
            
            detsBySummary = pd.DataFrame(columns=algorithms,index=np.arange(0,37,1,dtype=np.int))
            detsSummary = pd.DataFrame(columns=algorithms,index=['min','max','zeros'])
            
            with pd.ExcelWriter(f'{resultsPath}/{worldName}_litterDetectionDetails.xls', mode='a') as writer:
                for alg in algorithms:
                    print('\t',alg)
                    numberOfDetectionsDF = createDefaultForagingDF(start=1,step=1)
                    detectableByDF = createDefaultForagingDF(start=1,step=1)
                    
                    for sim in glob(resultsPath + '/*' + worldName + '-*/*' + alg + '_*litterDetectionDetails.csv'):
                        
                        simDF = pd.read_csv(sim)
                        simDF.index = [int(i.replace('m_litter','')) for i in simDF['name']]
                        simFileName = ntpath.basename(sim)
                        simNo = simFileName.split('_')[1]
                        numberOfDetectionsDF[simNo] = simDF['numberOfDetections']
        #                        print(simDF['detectableBy'].apply(lambda x: len(x.split(';'))))
                        
                        detectableByDF[simNo] = simDF['detectableBy'].astype(str).apply(countDetectableBy)
                    #alg_sheet = alg#.replace('aneous','').replace('ialization','')
                    algSheet = alg.replace('RW-0p0025P-','')
                    numberOfDetectionsDF.to_excel(writer,sheet_name=algSheet+'_numD')
                    detectableByDF.to_excel(writer,sheet_name=algSheet+'_dBy')
                    numberOfDetectionsDF = numberOfDetectionsDF.dropna(how='all',axis=1)
                    detectableByDF = detectableByDF.dropna(how='all',axis=1)
                    
                    meanDets[alg] = numberOfDetectionsDF.mean(axis=1)
                    ci95Dets[alg] = numberOfDetectionsDF.std(axis=1).div(np.sqrt(numberOfDetectionsDF.shape[1])) * 1.96
                    
                    meanDetBy[alg] = detectableByDF.mean(axis=1)
                    ci95DetBy[alg] = detectableByDF.std(axis=1).div(np.sqrt(detectableByDF.shape[1])) * 1.96
                    
                    for i in detsBySummary.index:
                        detsBySummary.loc[i,alg] = (detectableByDF == i).sum().mean()
                    
                    detsSummary.loc['min',alg] = numberOfDetectionsDF.min().min()
                    detsSummary.loc['max',alg] = numberOfDetectionsDF.max().max()
                    detsSummary.loc['zeros',alg] = (numberOfDetectionsDF == 0).sum().sum()
                    
            
            plotLitterDetectionsDetails(figName=f'{summaryFileName}_{worldName}_detBy',
                                        xlabel='Litter ID', ylabel='Number of Robots',
                                        dfData=meanDetBy, dfci95=ci95DetBy, kind = 'line')
            meanDetBy.to_excel(summary,sheet_name=worldName + '_detBy')
            ci95DetBy.to_excel(summary,sheet_name=worldName + '_detBy',startcol=ci95DetBy.shape[1] + 2)
            markers = ['.','*','+','x','_','^',
                       'o','v','s','>','<','4','d','|']
            plotLitterDetectionsDetails(figName=f'{summaryFileName}_{worldName}_numDets',
                                        xlabel='Litter ID', ylabel='Number of Time Steps',
                                        dfData=meanDets, dfci95=ci95Dets, kind = 'scatter',
                                        alpha=0.8,markers=markers,logy=True,markersize=30)
            
            meanDets.to_excel(summary,sheet_name=worldName + '_numDets')
            ci95Dets.to_excel(summary,sheet_name=worldName + '_numDets',startcol=ci95Dets.shape[1] + 2)
    #        
            plotLitterDetectionsDetails(figName=f'{summaryFileName}_{worldName}_detsSummary_min',
                                        xlabel='', ylabel='Number of Time Steps',
                                        dfData=pd.DataFrame(detsSummary.loc['min',:]).T, kind = 'scatter',
                                        alpha=1,markers=markers,logy=False,
                                        markersize=50)
            plotLitterDetectionsDetails(figName=f'{summaryFileName}_{worldName}_detsSummary_max',
                                        xlabel='', ylabel='Number of Time Steps',
                                        dfData=pd.DataFrame(detsSummary.loc['max',:]).T, kind = 'scatter',
                                        alpha=1,markers=markers,logy=False,
                                        markersize=50)
            plotLitterDetectionsDetails(figName=f'{summaryFileName}_{worldName}_detsSummary_zeros',
                                        xlabel='', ylabel='Number of Litter',
                                        dfData=pd.DataFrame(detsSummary.loc['zeros',:]).T, kind = 'scatter',
                                        alpha=1,markers=markers,logy=False,
                                        markersize=50)
            detsSummary.to_excel(summary,sheet_name=worldName + '_detsSummary')
            plotLitterDetectionsDetails(figName=f'{summaryFileName}_{worldName}_detBySummary',
                                        xlabel='Number of Robots', ylabel='Number of Litter',
                                        dfData=detsBySummary, kind = 'scatter',
                                        alpha=1,markers=markers,logy=False,
                                        markersize=30)
            detsBySummary.to_excel(summary,sheet_name=worldName + '_detsBySummary')

def plotLitterDetectionsDetails(figName, xlabel, ylabel, dfData, logy=False, xticks=[], dfci95=[], kind='scatter',markers=[],alpha=0,markersize=1):
    f = plt.figure()
    axis = f.gca()
    colors = plt.cm.inferno(np.linspace(0,0.9,len(dfData.columns)))
    for i, alg in enumerate(dfData.columns):
        label = alg.replace('RW-0p0025P-','')#alg.replace('0p','0.').replace('-1','-1.0').replace('ialization','').replace('antaneous','')
        if len(dfci95) != 0 and kind == 'line':
            axis.fill_between(dfData.index, dfData[alg] - dfci95[alg], dfData[alg] + dfci95[alg], alpha=0.5,color=colors[i])
        
        if kind == 'line':
            axis.plot(dfData.index,dfData[alg],color=colors[i],label=label,alpha=0.5)
        elif kind == 'scatter':
            axis.scatter(dfData.index,dfData[alg],color=colors[i],label=label,marker=markers[i],alpha=alpha,s=markersize)
            if logy:
                axis.set_yscale('log')
    plt.xlabel(xlabel,fontsize=13,fontweight='bold')
    plt.ylabel(ylabel,fontsize=13,fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=12)
    axis.legend(fontsize=13,ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.25))
    f.savefig(figName + '.pdf',bbox_inches='tight')
#    plt.close()
                    
def summarizeSimulationResults(resultsPath,worlds,algorithms = ['RepAtt-N0-Q1','RepAtt-N100-Q40','RW-0p0025P']):
    with pd.ExcelWriter(f'{resultsPath}/simulationSummary.xls',mode = 'a') as summary:
        for worldName in worlds:
            print(worldName)
            meanDeposit,ci95Deposit,meanPickup,ci95Pickup = readWorldData(resultsPath,worldName,algorithms)
            #print(meanDeposit.shape,ci95Deposit.shape,meanPickup.shape,ci95Pickup.shape)
            meanDeposit.to_excel(summary,sheet_name=worldName+'_dep')
            ci95Deposit.to_excel(summary,sheet_name=worldName+'_dep',startcol=meanDeposit.shape[1] + 2)
            
            meanPickup.to_excel(summary,sheet_name=worldName+'_pick')
            ci95Pickup.to_excel(summary,sheet_name=worldName+'_pick',startcol=meanPickup.shape[1] + 2)
def plotForagingProgression(mean,ci95,figName):
    f = plt.figure()
    ax = f.gca()
    colors = plt.cm.inferno(np.linspace(0.2,0.9,len(mean.columns)))
#    print(colors[0])
#    print(mean)
    for i,alg in enumerate(mean.columns):
        ax.plot(mean.index,mean[alg],color=colors[i],label=alg)
        if alg != 'ALNS':
            ax.fill_between(mean.index, mean[alg] - ci95[alg], mean[alg] + ci95[alg], alpha=0.4,color=colors[i])
    plt.xlabel('Number of Targets',fontsize=13,fontweight='bold')
    plt.ylabel('Time in seconds',fontsize=13,fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=13)
    
    f.savefig(figName + '.pdf',bbox_inches='tight')
    plt.close()
def importICRAresultsfromSummaryXLS(resultsPath,summaryXLS,
                                    summaryAlgorithms=['RW-0p0025P','RepAtt-N0-Q1','RepAtt-N100-Q40'],
                                    algorithmsDictionary=[],norm_with=None,worlds=None,
                                    worldSize='50m',resultType='depositTime',numTargets=180,yMax=2000):
    if len(algorithmsDictionary) == 0:
        algorithms = summaryAlgorithms + ['ALNS']
    else:
        algorithms=[algorithmsDictionary[alg] for alg in summaryAlgorithms] + ['ALNS']
    
    #worlds=['OneCluster','TwoClusters','FourClusters','HalfCluster','Uniform']
    mean = pd.DataFrame(columns=algorithms,index=worlds)
    ci95 = mean.copy()
    
    #import desired data from summaryXLS
    summaryXLSfile = pd.ExcelFile(resultsPath + '/' + summaryXLS)
    
    for sheet_name in summaryXLSfile.sheet_names:  # see all sheet names
#        print(sheet_name)
        world,resultCategory = sheet_name.split('_')
        if worldSize == '100m':#differentiate 100m from 50m world
            sz = worldSize in world
        else:
            sz = '100m' not in world
            
        if resultCategory == resultType and sz:
            print(world,resultCategory)
            alnsMean,alnsci95 = getALNS(world.replace(worldSize,'').split('Cluster')[0],'depositTime')
            
            summaryMean = summaryXLSfile.parse(sheet_name, index_col = 0, usecols='A:G')  # read a specific sheet to DataFrame
            summaryMean['ALNS'] = alnsMean.values
            summaryCI95 = summaryXLSfile.parse(sheet_name, index_col = 0, usecols='I:O')
            summaryCI95['ALNS'] = alnsci95.values
            
            
            mean.loc[world.replace(worldSize,''),:] = summaryMean.loc[numTargets,summaryAlgorithms + ['ALNS']].values
            ci95.loc[world.replace(worldSize,''),:] = summaryCI95.loc[numTargets,summaryAlgorithms + ['ALNS']].values
            
            if resultType == 'pick':
                figName = resultsPath + '/plots/linePlot_' + f'{world}_{resultType}_{numTargets}'
                columns = summaryAlgorithms + ['ALNS']
                sM = summaryMean.loc[:,columns].copy()
                sM.columns = [x.replace('ialization','').replace('antaneous','').\
                     replace('0p','0.').replace('-1','-1.0') for x in algorithms]
                sCI95 = summaryCI95.loc[:,columns].copy()
                sCI95.columns = sM.columns
                plotForagingProgression(sM,sCI95,figName)
            
#            mean.loc[world.replace(worldSize,''),'ALNS'] = alnsMean.loc[numTargets,'ALNS']
#            ci95.loc[world.replace(worldSize,''),'ALNS'] = alnsci95.loc[numTargets,'ALNS']
    if norm_with != None:
        mean,ci95 = mean.div(mean[norm_with],axis=0),ci95.div(mean[norm_with].values,axis=0)    
    plotNormalizedLinearComm(mean,ci95,resultsPath + '/plots/' + f'rw-norm-{worldSize}_{resultType}_{numTargets}',yMax=yMax,ncol=1)
    return mean,ci95
def generateLatexTable(filePath,summaryXLS,numTargets=180):
    
    #import desired data from summaryXLS
    summaryXLSfile = pd.ExcelFile(resultsPath + '/' + summaryXLS)
    for sheet_name in summaryXLSfile.sheet_names:
        meanCI95 =  pd.DataFrame(columns=['0','0.25','1'])
        world,resultCategory = sheet_name.split('_')
        
        summaryMean = summaryXLSfile.parse(sheet_name, index_col = 0, usecols='A:O')  # read a specific sheet to DataFrame
        
        summaryCI95 = summaryXLSfile.parse(sheet_name, index_col = 0, usecols='Q:AE')
        for i in summaryMean.columns:
            alg,detDur = i.split('-detDur-')
            detDur = detDur.replace('p','.').replace('00','0.')
            m = summaryMean.loc[numTargets,i]
            c = summaryCI95.loc[numTargets,i]
            meanCI95.loc[alg,detDur] = '${:.2f} \pm {:.2f}$'.format(m,c)
        meanCI95.to_latex(filePath + '/' + sheet_name + '.tex',escape=False)
        
        
if __name__ == '__main__':
    algorithmsDict = []#{'RW-0p0025P': 'Random Walk', 'RepAtt-N0-Q1': 'Rep-Att (Pure)', 'RepAtt-N100-Q40': 'Rep-Att-Q40',
                     #'RepAtt-N100-Q80':'RepAtt-Q80-slide','RepAtt-N10-Q10':'RepAtt-N10-Q10','RepAtt-N50-Q20':'RepAtt-N50-Q20'}
    resultsPath = '/home/elcymon/containers/swarm_sim/results/visionModel-detRate'#'simulation-results/free_space_attract/q1-120'

    
    worlds = ['Uniform']#['OneCluster','TwoClusters','FourClusters','HalfCluster','Uniform',\
              #'OneCluster100m','TwoClusters100m','FourClusters100m','HalfCluster100m','Uniform100m']
    csvAlgorithms = ['inst-1-detDur-0',
                    'init-1-detDur-0',
                    'inst-0p5-detDur-1',
                    'init-0p5-detDur-1',
                    'inst-0p2-detDur-0p25',
                    'init-0p2-detDur-0p25',
                    'inst-1-detDur-1',
                    'init-1-detDur-1',
                    'inst-1-detDur-0p25',
                    'init-1-detDur-0025',
                    'inst-0p5-detDur-0p25',
                    'init-0p5-detDur-0p25',
                    'inst-0p2-detDur-1',
                    'init-0p2-detDur-1']
#                    ['initialization-0p2', 'instantaneous-0p2',
#                     'initialization-0p5', 'instantaneous-0p5',
#                     'initialization-1', 'instantaneous-1']
        #['N100-Q1','N100-Q8','N100-Q20','N100-Q80','N100-Q120']
            #['RepAtt-N0-Q1','RepAtt-N100-Q40','RW-0p0025P']#['RepAtt-N10-Q10','RepAtt-N50-Q20']#['RepAtt-N100-Q80']
#    sdf = processLitterDetectionFiles(resultsPath,worlds,csvAlgorithms)
#    summarizeSimulationResults(resultsPath,worlds,csvAlgorithms)
    generateLatexTable(resultsPath,'simulationSummary.xls')
#    mean,ci95 = importICRAresultsfromSummaryXLS(resultsPath,'simulationSummary.xls',
#                                                summaryAlgorithms=csvAlgorithms,
#                                                algorithmsDictionary=algorithmsDict,worlds=worlds,
#                                                worldSize='50m',yMax=3600,resultType='pick')
#    xcelFile='imported-rss19-results/icra2020.xlsx'
#    sheet='sound-model-100m'
#    readNplotForagingData(xcelFile,sheet,meanCols='A:E',ci95cols='G:J',yMax=19000)
