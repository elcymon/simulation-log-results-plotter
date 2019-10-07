#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:42:07 2019

@author: elcymon
"""

import pandas as pd
import generateForagingData

def getALNSData(world):
    mean,ci95 = generateForagingData.getALNS(world.replace('100m','').split('Cluster')[0])
#    print(mean)
    m180,ci95180 = mean.loc[180,'ALNS'],ci95.loc[180,'ALNS']
    return m180,ci95180
def getSummaryXLSData(world,fileName):
    sheet_name = world + '_pickupTime'
    mean = pd.read_excel(fileName,sheet_name, index_col = 0, usecols='A:D')  # read a specific sheet to DataFrame
    ci95 = pd.read_excel(fileName,sheet_name, index_col = 0, usecols='F:I')
    
    return mean.loc[180,:],ci95.loc[180,:]
    
def getColumnsData(folderName,worlds):
    pass
def variedQsizeAnalysis(dataMean,dataCI95,summaryXLS='simulation-results/free_space_attract/q1-120/simulationSummary.xls'):
    dfMean = pd.read_excel(summaryXLS,'TwoClusters_pickupTime',index_col=0,usecols='A:F')
    dfCI95 = pd.read_excel(summaryXLS,'TwoClusters_pickupTime',index_col=0,usecols='H:M')
    dataMean = dataMean.append(dfMean.loc[180,:])
    dataCI95 = dataCI95.append(dfCI95.loc[180,:])
    dataMean['N100-Q1'] = dfMean['N100-Q1'].dropna().iloc[-1]
    dataCI95['N100-Q1'] = dfCI95['N100-Q1'].dropna().iloc[-1]
    dataMean,dataCI95 = dataMean.div(dataMean['Random Walk']),dataCI95.div(dataMean['Random Walk'])
    dsort = ['N100-Q1','N100-Q8','N100-Q20','N100-Q40','N100-Q80','N100-Q120','RepAtt (Pure)','Random Walk']
    generateForagingData.plotVariedQsize(dataMean[dsort],dataCI95[dsort],'../images/TwoClusters_q1-120',yMax=10)
    
if __name__ == '__main__' and False:
    rowColHeaders = pd.read_excel('simulation-results/summaryTable/tableHeader.xls',sheet_name='tableHeader',
                                  header=[0,1,2,3])
    data = rowColHeaders.astype(str)
    simFolder='simulation-results/'
    summaryDictionary = {'c1':'alns','c2-c4':f'{simFolder}cap_max_attract/simulationSummary.xls',
                         'c5-c6':f'{simFolder}free_space_attract/simulationSummary.xls',
                         'c7':f'{simFolder}free_space_attract/sliding_window/simulationSummary.xls'
                         }
    noiseDataRaw = 'simulation-results/free_space_attract/noise_10pct-50pct/simulationSummary.xls'
    
    noiseData = pd.read_excel('simulation-results/summaryTable/tableHeader.xls',sheet_name='noiseHeader',
                                  header=[0,1])
    noiseData = noiseData.astype(str)
    icraResultMean = pd.DataFrame(columns = ['Random Walk','RepAtt (Pure)','N10-Q10','N100-Q40','ALNS'],
                                                index = noiseData.index)
    icraResultci95 = icraResultMean.copy()
    
    for world in rowColHeaders.index:
        alnsMean,alnsci95 = getALNSData(world)
        print(world)
        data.loc[world,].iloc[0] = '${} \pm {}$'.format(alnsMean,alnsci95);
        
        colsMean,colsci95 = getSummaryXLSData(world,summaryDictionary['c2-c4'])
        data.loc[world,].iloc[1] = '${0:.1f} \pm {1:.2f}$'.format(colsMean['RW-0p0025P'],colsci95['RW-0p0025P'])
        data.loc[world,].iloc[2] = '${0:.1f} \pm {1:.2f}$'.format(colsMean['RepAtt-N0-Q1'],colsci95['RepAtt-N0-Q1'])
        data.loc[world,].iloc[3] = '${0:.1f} \pm {1:.2f}$'.format(colsMean['RepAtt-N100-Q40'],colsci95['RepAtt-N100-Q40'])
        
        colsMean,colsci95 = getSummaryXLSData(world,summaryDictionary['c5-c6'])
        data.loc[world,].iloc[4] = '${0:.1f} \pm {1:.2f}$'.format(colsMean['RepAtt-N0-Q1'],colsci95['RepAtt-N0-Q1'])
        data.loc[world,].iloc[5] = '${0:.1f} \pm {1:.2f}$'.format(colsMean['RepAtt-N100-Q40'],colsci95['RepAtt-N100-Q40'])
        
        c7Mean,c7ci95 = getSummaryXLSData(world,summaryDictionary['c7'])
        data.loc[world,].iloc[6] = '${0:.1f} \pm {1:.2f}$'.format(c7Mean['RepAtt-N100-Q80'],c7ci95['RepAtt-N100-Q80'])
        
        n10n50Mean,n10n50ci95 = getSummaryXLSData(world,noiseDataRaw)
        noiseData.loc[world].iloc[0:3] = data.loc[world].iloc[0:3].values
        noiseData.loc[world].iloc[3] = '${0:.1f} \pm {1:.2f}$'.format(n10n50Mean['RepAtt-N10-Q10'],n10n50ci95['RepAtt-N10-Q10'])
        noiseData.loc[world].iloc[4] = '${0:.1f} \pm {1:.2f}$'.format(n10n50Mean['RepAtt-N50-Q20'],n10n50ci95['RepAtt-N50-Q20'])
        noiseData.loc[world].iloc[-1] = data.loc[world].iloc[5]
        
        icraResultMean.loc[world,:] = list(colsMean.loc[['RW-0p0025P','RepAtt-N0-Q1']]) + [n10n50Mean['RepAtt-N10-Q10'], colsMean['RepAtt-N100-Q40'],alnsMean]
        icraResultci95.loc[world,:] = list(colsci95.loc[['RW-0p0025P','RepAtt-N0-Q1']]) + [n10n50ci95['RepAtt-N10-Q10'], colsci95['RepAtt-N100-Q40'],alnsci95]
        
    data.to_latex('simulation-results/summaryTable/algorithmsSummary.tex',escape=False)
    noiseData.to_latex('simulation-results/summaryTable/noiseSummary_free_space_attract.tex',escape=False)
    
    index = ['One', 'Two', 'Four', 'Half', 'Uniform']
    rw_norm_50m_Mean = icraResultMean.loc['OneCluster':'Uniform',:].copy()
    rw_norm_50m_ci95 = icraResultci95.loc['OneCluster':'Uniform',:].copy()
    
    rw_norm_50m_Mean,rw_norm_50m_ci95 = rw_norm_50m_Mean.div(rw_norm_50m_Mean['Random Walk'],axis=0),\
        rw_norm_50m_ci95.div(rw_norm_50m_Mean['Random Walk'],axis=0)
    rw_norm_50m_ci95.index = index
    rw_norm_50m_Mean.index = index
        
#    generateForagingData.plotNormalizedLinearComm(rw_norm_50m_Mean,rw_norm_50m_ci95,'../images/rw-norm-50m_pickupTime_180',yMax=1.3,ncol=1)
    rw_norm_100m_Mean = icraResultMean.loc['OneCluster100m':'Uniform100m',:].copy()
    rw_norm_100m_Mean.index = index#icraResultMean.loc['OneCluster':'Uniform',:].index
    
    rw_norm_100m_ci95 = icraResultci95.loc['OneCluster100m':'Uniform100m',:].copy()
    rw_norm_100m_ci95.index = index#icraResultMean.loc['OneCluster':'Uniform',:].index
    
    rw_norm_100m_Mean,rw_norm_100m_ci95 = rw_norm_100m_Mean.div(rw_norm_100m_Mean['Random Walk'],axis=0),\
        rw_norm_100m_ci95.div(rw_norm_100m_Mean['Random Walk'],axis=0)
        
#    generateForagingData.plotNormalizedLinearComm(rw_norm_100m_Mean,rw_norm_100m_ci95,'../images/rw-norm-100m_pickupTime_180',yMax=1.3,ncol=1)
        
if __name__ == '__main__':
    dataMean = variedQsizeAnalysis(icraResultMean.loc['TwoClusters',['Random Walk','RepAtt (Pure)','N100-Q40']],
                       icraResultci95.loc['TwoClusters',['Random Walk','RepAtt (Pure)','N100-Q40']])
    