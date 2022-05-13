# intended for use in anaconda - spyder ide
# click "run file" (or press F5) once
# proceed to line 520

##########
# import #
##########

import os
import numpy as np
import pandas as pd
import scipy as sp
import itertools as itert
import matplotlib.pyplot as plt
import imageio
from sklearn.model_selection import train_test_split
from scipy.stats.stats import spearmanr #correlation
# from scipy.stats.stats import pearsonr #correlation
import statsmodels.api as sm #OLS / WLS
from sklearn.ensemble import RandomForestRegressor #RF
import seaborn as sns;# sns.set(); sns.set_style('whitegrid')
# from numpy.polynomial.polynomial import polyfit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

####################
# custom functions #
####################

#Import csv files to python
def impcsv(path,s=None):
    if s == None:
        data = np.array(pd.read_csv(path, sep=';',header=None)[1:].drop(0,axis=1)).astype(np.float)
    else:
        data = pd.read_csv(path, sep=';')
        data = data.loc[:,data.columns.str.startswith(str(s))]
        data = np.array(data[0:]).astype(np.float)
    names = np.array(pd.read_csv(path, sep=';',header=None).transpose()[:1].drop(0,axis=1)).astype(np.str)[0]
    return data, names

#Import images to python
def imtopy(path):
    image = []
    names = []
    for i in os.listdir(path):
        if i.endswith('.tif'):
            names.append(i.replace('.tif',''))
            image.append(np.array(imageio.imread(path + i)[imageio.imread(path + i)>0]).astype(np.float))
    return np.array(image), np.array(names)

#multiple scatter plots with r (Spearman)
def multiplot(dx, dy, nx, ny, titulo = 'plots', path = ''):
    # matriz 2x2 de gráficos
    fig, ax = plt.subplots(nrows=dy.shape[0], ncols=dx.shape[0], figsize=(dx.shape[0]*2,dy.shape[0]*2), gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.suptitle(titulo, fontsize = 32)
    n1 = 0
    for x in dx:
        n2 = 0
        for y in dy:
            #plotar
            ax[n2,n1].plot(x, y, 'ro', ms = 2)
            #labels
            ax[0,n1].set_title(nx[n1],fontsize = 16)
            ax[n2,n1].set_ylabel(ny[n2],fontsize = 16)
            ax[n2,n1].label_outer()
            #ticks
            rangex = np.max(x)-np.min(x)
            xl1 = round((np.min(x)+rangex*0.15),3)
            xl3 = round((np.min(x)+rangex*0.85),3)
            rangey = np.max(y)-np.min(y)
            yl1 = round((np.min(y)+rangey*0.15),3)
            yl3 = round((np.min(y)+rangey*0.85),3)
            ax[n2,n1].yaxis.tick_right()
            ax[n2,n1].set_xticks([xl1,xl3])
            ax[n2,n1].set_yticks([yl1,yl3])
            ax[n2,n1].tick_params(axis='both', which='major', labelsize=16)
            ax[n2,n1].tick_params(axis='x', which='major', rotation=90)
            #r2
            ax[n2,n1].text(0.6,0.8,str(abs(round(spearmanr(x,y)[0],2))), fontsize=16, transform = ax[n2,n1].transAxes)
            n2 += 1
        n1 += 1
    if path != '':
        plt.savefig(path + titulo + '.png', dpi=300) #uncomment this and change the path to save files
    plt.show()

#multiple scatter plots with r2 (OLS)
def multiplotOLS(dx, dy, nx, ny, titulo = 'plots', size = 2, path = ''):
    # matriz 2x2 de gráficos
    fig, ax = plt.subplots(nrows=dy.shape[0], ncols=dx.shape[0], figsize=(dx.shape[0]*2,dy.shape[0]*2), gridspec_kw={'hspace': 0, 'wspace': 0})
    n1 = 0
    for x in dx:
        n2 = 0
        for y in dy:
            #modelo
            model = sm.OLS(y, sm.add_constant(x))
            fitted = model.fit()
            x_pred = np.linspace(x.min(), x.max(), 2)
            y_pred = fitted.predict(sm.add_constant(x_pred))
            ax[n2,n1].plot(x, y, 'ro', ms = size)
            ax[n2,n1].plot(x_pred, y_pred, 'b-', lw = 2)
            #labels
            ax[0,n1].set_title(nx[n1],fontsize = 16)
            ax[n2,n1].set_ylabel(ny[n2],fontsize = 16)
            ax[n2,n1].label_outer()
            #ticks
            rangex = np.max(x)-np.min(x)
            xl1 = round((np.min(x)+rangex*0.15),3)
            xl3 = round((np.min(x)+rangex*0.85),3)
            rangey = np.max(y)-np.min(y)
            yl1 = round((np.min(y)+rangey*0.15),1)
            yl3 = round((np.min(y)+rangey*0.85),1)
            ax[n2,n1].yaxis.tick_right()
            ax[n2,n1].set_xticks([xl1,xl3])
            ax[n2,n1].set_yticks([yl1,yl3])
            ax[n2,n1].tick_params(axis='both', which='major', labelsize=16)
            ax[n2,n1].tick_params(axis='x', which='major', rotation=90)
            #r2
            ax[n2,n1].text(np.min(x)+rangex*0.65,np.min(y)+rangey*0.85,str(abs(round(fitted.rsquared_adj,2))), fontsize=16)
            n2 += 1
        n1 += 1
    if path != '':
        plt.savefig(path + titulo + '.png', dpi=300) #uncomment this and change the path to save files
    plt.show()

#plotar r correlação
def plotr(dadoST, dados1, nomeST, nomes1, titulo = 'gráfico', cor = 'cubehelix', vmin = -0.7, vmax = 0.7, valores = True, cut = False):
    n = 0
    G = []
    for y in dados1:
        G.append([])
        for x in dadoST:
            G[n].append(round(spearmanr(y, x)[0],2))
        n += 1
    if cut == True:
        mask = np.zeros_like(G)
        mask[np.triu_indices_from(mask)] = True
    else:
        mask = None
    with sns.axes_style("white"):
        sns.heatmap(G, cmap = cor, xticklabels = nomeST, yticklabels = nomes1, annot=valores, square =False, mask = mask, vmin = vmin, vmax = vmax)
        plt.title(titulo + ' r (Spearman)')
        plt.gcf().set_size_inches(6.5,1.7)
        plt.yticks(rotation=0); plt.xticks(rotation=0)
        plt.tight_layout()
        # plt.savefig('C:/Users/gerez/Desktop/temp.png', dpi=300) #uncomment this and change the path to save files

#create simple ratios
def SimpRat(sentinel, nome):
    SRs = []
    nomesSRs = []
    n1 = 0
    for x in sentinel:
        n2 = 0
        for y in sentinel:
            if n2 > n1:
                SRs.append(x/y)
                nomesSRs.append(('SR'+nome[n1]+nome[n2]).replace('B',''))
            n2 += 1
        n1 += 1
    return np.array(SRs),np.array(nomesSRs) 

#create normalized differences
def NormDif(sentinel, nome):
    NDs = []
    nomesNDs = []
    n1 = 0
    for x in sentinel:
        n2 = 0
        for y in sentinel:
            if n2 > n1:
                NDs.append((x-y)/(x+y))
                nomesNDs.append(('ND'+nome[n1]+nome[n2]).replace('B',''))
            n2 += 1
        n1 += 1
    return np.array(NDs),np.array(nomesNDs) 

def plotSRr2(D1, D2, N1, N2, cor = 'gray', valores = True, vmin = 0, vmax = 0.5, nome = 'temp', path = ''):
    n1 = 0
    for L in D2:
        G = []
        n2 = 0
        for A in D1:
            G.append([])
            for B in D1:
                # G[n2].append(abs(round(spearmanr((A-B)/(A+B),L)[0], 2)))
                model = sm.OLS(L, sm.add_constant(A/B))
                fitted = model.fit()
                G[n2].append(abs(round(fitted.rsquared_adj, 2)))
            n2 += 1
        mask = np.zeros_like(G)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
            plt.title(str(N2[n1]) + ' vs Simple Ratios r2 (OLS)')
            sns.heatmap(G, cmap = cor, xticklabels = N1, yticklabels = N1, annot=valores, square =False, mask = mask, vmin = vmin, vmax = vmax)
            if path != '':
                plt.savefig(path + nome + str(N2[n1]) + ' vs Simple Ratios (r2 - OLS)'  + '.png', dpi=300)
            plt.show()
        n1 += 1

def plotNDr2(D1, D2, N1, N2, cor = 'gray', valores = True, vmin = 0, vmax = 0.5, nome = 'temp', path = ''):
    n1 = 0
    for L in D2:
        G = []
        n2 = 0
        for A in D1:
            G.append([])
            for B in D1:
                # G[n2].append(abs(round(spearmanr((A-B)/(A+B),L)[0], 2)))
                model = sm.OLS(L, sm.add_constant((A-B)/(A+B)))
                fitted = model.fit()
                G[n2].append(abs(round(fitted.rsquared_adj, 2)))
            n2 += 1
        mask = np.zeros_like(G)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
            plt.title(str(N2[n1]) + ' vs Normalized Differences r2 (OLS)')
            sns.heatmap(G, cmap = cor, xticklabels = N1, yticklabels = N1, annot=valores, square =False, mask = mask, vmin = vmin, vmax = vmax)
            if path != '':
                plt.savefig(path + nome + str(N2[n1]) + ' vs Normalized Differences (r2 - OLS)'  + '.png', dpi=300)
            plt.show()
        n1 += 1

#Mean absolute error (MAE)
def func_mae(predictions, targets):
    return (abs(predictions - targets)).mean()

#Root mean squared error (RMSE)
def func_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#powerset
def powerset(n):
    subsets = []
    for i in list(range(0, n+1)):
        for subset in list(itert.combinations(list(range(0, n)), i)):
            subsets.append(list(subset))
    del subsets[0]
    return subsets

#Calculate vegetation indices
def SAVI(B4,B8A):
    B4 = np.array(B4); B8A = np.array(B8A)
    savi = 1.25*((B8A-B4)/(B4+B8A+0.25))
    return savi

def EVI(B2,B4,B8A):
    B2 = np.array(B2); B4 = np.array(B4); B8A = np.array(B8A)
    evi = 2.5*(B8A-B4)/(B8A+6*B4-7.5*B2+0.25)
    return evi

def IRECI(B4, B5, B6, B7):
    B4 = np.array(B4); B5 = np.array(B5); B6 = np.array(B6); B7 = np.array(B7)
    ireci = B7-B4/(B5/B6)
    return ireci

def S2REP(B4, B5, B6, B7):
    B4 = np.array(B4); B5 = np.array(B5); B6 = np.array(B6); B7 = np.array(B7)
    s2rep = 705+35*((B7+B4)/2-B5)/(B6-B5)
    return s2rep

# exclude outliers
def outlier_index(data,n=3):
    index = []
    for i in data:
        i = sp.stats.boxcox(i)[0]
        number = 0
        mean = np.mean(i)
        std = np.std(i)
        for j in i:
            if j > mean+n*std or j < mean-n*std:
                if number not in index:
                    index.append(number)
            number += 1
    return index

# select features for OLS
def selectOLS_Bands(exp, res, name):
    n_var = exp.shape[0]
    valor = 999999
    tabela = []
    for i in powerset(n_var):
        modelo = sm.OLS(res, sm.add_constant(exp[i].transpose())).fit()
        tabela.append([name[i],modelo.aic])
        if valor > modelo.aic:
            valor = modelo.aic
            index = i
    tabela = np.array(tabela)
    tabela = tabela[np.argsort(tabela[:, 1],)]
    tabela = pd.DataFrame(tabela,columns=['Variables','AIC score'])
    print(tabela)
    return index, tabela

def selectOLS_VIS(exp, res, name):
    valor = 999999
    tabela = []
    n = 9
    for i in exp:
        modelo = sm.OLS(res, sm.add_constant(i)).fit()
        tabela.append([name[[n]],modelo.aic])
        if valor > modelo.aic:
            valor = modelo.aic
            index = n
        n += 1
    tabela = np.array(tabela)
    tabela = tabela[np.argsort(tabela[:, 1])]
    tabela = pd.DataFrame(tabela,columns=['Variables','AIC score'])
    print(tabela)
    return [index], tabela

# select features for RF
def selectRF_Bands(exp, res, name):
    n_var = exp.shape[0]
    valor = -1
    tabela = []
    for i in powerset(n_var):
        modelo = RandomForestRegressor(n_estimators=50, oob_score=True, random_state=42)
        modelo.fit(exp[i].transpose(), res)
        tabela.append([name[i],modelo.oob_score_])
        if valor < modelo.oob_score_:
            valor = modelo.oob_score_
            index = i
    tabela = np.array(tabela)
    tabela = tabela[np.argsort(-tabela[:, 1])]
    tabela = pd.DataFrame(tabela,columns=['Variables','OOB score'])
    print(tabela)
    return index, tabela

def selectRF_VIS(exp, res, name):
    valor = -1
    tabela = []
    n = 9
    n2 = 0
    for i in exp:
        modelo = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
        modelo.fit(exp[[n2]].transpose(), res)
        tabela.append([name[[n]],modelo.oob_score_])
        if valor < modelo.oob_score_:
            valor = modelo.oob_score_
            index = n
        n += 1
        n2 += 1
    tabela = np.array(tabela)
    tabela = tabela[np.argsort(-tabela[:, 1])]
    tabela = pd.DataFrame(tabela,columns=['Variables','OOB score'])
    print(tabela)
    return [index], tabela

# Random forest hyperparameter tuning
def RF_tuning(exp, res):
    random_forest_tuning = RandomForestRegressor(random_state = 42)
    # Randomized search in many parameters first:
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 200, num = 6)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    
    # Create big grid for random search
    random_grid = {'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'oob_score':[True]
        }
    RSCV = RandomizedSearchCV(estimator = random_forest_tuning, param_distributions = random_grid, n_iter = 50, cv = 5, verbose=2, random_state=42, n_jobs = -1)
    RSCV.fit(exp.transpose(), res)
    
    # Create small grid for full search
    param_grid = {'n_estimators': [RSCV.best_params_['n_estimators'],100],
        'max_features': [RSCV.best_params_['max_features']],
        'max_depth': [RSCV.best_params_['max_depth'], None],
        'min_samples_split': [2,RSCV.best_params_['min_samples_split']],
        'min_samples_leaf': [1,RSCV.best_params_['min_samples_leaf']],
        'oob_score':[True]
        }
    GSCV = GridSearchCV(estimator=random_forest_tuning, param_grid=param_grid, cv = 5 , n_jobs = -1)
    GSCV.fit(exp.transpose(), res)
    return GSCV.best_estimator_

# Print RF variable importances
def printImp(mod,name,res_n):
    imp = list(mod.feature_importances_)
    x_values = list(range(len(imp)))
    plt.bar(x_values,imp,orientation = 'vertical')
    plt.xticks(x_values, name, rotation='vertical')
    plt.title(res_n,fontsize = 16);
    plt.xlabel('Variables',fontsize = 16); plt.ylabel('Importances',fontsize = 16);
    plt.show()

####################
# create variables #
####################

#File paths
pathMain = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/') #get relative path
pathField = pathMain + "/Field.csv" #Field data
pathSentinel = pathMain + "/SentinelField.csv" #Sentinel for Field
pathSerie1 = pathMain + "/Serie1.csv" #time series Sentinel Data
pathSerie2 = pathMain + "/Serie2.csv" #time series LiDAR data
pathSerie = pathMain + "/SerieT.csv" #time series LiDAR data
pathL = pathMain + "/Lidar.csv" #time series LiDAR data

#Sentinel L1C Time Series B5
Serie1, SerieN1 = impcsv(pathSerie1)
Serie2, SerieN2 = impcsv(pathSerie2)
Serie, SerieN = impcsv(pathSerie)

#LiDAR variables time series
LT = impcsv(pathL)[0] #Both areas
L1, LN = impcsv(pathL,1) #Cantareira 1
L2 = impcsv(pathL,2)[0] #Cantareira 2

###################
# Field variables

Field = impcsv(pathField)[0] #variables values
Field_160108 = impcsv(pathField,1)[0]
Field_161223 = impcsv(pathField,2)[0]
FieldN = impcsv(pathField)[1] #variables names
Sentinel, SentinelN = impcsv(pathSentinel) #Sentinel 2 values for field data
Sentinel_160108 = impcsv(pathSentinel,1)[0]
Sentinel_161223 = impcsv(pathSentinel,2)[0]
indicesField = np.array([SAVI(Sentinel[2],Sentinel[6]),EVI(Sentinel[0],Sentinel[2],Sentinel[6]),IRECI(Sentinel[2],Sentinel[3],Sentinel[4],Sentinel[5]),S2REP(Sentinel[2],Sentinel[3],Sentinel[4],Sentinel[5])])

#######################
# Pixels

#Import images to python for per pixel analysis

#better names for bands
SN = np.array(['B2','B3','B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12'])

#LiDAR
pixelL1, pixelNamesL = imtopy(pathMain + "/Cantareira1/LiDAR/")
pixelL2, pixelNamesL = imtopy(pathMain + "/Cantareira2/LiDAR/")

LN = pixelNamesL[[1,2,3,0]]
pixelL1 = pixelL1[[1,2,3,0]]
pixelL2 = pixelL2[[1,2,3,0]]
pixelLT = np.concatenate((pixelL1,pixelL2),axis=1)

#Sentinel
pixelC1_150811, pixelNamesS = imtopy(pathMain + '/Cantareira1/Sentinel_2015_08_11/')
pixelC2_150811, pixelNamesS = imtopy(pathMain + '/Cantareira2/Sentinel_2015_08_11/')
pixelCT_150811 = np.concatenate((pixelC1_150811,pixelC2_150811),axis=1)
pixelC2_160108, pixelNamesS = imtopy(pathMain + '/Cantareira2/Sentinel_2016_01_08/')
pixelC1_161223, pixelNamesS = imtopy(pathMain + '/Cantareira1/Sentinel_2016_12_23/')
pixelC2_161223, pixelNamesS = imtopy(pathMain + '/Cantareira2/Sentinel_2016_12_23/')
pixelCT_161223 = np.concatenate((pixelC1_161223,pixelC2_161223),axis=1)
pixelC2_180122, pixelNamesS = imtopy(pathMain + '/Cantareira2/Sentinel_2018_01_22/')

#vegetation indices
names_VIS = np.array(['SAVI', 'EVI', 'IRECI', 'S2REP']) #names
pixelC1_150811_VIS = np.array([SAVI(pixelC1_150811[2],pixelC1_150811[6]),EVI(pixelC1_150811[0],pixelC1_150811[2],pixelC1_150811[6]),IRECI(pixelC1_150811[2],pixelC1_150811[3],pixelC1_150811[4],pixelC1_150811[5]),S2REP(pixelC1_150811[2],pixelC1_150811[3],pixelC1_150811[4],pixelC1_150811[5])])
pixelC2_150811_VIS = np.array([SAVI(pixelC2_150811[2],pixelC2_150811[6]),EVI(pixelC2_150811[0],pixelC2_150811[2],pixelC2_150811[6]),IRECI(pixelC2_150811[2],pixelC2_150811[3],pixelC2_150811[4],pixelC2_150811[5]),S2REP(pixelC2_150811[2],pixelC2_150811[3],pixelC2_150811[4],pixelC2_150811[5])])
pixelC2_160108_VIS = np.array([SAVI(pixelC2_160108[2],pixelC2_160108[6]),EVI(pixelC2_160108[0],pixelC2_160108[2],pixelC2_160108[6]),IRECI(pixelC2_160108[2],pixelC2_160108[3],pixelC2_160108[4],pixelC2_160108[5]),S2REP(pixelC2_160108[2],pixelC2_160108[3],pixelC2_160108[4],pixelC2_160108[5])])
pixelC1_161223_VIS = np.array([SAVI(pixelC1_161223[2],pixelC1_161223[6]),EVI(pixelC1_161223[0],pixelC1_161223[2],pixelC1_161223[6]),IRECI(pixelC1_161223[2],pixelC1_161223[3],pixelC1_161223[4],pixelC1_161223[5]),S2REP(pixelC1_161223[2],pixelC1_161223[3],pixelC1_161223[4],pixelC1_161223[5])])
pixelC2_161223_VIS = np.array([SAVI(pixelC2_161223[2],pixelC2_161223[6]),EVI(pixelC2_161223[0],pixelC2_161223[2],pixelC2_161223[6]),IRECI(pixelC2_161223[2],pixelC2_161223[3],pixelC2_161223[4],pixelC2_161223[5]),S2REP(pixelC2_161223[2],pixelC2_161223[3],pixelC2_161223[4],pixelC2_161223[5])])
pixelC2_180122_VIS = np.array([SAVI(pixelC2_180122[2],pixelC2_180122[6]),EVI(pixelC2_180122[0],pixelC2_180122[2],pixelC2_180122[6]),IRECI(pixelC2_180122[2],pixelC2_180122[3],pixelC2_180122[4],pixelC2_180122[5]),S2REP(pixelC2_180122[2],pixelC2_180122[3],pixelC2_180122[4],pixelC2_180122[5])])

pixelC1_150811_SR, names_SR = SimpRat(pixelC1_150811, SN)
pixelC2_150811_SR = SimpRat(pixelC2_150811, SN)[0]
pixelC2_160108_SR = SimpRat(pixelC2_160108, SN)[0]
pixelC1_161223_SR = SimpRat(pixelC1_161223, SN)[0]
pixelC2_161223_SR = SimpRat(pixelC2_161223, SN)[0]
pixelC2_180122_SR = SimpRat(pixelC2_180122, SN)[0]

pixelC1_150811_ND, names_ND = NormDif(pixelC1_150811, SN)
pixelC2_150811_ND = NormDif(pixelC2_150811, SN)[0]
pixelC2_160108_ND = NormDif(pixelC2_160108, SN)[0]
pixelC1_161223_ND = NormDif(pixelC1_161223, SN)[0]
pixelC2_161223_ND = NormDif(pixelC2_161223, SN)[0]
pixelC2_180122_ND = NormDif(pixelC2_180122, SN)[0]

#main VIs + bands 
names_BVIS = np.hstack([SN,names_VIS])
pixelC1_150811_BVIS = np.vstack([pixelC1_150811,pixelC1_150811_VIS])
pixelC2_150811_BVIS = np.vstack([pixelC2_150811,pixelC2_150811_VIS])
pixelC2_160108_BVIS = np.vstack([pixelC2_160108,pixelC2_160108_VIS])
pixelC1_161223_BVIS = np.vstack([pixelC1_161223,pixelC1_161223_VIS])
pixelC2_161223_BVIS = np.vstack([pixelC2_161223,pixelC2_161223_VIS])
pixelC2_180122_BVIS = np.vstack([pixelC2_180122,pixelC2_180122_VIS])

#ND and SR
names_NDSR = np.hstack([names_SR,names_ND])
pixelC1_150811_NDSR = np.vstack([pixelC1_150811_SR,pixelC1_150811_ND])
pixelC2_150811_NDSR = np.vstack([pixelC2_150811_SR,pixelC2_150811_ND])
pixelC2_160108_NDSR = np.vstack([pixelC2_160108_SR,pixelC2_160108_ND])
pixelC1_161223_NDSR = np.vstack([pixelC1_161223_SR,pixelC1_161223_ND])
pixelC2_161223_NDSR = np.vstack([pixelC2_161223_SR,pixelC2_161223_ND])
pixelC2_180122_NDSR = np.vstack([pixelC2_180122_SR,pixelC2_180122_ND])

#all variables
names_All = np.hstack([SN,names_VIS,names_SR,names_ND])
pixelC1_150811_All = np.vstack([pixelC1_150811,pixelC1_150811_VIS,pixelC1_150811_SR,pixelC1_150811_ND])
pixelC2_150811_All = np.vstack([pixelC2_150811,pixelC2_150811_VIS,pixelC2_150811_SR,pixelC2_150811_ND])
pixelC2_160108_All = np.vstack([pixelC2_160108,pixelC2_160108_VIS,pixelC2_160108_SR,pixelC2_160108_ND])
pixelC1_161223_All = np.vstack([pixelC1_161223,pixelC1_161223_VIS,pixelC1_161223_SR,pixelC1_161223_ND])
pixelC2_161223_All = np.vstack([pixelC2_161223,pixelC2_161223_VIS,pixelC2_161223_SR,pixelC2_161223_ND])
pixelC2_180122_All = np.vstack([pixelC2_180122,pixelC2_180122_VIS,pixelC2_180122_SR,pixelC2_180122_ND])

# Stop
raise Exception("Data successfully loaded. Go to line 520.")


#################################################
############ RUN IN PARTS BELOW THIS ############
#################################################

# Skip to line 660 for the regression analysis (OLS and RF)
# Continue below to view preliminary results


#######################
# Preliminary Results #
#######################

# select a code block and click "run selection or current line" (F9)

####################
# data statistics

# all available as spreadsheets as well (Variable Information - Preliminary results folder)

# below for all lidar data
infoL1 = pd.DataFrame(pixelL1,index = LN).T.describe() # LiDAR cantareira 1
print(infoL1)
infoL2 = pd.DataFrame(pixelL2,index = LN).T.describe() # LiDAR cantareira 2
print(infoL2)
infoLT = pd.DataFrame(pixelLT,index = LN).T.describe() # all lidar data
print(infoLT)

# below for lidar sentinel data
infoC1_150811 = pd.DataFrame(pixelC1_150811,index = SN).T.describe() # Sentinel 150811 Cantareira 1
print(infoC1_150811)
infoC2_150811 = pd.DataFrame(pixelC2_150811,index = SN).T.describe() # Sentinel 150811 Cantareira 2
print(infoC2_150811)
infoC_150811 = pd.DataFrame(pixelCT_150811,index = SN).T.describe() # Sentinel 150811 Cantareira 1 and 2
print(infoC_150811)
infoC2_160108 = pd.DataFrame(pixelC2_160108,index = SN).T.describe() # Sentinel 160108 Cantareira 2
print(infoC2_160108)
infoC1_161223 = pd.DataFrame(pixelC1_161223,index = SN).T.describe() # Sentinel 161223 Cantareira 1
print(infoC1_150811)
infoC2_161223 = pd.DataFrame(pixelC2_161223,index = SN).T.describe() # Sentinel 161223 Cantareira 2
print(infoC2_161223)
infoC_161223 = pd.DataFrame(pixelCT_161223,index = SN).T.describe() # Sentinel 161223 Cantareira 1 and 2
print(infoC_161223)
infoC2_180122 = pd.DataFrame(pixelC2_180122,index = SN).T.describe() # Sentinel 180122 Cantareira 2
print(infoC2_180122)

# below for all field data
infoF_160108 = pd.DataFrame(Field_160108,index = FieldN).T.describe() # Field
print(infoF_160108)
infoF_161223 = pd.DataFrame(Field_161223,index = FieldN).T.describe() # Field 160108
print(infoF_161223)
infoF = pd.DataFrame(Field,index = FieldN).T.describe() # Field 161223
print(infoF)

# below for field sentinel data
infoFS = pd.DataFrame(Sentinel,index = SentinelN).T.describe() # Sentinel 160108 and 161223
print(infoFS)
infoFS_160108 = pd.DataFrame(Sentinel_160108,index = SentinelN).T.describe() # Sentinel 160108
print(infoFS_160108)
infoFS_161223 = pd.DataFrame(Sentinel_161223,index = SentinelN).T.describe() # Sentinel and 161223
print(infoFS_161223)

##############
# time series

#Spearman r values for all cloudless L1C Sentinel 2 images (dates) and Cantareira 1 LiDAR data (LHmean)
c = 0
r1 = []
S=Serie1
for i in Serie1:
    temp = [SerieN1[c],round(spearmanr(L1[0], i)[0],2)]
    r1.append(temp)
    c += 1
r1 = np.array(r1)
print(r1)

#Spearman r values for all cloudless L1C Sentinel 2 images (dates) and Cantareira 2 LiDAR data (LHmean)
c = 0
r2 = []
S=Serie2
for i in Serie2:
    temp = [SerieN2[c],round(spearmanr(L2[0], i)[0],2)]
    r2.append(temp)
    c += 1
r2 = np.array(r2)
print(r2)

#Spearman r values for all cloudless L1C Sentinel 2 images (dates) and All LiDAR data (LHmean)
c = 0
r = []
S=Serie
for i in Serie:
    temp = [SerieN[c],round(spearmanr(LT[0], i)[0],2)]
    r.append(temp)
    c += 1
r = np.array(r)
print(r)


################################################
# correlations spearman r and preliminar OLS r2

# run one line at a time to visualize the results in the Plots tab
# run "pathSave = '' first"

pathSave = '' # don't save a plot file
# pathSave = 'C:/Users/gerez/Desktop/' # uncomment and change path to save a plot file

# Sentinel vs sentinel bands - spearman r
multiplot(pixelC2_150811, pixelC2_150811, SN, SN, 'Sentinel 2 11-08-2015', path = pathSave)
multiplot(pixelC2_160108, pixelC2_160108, SN, SN, 'Sentinel 2 08-01-2016', path = pathSave)
multiplot(pixelC2_161223, pixelC2_161223, SN, SN, 'Sentinel 2 23-12-2016', path = pathSave)
multiplot(pixelC2_180122, pixelC2_180122, SN, SN, 'Sentinel 2 22-01-2018', path = pathSave)
multiplot(Sentinel, Sentinel, SN, SN, 'Sentinel 2 22-01-2018', path = pathSave)

# Multiple scatter plots OLS r2
multiplotOLS(pixelC2_150811, pixelL2, SN, LN, 'Sentinel 2 11-08-2015', size = 0.25, path = pathSave) #Vegetation indices (11/08/2015) vs cantareira 2
multiplotOLS(pixelC2_160108, pixelL2, SN, LN, 'Sentinel 2 08-01-2016', size = 0.25, path = pathSave) #Vegetation indices (08/01/2016) vs cantareira 2
multiplotOLS(pixelC2_161223, pixelL2, SN, LN, 'Sentinel 2 23-12-2016', size = 0.25, path = pathSave) #Vegetation indices (23/12/2016) vs cantareira 2
multiplotOLS(pixelC2_180122, pixelL2, SN, LN, 'Sentinel 2 22-01-2018', size = 0.25, path = pathSave) #Vegetation indices (22/01/2018) vs cantareira 2

# Multiple scatter plots OLS r2 VIS
multiplotOLS(pixelC2_150811_VIS, pixelL2, names_VIS, LN, 'Sentinel 2 11-08-2015', size = 0.25, path = pathSave) #Vegetation indices (11/08/2015) vs cantareira 2
multiplotOLS(pixelC2_160108_VIS, pixelL2, names_VIS, LN, 'Sentinel 2 08-01-2016', size = 0.25, path = pathSave) #Vegetation indices (08/01/2016) vs cantareira 2
multiplotOLS(pixelC2_161223_VIS, pixelL2, names_VIS, LN, 'Sentinel 2 23-12-2016', size = 0.25, path = pathSave) #Vegetation indices (23/12/2016) vs cantareira 2
multiplotOLS(pixelC2_180122_VIS, pixelL2, names_VIS, LN, 'Sentinel 2 22-01-2018', size = 0.25, path = pathSave) #Vegetation indices (22/01/2018) vs cantareira 2

# Plot OLS r2 values for SR and ND indices
plotSRr2(pixelC2_150811, pixelL2, SN, LN, nome = 'Sentinel 2 11-08-2015', path = pathSave)
plotSRr2(pixelC2_160108, pixelL2, SN, LN, nome = 'Sentinel 2 08-01-2016', path = pathSave)
plotSRr2(pixelC2_161223, pixelL2, SN, LN, nome = 'Sentinel 2 23-12-2016', path = pathSave)
plotSRr2(pixelC2_180122, pixelL2, SN, LN, nome = 'Sentinel 2 22-01-2018', path = pathSave)

plotNDr2(pixelC2_150811, pixelL2, SN, LN, nome = 'Sentinel 2 11-08-2015', path = pathSave)
plotNDr2(pixelC2_160108, pixelL2, SN, LN, nome = 'Sentinel 2 08-01-2016', path = pathSave)
plotNDr2(pixelC2_161223, pixelL2, SN, LN, nome = 'Sentinel 2 23-12-2016', path = pathSave)
plotNDr2(pixelC2_180122, pixelL2, SN, LN, nome = 'Sentinel 2 22-01-2018', path = pathSave)


#######################
# preliminar OLS Field

multiplotOLS(Sentinel, Field, SentinelN, FieldN, 'Sentinel 2 Bands 220118 and 231216', path = pathSave) #Sentinel 2 vs Field data
multiplotOLS(indicesField, Field, names_VIS, FieldN, 'Sentinel 2 VIs 220118 and 231216', path = pathSave) #Vegetation indices vs Field data
plotSRr2(Sentinel, Field, SentinelN, FieldN, nome = 'Sentinel 2 220118 and 231216 ', path = pathSave) # plot OLS r2 values for SR indices
plotNDr2(Sentinel, Field, SentinelN, FieldN, nome = 'Sentinel 2 220118 and 231216 ', path = pathSave) # plot OLS r2 values for ND indices



#######################
# Regression Analysis #
#######################

# Select a code block and click "run selection or current line" (F9)

#Choose ONLY ONE validation method below (run from "valnamesave" to "names_e = names_All")
#Then proceed to choose a model type in line 1035


#Spatial Validation 23/12/2016
valnamesave = 'spatial validation'
valtype = 'spatial\nvalidation'
treino_e = pixelC1_161223_All; treino_r = pixelL1
teste_e = pixelC2_161223_All; teste_r = pixelL2
names_r = LN
names_e = names_All

##################################################################################
#with images 23/12/2016 (train) and 08/01/2016 (test)

#Temporal Validation 23/12/2016 (train) and 08/01/2016 (test)
valnamesave = 'temporal validation'
valtype = 'temporal\nvalidation'
treino_e = pixelC2_161223_All; treino_r = pixelL2
teste_e = pixelC2_160108_All; teste_r = pixelL2
names_r = LN
names_e = names_All

#Spatial and Temporal Validation 23/12/2016 (train) and 08/01/2016 (test)
valnamesave = 'spatial and temporal validation'
valtype = 'spatial\nand temporal validation'
treino_e = pixelC1_161223_All; treino_r = pixelL1
teste_e = pixelC2_160108_All; teste_r = pixelL2
names_r = LN
names_e = names_All

#Local Spatial and Temporal Validation 23/12/2016 (train) and 08/01/2016 (test)
valnamesave = 'local spatial and temporal validation'
valtype = 'local spatial\nand temporal validation'
treino_e = train_test_split(pixelC2_161223_All.transpose(), pixelL2.transpose(), test_size = 0.25, random_state = 42)[0].transpose();
treino_r = train_test_split(pixelC2_161223_All.transpose(), pixelL2.transpose(), test_size = 0.25, random_state = 42)[2].transpose();
teste_e = train_test_split(pixelC2_160108_All.transpose(), pixelL2.transpose(), test_size = 0.25, random_state = 42)[1].transpose();
teste_r = train_test_split(pixelC2_160108_All.transpose(), pixelL2.transpose(), test_size = 0.25, random_state = 42)[3].transpose();
names_r = LN
names_e = names_All

########################################################################################
#with images 08/01/2016 and 22/01/2018 for training in C2, 23/12/2016 for testing in C1

#Enhanced Spatial and Temporal Validation 08/01/2016 (train) and 22/01/2018 (train) and 23/12/2016 (test)
treino_e1 = train_test_split(pixelC2_180122_All.transpose(), pixelL2.transpose(), test_size = 0.5, random_state = 42)[0].transpose();
treino_r1 = train_test_split(pixelC2_180122_All.transpose(), pixelL2.transpose(), test_size = 0.5, random_state = 42)[2].transpose();
treino_e2 = train_test_split(pixelC2_160108_All.transpose(), pixelL2.transpose(), test_size = 0.5, random_state = 42)[1].transpose();
treino_r2 = train_test_split(pixelC2_160108_All.transpose(), pixelL2.transpose(), test_size = 0.5, random_state = 42)[3].transpose();

valnamesave = 'enhanced spatial and temporal validation'
valtype = 'enhanced spatial\nand temporal validation'
treino_e = np.hstack([treino_e1,treino_e2])
treino_r = np.hstack([treino_r1,treino_r2])
teste_e = pixelC1_161223_All; teste_r = pixelL1
names_r = LN
names_e = names_All



##############################################################################
# Choose a vegetation variable (n): 0 = mean, 1 = median, 2 = stdev, 3 = max #
##############################################################################

# WARNING: choose a validation method first.

n = 0 # mean

n = 1 # median

n = 2 # stdev

n = 3 # max


#######
# OLS #
#######

############
# SELECTION

# WARNING: choose a variable (n) first.

# Choose an index

# The index will be used to select explanatory variables
# You can view the best models by AIC in the "table" variable
# Choose an index below before running the model

##############################################################
# generate index for band combinations OLS model with best AIC

index, table = selectOLS_Bands(treino_e[list(range(0,9))],treino_r[n],names_e); varstype = '1'

#################################################################
# generate index for single literature VI OLS model with best AIC

index, table = selectOLS_VIS(treino_e[list(range(9,13))],treino_r[n],names_e); varstype = '2'

############################################################
# generate index for single SR/ND VI OLS model with best AIC

index, table = selectOLS_VIS(treino_e[list(range(13,treino_e.shape[0]))],treino_r[n],names_e); varstype = '3'

# Type names_e[index] in the console to see all explanatory variables used in the model


# index = list(range(0,9)) # run this to use all bands
# index = list(range(0,13)) # run this to use all bands and literature VIs
# index = list(range(0,treino_e.shape[0])) # run this to use all variables


###################################################################
# Generate OLS or WLS model and predictions (choose an index above first!)

######
# OLS
regtype = 'OLS_'
modelo = sm.OLS(treino_r[n], sm.add_constant(treino_e[index].transpose())).fit()
names = 'Predição OLS - '; print(names_e[index],names_r[n])
for i in names_e[index]: names += i + ' '
pred = modelo.predict(sm.add_constant(teste_e[index].transpose()))
me = (pred-teste_r[n]).mean()
mae = func_mae(pred,teste_r[n])
rmse = func_rmse(pred,teste_r[n])
rrse = rmse/(np.std(teste_r[n])) #*4
norm = 100*rmse/(np.percentile(teste_r[n],95)-np.percentile(teste_r[n],5))
r2 = modelo.rsquared_adj
label = 'Predicted OLS - ' + valtype + '\n'
print(valnamesave)
print('ME:',me,'MAE:',mae,'RMSE:',rmse,'RRSE:',rrse)


######
# WLS

# OLS with weights (w), not used in our work

########################################################################
# estimate w per variance, stdev or number of samples in classes (bins)

df = pd.Series(treino_r[n])
intervals = df.value_counts(bins=5, sort=False)

########### comment this block to use number of samples as weights #######
# variace/stdev as weights
split = np.split(np.sort(treino_r[n]),np.cumsum(list(intervals.values))[:-1])
varstd = []
# for i in split: varstd.append(i.var()) # variance
for i in split: varstd.append(i.std()) # standard deviation
intervals = pd.Series(varstd,intervals.index)
##########################################################################

out = pd.cut(list(df),intervals.index,labels=np.round(intervals.values,decimals=3).astype(str))
w = intervals.values[out.codes]


################################################################
# estimate w from previous OLS model (run OLS the model first!)

modelo_w = sm.OLS(abs(modelo.resid), modelo.model.exog).fit() #squared (or not) residuals
#modelo_w = sm.OLS(abs(modelo.resid), modelo.model.endog).fit() #squared (or not) residuals
#w = abs(modelo_w.fittedvalues)**-1
w = abs(modelo_w.fittedvalues).max() + abs(modelo_w.fittedvalues).min() - abs(modelo_w.fittedvalues)


###########
# invert w

w = w.max() + w.min() - w
# w = w/w.max() # normalizing has no effect


##########################################
# plot w vs the response variable

plt.plot(treino_r[n],w,'ko',ms=0.5)



#########################
# create WLS model

regtype = 'WLS_'
#w = (treino_r[4].min()+treino_r[4].max())*np.ones(treino_r.shape[1])-treino_r[4]
# w = np.ones(treino_r.shape[1])
modelo = sm.WLS(treino_r[n], sm.add_constant(treino_e[index].transpose()), w).fit()
names = 'Predição WLS - '; print(names_e[index],names_r[n])
for i in names_e[index]: names += i + ' '
pred = modelo.predict(sm.add_constant(teste_e[index].transpose()))
me = (pred-teste_r[n]).mean()
mae = func_mae(pred,teste_r[n])
rmse = func_rmse(pred,teste_r[n])
rrse = rmse/(np.std(teste_r[n])) #*4
norm = 100*rmse/(np.percentile(teste_r[n],95)-np.percentile(teste_r[n],5))
r2 = modelo.rsquared_adj
label = 'Predicted WLS - ' + valtype + '\n'
print(valnamesave)
print('ME:',me,'MAE:',mae,'RMSE:',rmse,'RRSE:',rrse)


######
# RF #
######

############
# SELECTION

# WARNING: choose a variable (n) first.

# Choose an index

# The index will be used to select explanatory variables
# You can view the best models by AIC in the "table" variable
# Choose an index below before running the model

##############################################################
# generate index for band combinations with best OOB score

index, table = selectRF_Bands(treino_e[list(range(0,9))],treino_r[n],names_e); varstype = '1'

###############################################################################
# generate index for single literature vegetation index with best OOB score

index, table = selectRF_VIS(treino_e[list(range(9,13))],treino_r[n],names_e); varstype = '2'

###############################################################################
# generate index for single SR / ND vegetation index with best OOB score

index, table = selectRF_VIS(treino_e[list(range(13,treino_e.shape[0]))],treino_r[n],names_e); varstype = '3'

# Type names_e[index] in the console to see all explanatory variables used in the model

#########
# other

# index = list(range(0,9)) # uncomment and run this to use all bands
# index = list(range(0,13)) # uncomment and run this to use all bands and literature VIs
# index = list(range(0,treino_e.shape[0])) # uncomment and run this to use all variables


######
# RF

regtype = 'RF_'
modelo = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
# modelo = RF_tuning(treino_e[index],treino_r[n]) # uncomment to tune model parameters (surprisingly, normaly has bad performance in the test set compared to standard parameters)
# print(modelo.get_params()) # uncomment to see the parameters used in the model
modelo.fit(treino_e[index].transpose(), treino_r[n])
names = 'Predição RF - '; print(names_e[index],names_r[n])
for i in names_e[index]: names += i + ' '
pred = modelo.predict(teste_e[index].transpose()).transpose()
me = (pred-teste_r[n]).mean()
mae = func_mae(pred,teste_r[n])
rmse = func_rmse(pred,teste_r[n])
rrse = rmse/(np.std(teste_r[n])) #*4
norm = 100*rmse/(np.percentile(teste_r[n],95)-np.percentile(teste_r[n],5))
r2 = modelo.oob_score_
label = 'Predicted RF - ' + valtype + '\n'
printImp(modelo,names_e[index],names_r[n]) #uncomment this to print variable importances
print(valnamesave)
print('ME:',me,'MAE:',mae,'RMSE:',rmse,'RRSE:',rrse)


##############################
# Plot the model predictions #
##############################

# WARNING: run a model first!.

# %matplotlib inline # plot in the plots tab
# %matplotlib qt # plot in a separate window

#plot the chosen model (run one of the models above first)
label2 = label
for i in names_e[index]:
    label2 = label2 + i + ' '
fsize = 22 #32
plt.xlabel(label2,fontsize = fsize)
plt.ylabel(names_r[n],fontsize = fsize)
# plt.ylabel(FieldN[n],fontsize = fsize) # uncomment if you are using field variables
plt.xticks(fontsize = fsize,)
plt.yticks(fontsize = fsize)
curve = np.polyfit(pred, teste_r[n], 1)
fiter = np.poly1d(curve)
plt.title('ME: ' + str(round(me,2)) + ' MAE: ' + str(round(mae,2)) + '\nRMSE: ' + str(round(rmse,2)) + ' RRSE: ' + str(round(rrse,2)),fontsize = fsize)
linha = np.linspace(np.min(teste_r[n]),np.max(teste_r[n]),2)
sticks = [3,3,2,4] # change ticks spacing for each variable
plt.xticks(np.arange(0, 100, sticks[n]))
plt.yticks(np.arange(0, 100, sticks[n]))
#plot
plt.plot(pred,teste_r[n],'ro',ms=0.5)
plt.plot(pred, fiter(pred), 'b-', linewidth = 3)
plt.plot(linha, linha, 'k', linewidth = 2, linestyle = '--')

#save
#change the directory (string) below to save a figure file
# plt.savefig('C:/Users/geral/Desktop/'+valnamesave+regtype+names_r[n]+varstype, dpi=300, bbox_inches='tight')



##############################
# Errors in 3 height classes #
##############################

joint = np.vstack([teste_r[n],pred])
space = (joint[0].max()-joint[0].min())/3
val1 = joint[0].min() + space
val2 = joint[0].min() + space*2
# index
arg1 = np.where(joint[0] < val1)[0]
arg2 = np.where(np.logical_and(joint[0] >= val1, joint[0] < val2))[0]
arg3 = np.where(joint[0] >= val2)[0]
# set 3 groups
bot = joint[:,arg1]
mid = joint[:,arg2]
top = joint[:,arg3]
# error values
me1 = (bot[1]-bot[0]).mean(); mae1 = func_mae(bot[1],bot[0])
me2 = (mid[1]-mid[0]).mean(); mae2 = func_mae(mid[1],mid[0])
me3 = (top[1]-top[0]).mean(); mae3 = func_mae(top[1],top[0])
# print(me1);print(me2);print(me3)
# print(mae1);print(mae2);print(mae3)

# plot mae
plt.bar(['low','mid','high'],[mae1,mae2,mae3])

# plot me
plt.bar(['low','mid','high'],[me1,me2,me3])



###################
# OLS diagnostics #
###################

# plots:

# normal
fig, ax = plt.subplots(figsize=(6,2.5))
_, (__, ___, r) = sp.stats.probplot(modelo.resid, plot=ax, fit=True)
r**2

# residuals histogram
plt.hist(modelo.resid,15)

# residuals vs fitted values
plt.plot(modelo.fittedvalues,modelo.resid,'ko',ms=1)
plt.plot([modelo.fittedvalues.min(),modelo.fittedvalues.max()],np.zeros(2),'r--')

# real values vs fitted values
plt.plot(modelo.fittedvalues,modelo.model.endog,'ko',ms=1)
mini = min(modelo.model.endog.min(),modelo.fittedvalues.min())
maxi = max(modelo.model.endog.max(),modelo.fittedvalues.max())
plt.plot([mini,maxi],[mini,maxi],'r--')

# plots for single predictive variables at a time
# change the variable with the number (second argument)
sm.graphics.plot_regress_exog(modelo,0)



