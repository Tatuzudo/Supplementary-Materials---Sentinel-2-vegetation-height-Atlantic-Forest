# intended for use in anaconda - spyder ide
# click "run file" (or press F5) once
# proceed to line 625

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
def multiplot(dx, dy, nx, ny, titulo = 'plots'):
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
    plt.savefig('C:/Users/gerez/Desktop/' + titulo + '.png', dpi=300) #uncomment this and change the path to save files
    plt.show()

#multiple scatter plots with r2 (OLS)
def multiplotOLS(dx, dy, nx, ny, titulo = 'plots'):
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
            ax[n2,n1].plot(x, y, 'ro', ms = 2)
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
    # plt.savefig('C:/Users/gerez/Desktop/temp.png', dpi=300) #uncomment this and change the path to save files
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

#plot simple ratio r values
def plotSRr(D1, D2, N1, N2, cor = 'cubehelix', valores = True, vmin = -0.7, vmax = 0.7, nome = 'temp'):
    n1 = 0
    for L in D2:
        G = []
        n2 = 0
        for B in D1:
            G.append([])
            for A in D1:
                G[n2].append(round(spearmanr(B/A,L)[0], 2))
            n2 += 1
        mask = np.zeros_like(G)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
            # ax0 = plt.figure().add_axes([1,1,np.size(N2)*0.14,np.size(N2)*0.12])
            plt.title(str(N2[n1]) + ' vs Simple Ratios r (Spearman)')
            sns.heatmap(G, cmap = cor, xticklabels = N1, yticklabels = N1, annot=valores, square =False, mask = mask, vmin = vmin, vmax = vmax)
            # plt.savefig('C:/Users/gerez/Desktop/' + nome + str(N2[n1]) + ' vs Simple Ratios r (Spearman)'  + '.png', dpi=300) #uncomment this and change the path to save files
            plt.show()
        n1 += 1

#plot normalized differences r values
def plotNDr(D1, D2, N1, N2, cor = 'cubehelix', valores = True, vmin = -0.7, vmax = 0.7, nome = 'temp'):
    n1 = 0
    for L in D2:
        G = []
        n2 = 0
        for B in D1:
            G.append([])
            for A in D1:
                G[n2].append(abs(round(spearmanr((A-B)/(A+B),L)[0], 2)))
            n2 += 1
        mask = np.zeros_like(G)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
            plt.title(str(N2[n1]) + ' vs Normalized Differences r (Spearman)')
            sns.heatmap(G, cmap = cor, xticklabels = N1, yticklabels = N1, annot=valores, square =False, mask = mask, vmin = vmin, vmax = vmax)
            # plt.savefig('C:/Users/gerez/Desktop/' + nome + str(N2[n1]) + ' vs Normalized Differences r (Spearman)'  + '.png', dpi=300) #uncomment this and change the path to save files
            plt.show()
        n1 += 1

def plotSRr2(D1, D2, N1, N2, cor = 'gray', valores = True, vmin = 0, vmax = 0.5, nome = 'temp'):
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
            plt.savefig('C:/Users/gerez/Desktop/' + nome + str(N2[n1]) + ' vs Simple Ratios (r2 - OLS)'  + '.png', dpi=300) #uncomment this and change the path to save files
            plt.show()
        n1 += 1

def plotNDr2(D1, D2, N1, N2, cor = 'gray', valores = True, vmin = 0, vmax = 0.5, nome = 'temp'):
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
            # plt.savefig('C:/Users/gerez/Desktop/' + nome + str(N2[n1]) + ' vs Normalized Differences (r2 - OLS)'  + '.png', dpi=300) #uncomment this and change the path to save files
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
pathMain = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
pathL = pathMain + "/Lidar.csv" #LiDAR data
pathLW = pathMain + "/Lidar_weight.csv" #LiDAR data
pathField = pathMain + "/Field.csv" #Field data
pathC150811 = pathMain + "/C150811.csv" #Sentinel for LiDAR
pathC160108 = pathMain + "/C160108.csv" #Sentinel for LiDAR
pathC161223 = pathMain + "/C161223.csv" #Sentinel for LiDAR
pathC180122 = pathMain + "/C180122.csv" #Sentinel for LiDAR
pathSentinel = pathMain + "/SentinelField.csv" #Sentinel for Field
pathSerie1 = pathMain + "/Serie1.csv" #time series
pathSerie2 = pathMain + "/Serie2.csv" #time series
pathSerie = pathMain + "/SerieT.csv" #time series

#Sentinel L1C Time Series B5
Serie1, SerieN1 = impcsv(pathSerie1)
Serie2, SerieN2 = impcsv(pathSerie2)
Serie, SerieN = impcsv(pathSerie)

#Sentinel L2A
C_150811, SN = impcsv(pathC150811) #Both areas and band names
C1_150811 = impcsv(pathC150811,1)[0] #Cantareira 1
C2_150811 = impcsv(pathC150811,2)[0] #Cantareira 2
C2_160108 = impcsv(pathC160108,2)[0] #Cantareira 2
C_161223 = impcsv(pathC161223)[0] #Both areas
C1_161223 = impcsv(pathC161223,1)[0] #Cantareira 1
C2_161223 = impcsv(pathC161223,2)[0] #Cantareira 2
C2_180122 = impcsv(pathC180122,2)[0] #clouds in cantareira 1


#LiDAR variables
LT = impcsv(pathL)[0] #Both areas
L1, LN = impcsv(pathL,1) #Cantareira 1
L2 = impcsv(pathL,2)[0] #Cantareira 2
#LN = ['Altura mínima','Altura máxima 10','Altura máxima 5','Altura média','Altura relativa','Altura stdev','Cobertura 10','Cobertura 1','Cobertura 5','Cobertura relativa','Densidade','Ecos','Profundidade']


#LiDAR variables with weight
LT = impcsv(pathLW)[0] #Both areas
L1, LN = impcsv(pathLW,1) #Cantareira 1
L2 = impcsv(pathLW,2)[0] #Cantareira 2
#LN = ['Altura mínima','Altura máxima 10','Altura máxima 5','Altura média','Altura relativa','Altura stdev','Cobertura 10','Cobertura 1','Cobertura 5','Cobertura relativa','Densidade','Ecos','Profundidade']

#vegetation indices
names_VIS = np.array(['SAVI', 'EVI', 'IRECI', 'S2REP']) #names
C1_150811_VIS = np.array([SAVI(C1_150811[2],C1_150811[6]),EVI(C1_150811[0],C1_150811[2],C1_150811[6]),IRECI(C1_150811[2],C1_150811[3],C1_150811[4],C1_150811[5]),S2REP(C1_150811[2],C1_150811[3],C1_150811[4],C1_150811[5])])
C2_150811_VIS = np.array([SAVI(C2_150811[2],C2_150811[6]),EVI(C2_150811[0],C2_150811[2],C2_150811[6]),IRECI(C2_150811[2],C2_150811[3],C2_150811[4],C2_150811[5]),S2REP(C2_150811[2],C2_150811[3],C2_150811[4],C2_150811[5])])
C2_160108_VIS = np.array([SAVI(C2_160108[2],C2_160108[6]),EVI(C2_160108[0],C2_160108[2],C2_160108[6]),IRECI(C2_160108[2],C2_160108[3],C2_160108[4],C2_160108[5]),S2REP(C2_160108[2],C2_160108[3],C2_160108[4],C2_160108[5])])
C1_161223_VIS = np.array([SAVI(C1_161223[2],C1_161223[6]),EVI(C1_161223[0],C1_161223[2],C1_161223[6]),IRECI(C1_161223[2],C1_161223[3],C1_161223[4],C1_161223[5]),S2REP(C1_161223[2],C1_161223[3],C1_161223[4],C1_161223[5])])
C2_161223_VIS = np.array([SAVI(C2_161223[2],C2_161223[6]),EVI(C2_161223[0],C2_161223[2],C2_161223[6]),IRECI(C2_161223[2],C2_161223[3],C2_161223[4],C2_161223[5]),S2REP(C2_161223[2],C2_161223[3],C2_161223[4],C2_161223[5])])
C2_180122_VIS = np.array([SAVI(C2_180122[2],C2_180122[6]),EVI(C2_180122[0],C2_180122[2],C2_180122[6]),IRECI(C2_180122[2],C2_180122[3],C2_180122[4],C2_180122[5]),S2REP(C2_180122[2],C2_180122[3],C2_180122[4],C2_180122[5])])

C1_150811_SR, names_SR = SimpRat(C1_150811, SN)
C2_150811_SR = SimpRat(C2_150811, SN)[0]
C2_160108_SR = SimpRat(C2_160108, SN)[0]
C1_161223_SR = SimpRat(C1_161223, SN)[0]
C2_161223_SR = SimpRat(C2_161223, SN)[0]
C2_180122_SR = SimpRat(C2_180122, SN)[0]

C1_150811_ND, names_ND = NormDif(C1_150811, SN)
C2_150811_ND = NormDif(C2_150811, SN)[0]
C2_160108_ND = NormDif(C2_160108, SN)[0]
C1_161223_ND = NormDif(C1_161223, SN)[0]
C2_161223_ND = NormDif(C2_161223, SN)[0]
C2_180122_ND = NormDif(C2_180122, SN)[0]

#main VIs + bands 
names_BVIS = np.hstack([SN,names_VIS])
C1_150811_BVIS = np.vstack([C1_150811,C1_150811_VIS])
C2_150811_BVIS = np.vstack([C2_150811,C2_150811_VIS])
C2_160108_BVIS = np.vstack([C2_160108,C2_160108_VIS])
C1_161223_BVIS = np.vstack([C1_161223,C1_161223_VIS])
C2_161223_BVIS = np.vstack([C2_161223,C2_161223_VIS])
C2_180122_BVIS = np.vstack([C2_180122,C2_180122_VIS])

#ND and SR
names_NDSR = np.hstack([names_SR,names_ND])
C1_150811_NDSR = np.vstack([C1_150811_SR,C1_150811_ND])
C2_150811_NDSR = np.vstack([C2_150811_SR,C2_150811_ND])
C2_160108_NDSR = np.vstack([C2_160108_SR,C2_160108_ND])
C1_161223_NDSR = np.vstack([C1_161223_SR,C1_161223_ND])
C2_161223_NDSR = np.vstack([C2_161223_SR,C2_161223_ND])
C2_180122_NDSR = np.vstack([C2_180122_SR,C2_180122_ND])

#all variables
names_All = np.hstack([SN,names_VIS,names_SR,names_ND])
C1_150811_All = np.vstack([C1_150811,C1_150811_VIS,C1_150811_SR,C1_150811_ND])
C2_150811_All = np.vstack([C2_150811,C2_150811_VIS,C2_150811_SR,C2_150811_ND])
C2_160108_All = np.vstack([C2_160108,C2_160108_VIS,C2_160108_SR,C2_160108_ND])
C1_161223_All = np.vstack([C1_161223,C1_161223_VIS,C1_161223_SR,C1_161223_ND])
C2_161223_All = np.vstack([C2_161223,C2_161223_VIS,C2_161223_SR,C2_161223_ND])
C2_180122_All = np.vstack([C2_180122,C2_180122_VIS,C2_180122_SR,C2_180122_ND])

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

#LiDAR
pixelL1, pixelNamesL = imtopy(pathMain + "/Cantareira1/LiDAR/")
pixelL2, pixelNamesL = imtopy(pathMain + "/Cantareira2/LiDAR/")

pixelNamesL = pixelNamesL[[1,2,0]]
pixelL1 = pixelL1[[1,2,0]]
pixelL2 = pixelL2[[1,2,0]]

#Sentinel
pixelC1_150811, pixelNamesS = imtopy(pathMain + '/Cantareira1/Sentinel_2015_08_11/')
pixelC2_150811, pixelNamesS = imtopy(pathMain + '/Cantareira2/Sentinel_2015_08_11/')
pixelC2_160108, pixelNamesS = imtopy(pathMain + '/Cantareira2/Sentinel_2016_01_08/')
pixelC1_161223, pixelNamesS = imtopy(pathMain + '/Cantareira1/Sentinel_2016_12_23/')
pixelC2_161223, pixelNamesS = imtopy(pathMain + '/Cantareira2/Sentinel_2016_12_23/')
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
raise Exception("Data successfully loaded. Go to line 625.")


#################################################
############ RUN IN PARTS BELOW THIS ############
#################################################

# Skip to line 810 for the regression analysis (OLS and RF)
# Continue below to view preliminary results


#######################
# Preliminary Results #
#######################

# select a code block and click "run selection or current line" (F9)

####################
# data statistics

# all available as spreadsheets as well (Variable Information - Preliminary results folder)

# below for LiDAR data
infoL1 = pd.DataFrame(L1,index = LN).T.describe() # LiDAR cantareira 1
print(infoL1)
infoL2 = pd.DataFrame(L2,index = LN).T.describe() # LiDAR cantareira 2
print(infoL2)
infoL = pd.DataFrame(LT,index = LN).T.describe() # all lidar data
print(infoL)

# below for lidar sentinel data
infoC1_150811 = pd.DataFrame(C1_150811,index = SN).T.describe() # Sentinel 150811 Cantareira 1
print(infoC1_150811)
infoC2_150811 = pd.DataFrame(C2_150811,index = SN).T.describe() # Sentinel 150811 Cantareira 2
print(infoC2_150811)
infoC_150811 = pd.DataFrame(C_150811,index = SN).T.describe() # Sentinel 150811 Cantareira 1 and 2
print(infoC_150811)
infoC2_160108 = pd.DataFrame(C2_160108,index = SN).T.describe() # Sentinel 160108 Cantareira 2
print(infoC2_160108)
infoC1_161223 = pd.DataFrame(C1_161223,index = SN).T.describe() # Sentinel 161223 Cantareira 1
print(infoC1_150811)
infoC2_161223 = pd.DataFrame(C2_161223,index = SN).T.describe() # Sentinel 161223 Cantareira 2
print(infoC2_161223)
infoC_161223 = pd.DataFrame(C_161223,index = SN).T.describe() # Sentinel 161223 Cantareira 1 and 2
print(infoC_161223)
infoC2_180122 = pd.DataFrame(C2_180122,index = SN).T.describe() # Sentinel 180122 Cantareira 2
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

#Spearman r values for all L1C Sentinel 2 images (dates) and Cantareira 1 LiDAR data (LHmean)
c = 0
r1 = []
S=Serie1
for i in Serie1:
    temp = [SerieN1[c],round(spearmanr(L1[0], i)[0],2)]
    r1.append(temp)
    c += 1
r1 = np.array(r1)
print(r1)

#Spearman r values for all L1C Sentinel 2 images (dates) and Cantareira 2 LiDAR data (LHmean)
c = 0
r2 = []
S=Serie2
for i in Serie2:
    temp = [SerieN2[c],round(spearmanr(L2[0], i)[0],2)]
    r2.append(temp)
    c += 1
r2 = np.array(r2)
print(r2)

#Spearman r values for all L1C Sentinel 2 images (dates) and All LiDAR data (LHmean)
c = 0
r = []
S=Serie
for i in Serie:
    temp = [SerieN[c],round(spearmanr(LT[0], i)[0],2)]
    r.append(temp)
    c += 1
r = np.array(r)
print(r)

# correlations spearman r or OLS r2

# run one line at a time to visualize the results in the Plots tab

#plot Spearman r values for single bands (LiDAR)
plotr(C2_150811, L2, SN, LN, 'LiDAR vs Sentinel 11/08/2015')
plotr(C2_160108, L2, SN, LN, 'LiDAR vs Sentinel 08/01/2016')
plotr(C2_161223, L2, SN, LN, 'LiDAR vs Sentinel 23/12/2016')
plotr(C2_180122, L2, SN, LN, 'LiDAR vs Sentinel 22/01/2018')

#plot Spearman r values for vegetation indices (LiDAR)
plotr(L2, C2_150811_VIS, LN, names_VIS, 'LiDAR vs VIs Sentinel 11/08/2015')
plotr(L2, C2_160108_VIS, LN, names_VIS, 'LiDAR vs VIs Sentinel 08/01/2016')
plotr(L2, C2_161223_VIS, LN, names_VIS, 'LiDAR vs VIs Sentinel 23/12/2016')
plotr(L2, C2_180122_VIS, LN, names_VIS, 'LiDAR vs VIs Sentinel 22/01/2018')
plotr(Field, indicesField, FieldN, names_VIS, 'Field vs VIs Sentinel 08/01/2016 and 23/12/2016')

#plot Spearman r values for SR and ND indices (LiDAR)
plotSRr(C2_150811, L2, SN, LN, nome = 'Sentinel 2 110815 ')
plotNDr(C2_150811, L2, SN, LN, nome = 'Sentinel 2 110815 ')
plotSRr(C2_160108, L2, SN, LN, nome = 'Sentinel 2 080116 ')
plotNDr(C2_160108, L2, SN, LN, nome = 'Sentinel 2 080116 ')
plotSRr(C2_161223, L2, SN, LN, nome = 'Sentinel 2 231216 ')
plotNDr(C2_161223, L2, SN, LN, nome = 'Sentinel 2 231216 ')
plotSRr(C2_180122, L2, SN, LN, nome = 'Sentinel 2 220118 ')
plotNDr(C2_180122, L2, SN, LN, nome = 'Sentinel 2 220118 ')

#plot OLS r2 values for SR and ND indices (LiDAR)
plotSRr2(C2_150811, L2, SN, LN, nome = 'Sentinel 2 110815 ')
plotSRr2(C2_160108, L2, SN, LN, nome = 'Sentinel 2 080116 ')
# plotSRr2(C_161223, LT, SN, LN, nome = 'Sentinel 2 231216 ') # cantareira 1 and 2
plotSRr2(C2_161223, L2, SN, LN, nome = 'Sentinel 2 231216 ')
plotSRr2(C2_180122, L2, SN, LN, nome = 'Sentinel 2 220118 ')
plotNDr2(C2_150811, L2, SN, LN, nome = 'Sentinel 2 110815 ')
plotNDr2(C2_160108, L2, SN, LN, nome = 'Sentinel 2 080116 ')
plotNDr2(C2_161223, L2, SN, LN, nome = 'Sentinel 2 231216 ')
plotNDr2(C_161223, LT, SN, LN, nome = 'Sentinel 2 231216 ') # cantareira 1 and 2
plotNDr2(C2_180122, L2, SN, LN, nome = 'Sentinel 2 220118 ')

#plot spearman r or OLS r2 values for field data
plotr(Sentinel, Field, SentinelN, FieldN, 'Field vs Sentinel 08/01/2016 and 23/12/2016')
plotr(indicesField, Field, names_VIS, FieldN, 'Field vs Vegetation Indices')
plotSRr(Sentinel, Field, SentinelN, FieldN) # plot Spearman r values for SR indices
plotNDr(Sentinel, Field, SentinelN, FieldN) # plot Spearman r values for ND indices
plotSRr2(Sentinel, Field, SentinelN, FieldN, nome = 'Sentinel 2 220118 and 231216 ') # plot OLS r2 values for SR indices
plotNDr2(Sentinel, Field, SentinelN, FieldN, nome = 'Sentinel 2 220118 and 231216 ') # plot OLS r2 values for ND indices

#multiple scatter plots - correlation spearman r
multiplot(Sentinel, Field, SentinelN, FieldN) #Sentinel 2 vs Field data
multiplot(C_150811, LT, SN, LN) #Sentinel 2 vs Cantareira 1 and 2
multiplot(C2_150811, L2, SN, LN) #Sentinel 2 vs cantareira 2
multiplot(C_161223, LT, SN, LN) #Sentinel 2 vs Cantareira 1 and 2
multiplot(C2_160108, L2, SN, LN) #Sentinel 2 vs cantareira 2
multiplot(C2_161223, L2, SN, LN) #Sentinel 2 vs cantareira 2
multiplot(C2_180122, L2, SN, LN) #Sentinel 2 vs cantareira 2

multiplot(indicesField, Field, names_VIS, FieldN) #Vegetation indices vs Field data
multiplot(C2_150811_VIS, L2, names_VIS, LN) #Vegetation indices (11/08/2015) vs cantareira 2
multiplot(C2_160108_VIS, L2, names_VIS, LN) #Vegetation indices (08/01/2016) vs cantareira 2
multiplot(C2_161223_VIS, L2, names_VIS, LN) #Vegetation indices (23/12/2016) vs cantareira 2
multiplot(C2_180122_VIS, L2, names_VIS, LN) #Vegetation indices (22/01/2018) vs cantareira 2

#multiple scatter plots OLS r2
multiplotOLS(Sentinel, Field, SentinelN, FieldN) #Sentinel 2 vs Field data
multiplotOLS(Sentinel_160108, Field_160108, SentinelN, FieldN)
multiplotOLS(Sentinel_161223, Field_161223, SentinelN, FieldN)
# multiplotOLS(C_150811, LT, SN, LN) #Sentinel 2 vs Cantareira 1 and 2
multiplotOLS(C2_150811, L2, SN, LN) #Sentinel 2 vs cantareira 2
# multiplotOLS(C_161223, LT, SN, LN) #Sentinel 2 vs Cantareira 1 and 2
multiplotOLS(C2_160108, L2, SN, LN) #Sentinel 2 vs cantareira 2
multiplotOLS(C2_161223, L2, SN, LN) #Sentinel 2 vs cantareira 2
multiplotOLS(C2_180122, L2, SN, LN) #Sentinel 2 vs cantareira 2

multiplotOLS(indicesField, Field, names_VIS, FieldN) #Vegetation indices vs Field data
multiplotOLS(C2_150811_VIS, L2, names_VIS, LN) #Vegetation indices (11/08/2015) vs cantareira 2
multiplotOLS(C2_160108_VIS, L2, names_VIS, LN) #Vegetation indices (08/01/2016) vs cantareira 2
multiplotOLS(C2_161223_VIS, L2, names_VIS, LN) #Vegetation indices (23/12/2016) vs cantareira 2
multiplotOLS(C2_180122_VIS, L2, names_VIS, LN) #Vegetation indices (22/01/2018) vs cantareira 2

# Sentinel 2 vs Sentinel bands 2 spearman r
multiplot(C2_150811, C2_150811, SN, SN, 'Sentinel 2 11-08-2015')
multiplot(C2_160108, C2_160108, SN, SN, 'Sentinel 2 08-01-2016')
multiplot(C2_161223, C2_161223, SN, SN, 'Sentinel 2 23-12-2016')
multiplot(C2_180122, C2_180122, SN, SN, 'Sentinel 2 22-01-2018')




#######################
# Regression Analysis #
#######################

# Select a code block and click "run selection or current line" (F9)

#Choose ONLY ONE validation method below (run from "valnamesave" to "names_e = names_All")
#Then proceed to choose a model type in line 1035

#################
# For LiDAR data
#    V  V  V 

################
###### Buffers #

#Spatial Validation 23/12/2016
valnamesave = 'spatial validation'
valtype = 'spatial\nvalidation'
treino_e = C1_161223_All; treino_r = L1
teste_e = C2_161223_All; teste_r = L2
names_r = LN
names_e = names_All

##################################################################################
#with images 23/12/2016 (train) and 08/01/2016 (test)

#Temporal Validation 23/12/2016 (train) and 08/01/2016 (test)
valnamesave = 'temporal validation'
valtype = 'temporal\nvalidation'
treino_e = C2_161223_All; treino_r = L2
teste_e = C2_160108_All; teste_r = L2
names_r = LN
names_e = names_All

#Spatial and Temporal Validation 23/12/2016 (train) and 08/01/2016 (test)
valnamesave = 'spatial and temporal validation'
valtype = 'spatial\nand temporal validation'
treino_e = C1_161223_All; treino_r = L1
teste_e = C2_160108_All; teste_r = L2
names_r = LN
names_e = names_All

#Local Spatial and Temporal Validation 23/12/2016 (train) and 08/01/2016 (test)
valnamesave = 'local spatial and temporal validation'
valtype = 'local spatial\nand temporal validation'
treino_e = train_test_split(C2_161223_All.transpose(), L2.transpose(), test_size = 0.25, random_state = 42)[0].transpose();
treino_r = train_test_split(C2_161223_All.transpose(), L2.transpose(), test_size = 0.25, random_state = 42)[2].transpose();
teste_e = train_test_split(C2_160108_All.transpose(), L2.transpose(), test_size = 0.25, random_state = 42)[1].transpose();
teste_r = train_test_split(C2_160108_All.transpose(), L2.transpose(), test_size = 0.25, random_state = 42)[3].transpose();
names_r = LN
names_e = names_All

########################################################################################
#with images 08/01/2016 and 22/01/2018 for training in C2, 23/12/2016 for testing in C1

#Enhanced Spatial and Temporal Validation 08/01/2016 (train), 22/01/2018 (train) and 23/12/2016 (test)
treino_e1 = train_test_split(C2_180122_All.transpose(), L2.transpose(), test_size = 0.5, random_state = 42)[0].transpose();
treino_r1 = train_test_split(C2_180122_All.transpose(), L2.transpose(), test_size = 0.5, random_state = 42)[2].transpose();
treino_e2 = train_test_split(C2_160108_All.transpose(), L2.transpose(), test_size = 0.5, random_state = 42)[1].transpose();
treino_r2 = train_test_split(C2_160108_All.transpose(), L2.transpose(), test_size = 0.5, random_state = 42)[3].transpose();

valnamesave = 'enhanced spatial and temporal validation'
valtype = 'enhanced spatial\nand temporal validation'
treino_e = np.hstack([treino_e1,treino_e2])
treino_r = np.hstack([treino_r1,treino_r2])
teste_e = C1_161223_All; teste_r = L1
names_r = LN
names_e = names_All



# note: the results from below this are not shown in our work.

##################################################################################
#with images 23/12/2016 (train) and 22/01/2018 (test)

#Temporal Validation 23/12/2016 (train) and 22/01/2018 (test)
valnamesave = 'temporal validation'
valtype = 'temporal\nvalidation'
treino_e = C2_161223_All; treino_r = L2
teste_e = C2_180122_All; teste_r = L2
names_r = LN
names_e = names_All

#Spatial and Temporal Validation 23/12/2016 (train) and 22/01/2018 (test)
valnamesave = 'spatial and temporal validation'
valtype = 'spatial\nand temporal validation'
treino_e = C1_161223_All; treino_r = L1
teste_e = C2_180122_All; teste_r = L2
names_r = LN
names_e = names_All

#Local Spatial and Temporal Validation 23/12/2016 (train) and 22/01/2018 (test)
valnamesave = 'local spatial and temporal validation'
valtype = 'local spatial\nand temporal validation'
treino_e = train_test_split(C2_161223_All.transpose(), L2.transpose(), test_size = 0.25, random_state = 42)[0].transpose();
treino_r = train_test_split(C2_161223_All.transpose(), L2.transpose(), test_size = 0.25, random_state = 42)[2].transpose();
teste_e = train_test_split(C2_180122_All.transpose(), L2.transpose(), test_size = 0.25, random_state = 42)[1].transpose();
teste_r = train_test_split(C2_180122_All.transpose(), L2.transpose(), test_size = 0.25, random_state = 42)[3].transpose();
names_r = LN
names_e = names_All

#with images 23/12/2016 (train) and 11/08/2015 (test)

#Temporal Validation 23/12/2016 (train) and 11/08/2015 (test)
valnamesave = 'temporal validation'
valtype = 'temporal\nvalidation'
treino_e = C2_161223_All; treino_r = L2
teste_e = C2_150811_All; teste_r = L2
names_r = LN
names_e = names_All

#Spatial and Temporal Validation 23/12/2016 (train) and 11/08/2015 (test)
valnamesave = 'spatial and temporal validation'
valtype = 'spatial\nand temporal validation'
treino_e = C1_161223_All; treino_r = L1
teste_e = C2_150811_All; teste_r = L2
names_r = LN
names_e = names_All

#Local Spatial and Temporal Validation 23/12/2016 (train) and 11/08/2015 (test)
valnamesave = 'local spatial and temporal validation'
valtype = 'local spatial\nand temporal validation'
treino_e = train_test_split(C2_161223_All.transpose(), L2.transpose(), test_size = 0.25, random_state = 42)[0].transpose();
treino_r = train_test_split(C2_161223_All.transpose(), L2.transpose(), test_size = 0.25, random_state = 42)[2].transpose();
teste_e = train_test_split(C2_150811_All.transpose(), L2.transpose(), test_size = 0.25, random_state = 42)[1].transpose();
teste_r = train_test_split(C2_150811_All.transpose(), L2.transpose(), test_size = 0.25, random_state = 42)[3].transpose();
names_r = LN
names_e = names_All




################
###### Pixels #

# You can train models in buffers and test in pixels
# To do this, run treino_e and treino_r in buffers
# Then run teste_e and teste_r in pixels
# Remember to choose the same validation method for both

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
#with images 08/01/2016 and 22/01/2018 for training in pixelC2, 23/12/2016 for testing in pixelC1

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




##################
# for Field data #

#Spatial validation
valnamesave = 'spatial validation'
valtype = 'spatial\nvalidation'
treino_e = train_test_split(Sentinel.transpose(), Field.transpose(), test_size = 0.25, random_state = 42)[0].transpose();
treino_r = train_test_split(Sentinel.transpose(), Field.transpose(), test_size = 0.25, random_state = 42)[2].transpose();
teste_e = train_test_split(Sentinel.transpose(), Field.transpose(), test_size = 0.25, random_state = 42)[1].transpose();
teste_r = train_test_split(Sentinel.transpose(), Field.transpose(), test_size = 0.25, random_state = 42)[3].transpose();
names_r = FieldN
names_e = SentinelN

#Spatial and Temporal Validation
valnamesave = 'spatial and temporal validation'
valtype = 'spatial\nand temporal validation'
treino_e = Sentinel_161223; treino_r = Field_161223
teste_e = Sentinel_160108; teste_r = Field_160108
names_r = FieldN
names_e = SentinelN





##############################################################################
# Choose a vegetation variable (n): 0 = mean, 1 = median, 2 = stdev, 3 = max #
##############################################################################

# WARNING: choose a validation method first.

n = 0 # mean

n = 1 # median

n = 2 # stdev (not available for test with pixels)

n = 3 # max (not available for test with pixels)


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

index, table = selectOLS_Bands(treino_e[list(range(0,9))],treino_r[n],names_e)

#################################################################
# generate index for single literature VI OLS model with best AIC

index, table = selectOLS_VIS(treino_e[list(range(9,13))],treino_r[n],names_e)

############################################################
# generate index for single SR/ND VI OLS model with best AIC

index, table = selectOLS_VIS(treino_e[list(range(13,treino_e.shape[0]))],treino_r[n],names_e)

# Type names_e[index] in the console to see all explanatory variables used in the model

########
# other

# index = list(range(0,9)) # run this to use all bands
# index = list(range(0,13)) # run this to use all bands and literature VIs
# index = list(range(0,treino_e.shape[0])) # run this to use all variables

# below is an index for an interesting band / VIs combination with very good OLS results:
# index = [1,5,9,12,34,36,47,70,72]
# ['B03', 'B07', 'SAVI', 'S2REP', 'SRB05B06', 'SRB05B8A', 'SRB8AB12', 'NDB05B06', 'NDB05B8A']


###################################################################
# Generate OLS model and predictions (choose an index above first!)

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

# WLS (OLS with weights, not used in our work)
# WARNING: the weights (w) are only defined for LHmean (n = 0) with buffers
regtype = 'WLS_'
w = (treino_r[4].min()+treino_r[4].max())*np.ones(treino_r.shape[1])-treino_r[4]
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

index, table = selectRF_Bands(treino_e[list(range(0,9))],treino_r[n],names_e)

###############################################################################
# generate index for single literature vegetation index with best OOB score

index, table = selectRF_VIS(treino_e[list(range(9,13))],treino_r[n],names_e)

###############################################################################
# generate index for single SR / ND vegetation index with best OOB score

index, table = selectRF_VIS(treino_e[list(range(13,treino_e.shape[0]))],treino_r[n],names_e)

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
sticks = [2,2,1,2] # change ticks spacing for each variable
plt.xticks(np.arange(0, 100, sticks[n]))
plt.yticks(np.arange(0, 100, sticks[n]))
#plot
plt.plot(pred,teste_r[n],'ro',ms=3)
plt.plot(pred, fiter(pred), 'b-', linewidth = 3)
plt.plot(linha, linha, 'k', linewidth = 2, linestyle = '--')

#save
# change the directory below to save a figure file
# plt.savefig('C:/Users/gerez/Desktop/'+valnamesave+regtype+names_r[n], dpi=300, bbox_inches='tight')



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
plt.plot(modelo.fittedvalues,modelo.resid,'ko')
plt.plot([modelo.fittedvalues.min(),modelo.fittedvalues.max()],np.zeros(2),'r--')

# real values vs fitted values
plt.plot(modelo.fittedvalues,modelo.model.endog,'ko')
mini = min(modelo.model.endog.min(),modelo.fittedvalues.min())
maxi = max(modelo.model.endog.max(),modelo.fittedvalues.max())
plt.plot([mini,maxi],[mini,maxi],'r--')

# plots for single predictive variables at a time
# change the variable with the number (second argument)
sm.graphics.plot_regress_exog(modelo,0)


