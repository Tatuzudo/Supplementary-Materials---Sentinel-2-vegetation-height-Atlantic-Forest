###########
# Folders #
###########

Time series:
A time series of the Spearman’s ranked correlation between the LiDAR height mean variable (LHmean) and all available level-1C (TOA) Sentinel 2 imagery. This procedure was made using Google Earth Engine infrastructure and python. Since field data was distributed in a larger region, it was prone to cloud interference and, thus, not used in the time series. Additionally, the Spearman’s value was chosen instead of Pearson’s for being less affected by outliers and for its ability to deal with non-linear monotonic relations. For convenience and based on preliminary correlation tests made with the 23/12/2016 image, only band 5 (705 nm) was used in the time series analysis. Only images without clouds in at least one of the Cantareira 1 and 2 areas were used. When both areas were clear, the r value of highest magnitude was considered.

Preliminary results:
The results of basic statistics for all the variables used in the study and correlations (Spearman's) with scatter plots for sentinel 2 bands with each other. All preliminary OLS models between vegetation variables and sentinel 2 data are also in this folder. They are separated in single bands, vegetation indices, simple ratio indices between 2 bands and normalized difference indices between 2 bands. Dispersion plots with OLS r² with a first degree polynomial fit were made for single bands and vegetation indices. The plots for simple ratio and normalized difference indices show the OLS r² values for each index for a single vegetation variable per plot. The bands used for making each index are shown in the x and y axis. Because of the great number of variables involved, no dispersion plots were made in this case. This procedure was repeated for each of the four Sentinel 2 L2A images used.

Height Predictions:
Scater plots showing the predictive capability of each regression model created, with error values and a first degree polynomial fit to observe biases. There are also figures that help to visualise the methods of validation used for the models identifying exactly which data set (Sentinel 2 images and LiDAR data) was used for what. The figures are separated by model type  (Random Forest - RF, Ordinary Least Squares - OLS or Weighted Least Square - WLS) is also shown in each figure. 

Data:
All the data used in the study as .csv files, and the python code to reproduce the results. This doesn't include raw Sentinel 2 or LiDAR data. The python script should work as is, but the paths to the .csv and .tif files may have to be edited manually. The script is detailed as much as possible to provide an easy way to reproduce all the results shown in our work. It's recommended to run the script using Spyder ide (Anaconda). All variables can be found at the "Cantareira 1" and "Cantareira 2" folders. All Sentinel 2 images are available for download in https://scihub.copernicus.eu/dhus/#/home. All LiDAR data was kindly provided by the sustainable landscapes project and cannot be obtained without request. For more information, visit the sustainable landscapes project WebGis at https://www.paisagenslidar.cnptia.embrapa.br/webgis/.

To reproduce all results, use the file located at Data/Code.py
