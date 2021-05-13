library(stream)
library(animation)
library(gganimate)
library(ggplot2)

start_time <- Sys.time()

# library(clv)

## Loading Data into Stream (DSD)
# stream <- DSD_ReadCSV("C:/Users/neshragh/ecounter/Affinity_Sample_SPY/Luke/Python/ecounter.csv", sep=",", header = TRUE, loop=TRUE)
stream <- DSD_ReadCSV("C:/Users/neshragh/OneDrive - University of New Brunswick/UNB_thesis_Work/Affinity_Sample_SPY/2-Dataset_Wifi/indoor localization dataset/StreamingKmean-R/june20.csv", sep=",", header = TRUE, loop=TRUE)


# C:/Users/neshragh/OneDrive - University of New Brunswick/NB_thesis_Work/Affinity_Sample_SPY/1-Dataset_ecounter/StreamKmeansR/firstweekinterven.csv
# alldat <- read.csv(file = 'C:/Users/neshragh/ecounter/Affinity_Sample_SPY/Luke/Python/hi.csv', header = TRUE)



# win_km <- DSC_TwoStage(
#   micro=DSC_Window(horizon = 30),
#   macro=DSC_Kmeans(k=6)
# )
# win_km
# update(win_km, stream, 6422)




win_km <- DSC_TwoStage(
  micro=DSC_Kmeans(k=20),
  # micro=DSC_Window(horizon=6500),
  macro=DSC_Kmeans(k=6)
)
update(win_km, stream, 6500)# the distribution of data is around 14000, BOVE THIS GIVES US SMALLER SAMPLE AREA
plot(win_km, type="both", main="Streaming K-means:May 27 - June 2" )





# 
# #### Creating the DSC object
## sliding window #########################
# win_km <- DSC_TwoStage(
#   # micro=DSC_Kmeans(k=31),
#   micro=DSC_Window(horizon=100),
#   macro=DSC_Kmeans(k=7)
# )
# update(win_km, stream, 1000)# the distribution of data is around 14000, BOVE THIS GIVES US SMALLER SAMPLE AREA
# plot(win_km, stream,type="both", main="Streaming K-means:e-counter dataset" )
################## ALL DATA ###################################################################################
# #it should be (all data points)
# win_km <- DSC_TwoStage(
#   micro=DSC_Window(horizon = 18000, lambda = 0.00),
#   macro=DSC_Kmeans(k=7)
# )
# update(win_km, stream, 19000)# the distribution of data is around 14000, BOVE THIS GIVES US SMALLER SAMPLE AREA
# plot(win_km, type="both", main="Streaming K-means:Wifi dataset" ,xlim=c(0,140), ylim=c(0,18))

######################################################## RANDOM sampling ####################################
# win_km <- DSC_TwoStage(
#   micro=DSC_Sample(k = 100, biased = TRUE), # Random SAmpling
#   macro=DSC_Kmeans(k=5)
# )
# for (i in 1:85){
#   print(i)
# update(win_km, stream, 100)
# plot(win_km, type="both", main="Streaming K-means:Wifi dataset" ,xlim=c(0,140), ylim=c(0,18))
# }
######################################


# #animate_cluster(win_km, stream,n=64000, xlim=c(0,160), ylim=c(0,160))
# m <- read.csv(file = 'C:/Users/neshragh/ecounter/Affinity_Sample_SPY/Luke/Python/cleaned.csv', header = TRUE)
###############################################################################################################
###evaluate################################
evaluate(win_km, stream, assign="macro")
############################################
#plotting
# plot(alldat,add=TRUE)
# qplot(x=alldat$SPACEID,y=alldat$PHONEID)
###############################################
library(pryr)
mem_used()
## a way to time an R expression: system.time is preferred
ptm <- proc.time()
proc.time() - ptm
mem_used()
#memory.size()
# index.DB(stream, cl, d=NULL, centrotypes="centroids", p=2, q=2)

# for (i in 1:50) mad(stats::runif(500))


# library (cluster)
# library (vegan)
# data(varespec)
# dis = vegdist(varespec)
# res = pam(dis,3) # or whatever your choice of clustering algorithm is
# sil = silhouette (res$clustering,dis) # or use your cluster vector
# windows() # RStudio sometimes does not display silhouette plots correctly
# plot(sil)

# sil = silhouette (win_km$clustering,dsap)

##########################################running tiem
# sleep_for_a_minute <- function() { Sys.sleep(60) }
# evaluate(win_km, stream, assign="macro")
end_time <- Sys.time()
memory.size()
tiiiimmm <- end_time - start_time
######################################################
close_stream(stream)
