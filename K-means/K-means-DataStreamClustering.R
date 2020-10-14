library(stream)
library(animation)
library(gganimate)
library(ggplot2)
rm(list = ls())
## Loading Data into Stream (DSD)
stream <- DSD_ReadCSV("C:/Users/neshragh/ecounter/Affinity_Sample_SPY/R/R/12.csv", sep=",", header = TRUE, loop=TRUE)


#### Creating the DSC object
win_km <- DSC_TwoStage(
  micro=DSC_Window(horizon = 150, lambda=0),
  macro=DSC_Kmeans(k=4)
)
win_km
update(win_km, stream, 90)
plot(win_km, stream, type="both")
# x = array(1:143)
# for (val in x){
#   update(win_km, stream)
#   plot(win_km, stream, type="both")
# }

  
#animate_cluster(win_km, stream,n=64000, xlim=c(0,6), ylim=c(0,6))

library(pryr)
mem_used()
## a way to time an R expression: system.time is preferred
ptm <- proc.time()
for (i in 1:50) mad(stats::runif(500))
proc.time() - ptm


library (cluster)
library (vegan)
data(varespec)
dis = vegdist(varespec)
res = pam(dis,3) # or whatever your choice of clustering algorithm is
sil = silhouette (res$clustering,dis) # or use your cluster vector
windows() # RStudio sometimes does not display silhouette plots correctly
plot(sil)




