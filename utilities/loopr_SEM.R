# install.packages("dynr")
args = commandArgs(trailingOnly=TRUE)
options(show.error.locations = TRUE)

if (length(args)==0) {
  RANK = 5
}
if (length(args)==1){
  RANK = as.integer(args[1])
}

TYPE = "SEM"

R_path="~/R/x86_64-redhat-linux-gnu-library/4.0"
.libPaths(R_path)
library("lavaan")
set.seed(SEED)

# load data
data = read.csv("./data/loopr_data.csv")
data = subset(data, select = -c(1) )
n = nrow(data)
m = ncol(data)
C = 5

colnames(data) = unlist(lapply(1:m,function(i) paste("y",as.character(i), sep="")))

if (RANK==1){
  myModel <- '
    q1 =~ y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12 + y13 + y14 + y15 + y16 + y17 + y18 + y19 + y20 + y21 + y22 + y23 + y24 + y25 + y26 + y27 + y28 + y29 + y30 + y31 + y32 + y33 + y34 + y35 + y36 + y37 + y38 + y39 + y40 + y41 + y42 + y43 + y44 + y45 + y46 + y47 + y48 + y49 + y50 + y51 + y52 + y53 + y54 + y55 + y56 + y57 + y58 + y59 + y60
'
}
if (RANK==2){
  myModel <- '
  q1 + q2 =~ y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12 + y13 + y14 + y15 + y16 + y17 + y18 + y19 + y20 +
   y21 + y22 + y23 + y24 + y25 + y26 + y27 + y28 + y29 + y30 + y31 + y32 + y33 + y34 + y35 + y36 + y37 + y38 + y39 + y40 +
   y41 + y42 + y43 + y44 + y45 + y46 + y47 + y48 + y49 + y50 + y51 + y52 + y53 + y54 + y55 + y56 + y57 + y58 + y59 + y60
'
}
if (RANK==3){
  myModel <- '
  q1 + q2 + q3 + q4 + q5 =~
   y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12 + y13 + y14 + y15 + y16 + y17 + y18 + y19 + y20 +
   y21 + y22 + y23 + y24 + y25 + y26 + y27 + y28 + y29 + y30 + y31 + y32 + y33 + y34 + y35 + y36 + y37 + y38 + y39 + y40 +
   y41 + y42 + y43 + y44 + y45 + y46 + y47 + y48 + y49 + y50 + y51 + y52 + y53 + y54 + y55 + y56 + y57 + y58 + y59 + y60
'
}
if (RANK==4){
  myModel <- '
  q1 + q2 + q3 + q4 = ~
   y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12 + y13 + y14 + y15 + y16 + y17 + y18 + y19 + y20 +
   y21 + y22 + y23 + y24 + y25 + y26 + y27 + y28 + y29 + y30 + y31 + y32 + y33 + y34 + y35 + y36 + y37 + y38 + y39 + y40 +
   y41 + y42 + y43 + y44 + y45 + y46 + y47 + y48 + y49 + y50 + y51 + y52 + y53 + y54 + y55 + y56 + y57 + y58 + y59 + y60 
'
}
if (RANK==5){
  myModel <- '
  q1 + q2 + q3 + q4 + q5 =~
   y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12 + y13 + y14 + y15 + y16 + y17 + y18 + y19 + y20 +
   y21 + y22 + y23 + y24 + y25 + y26 + y27 + y28 + y29 + y30 + y31 + y32 + y33 + y34 + y35 + y36 + y37 + y38 + y39 + y40 +
   y41 + y42 + y43 + y44 + y45 + y46 + y47 + y48 + y49 + y50 + y51 + y52 + y53 + y54 + y55 + y56 + y57 + y58 + y59 + y60
'
}

fit <- cfa(model = myModel, 
           data = data) 
correlation_matrix = matrix(0, nrow = m, ncol=m)
for (i in 1:m){
  for (j in 1:m){
    correlation_matrix[i,j] = fitted(fit)$cov[i,j]
    correlation_matrix[j,i] = fitted(fit)$cov[j,i]
  }
}

loadings = parameterEstimates(fit)[1:(m*RANK),"est"]

correlation_matrix = loadings %*% t(loadings)
write.csv(loadings, file=paste("./results/loopr/", TYPE,"_", RANK, ".csv" , sep=""))
# quit()

