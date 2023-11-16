args = commandArgs(trailingOnly=TRUE)
options(show.error.locations = TRUE)

TYPEs = c("GIMME")

if (length(args)==0) {
  SEED = 1
  n = 10
  m = 20
  horizon = 30
  RANK = 5
  TYPE = "GIMME"
}
if (length(args)==6){
  n = as.integer(args[1])
  m = as.integer(args[2])
  horizon = as.integer(args[3])
  RANK = as.integer(args[4])
  SEED = as.integer(args[5])
  TYPE = args[6]
}

R_path="~/R/x86_64-redhat-linux-gnu-library/4.0"
.libPaths(R_path)
library(mirt)
library(dplyr)
set.seed(SEED)

# load data
HYP = paste("n", n, '_m', m, '_t', horizon, '_rank', RANK, '_SEED', SEED, sep="")
data = read.csv(paste("data/synthetic/data_", HYP,'.csv', sep=""))

# build n*horizon*m 3d matrix
mirt_data = array(array(0, n*horizon*m), c(n,horizon,m))
train_mask = array(array(0, n*horizon*m), c(n,horizon,m))
for(i in 1:n){
  for(j in 1:m){
    for(h in 1:horizon){
      tmp = data[data$unit==(i-1) & data$item==(j-1) & data$time==(h-1),]
      mirt_data[i,h,j] = tmp$y
      if (tmp$train){ train_mask[i,h,j]=1 }
    }
  }
}

train_data = mirt_data
test_data = mirt_data
train_data[train_mask==0] = NA
test_data[train_mask==1] = NA

# reshape to list of 2d matrix
C = length(unique(array(mirt_data, c(n*horizon*m))))
data = list()
for(i in 1:n){
  data[[paste("group_1_",i,sep="")]] = data.frame(train_data[i,,])
  for(j in 1:m){
    tmp = train_data[i,,j]
    if(var(tmp[!is.na(tmp)])<=0.1){
      data[[paste("group_1_",i,sep="")]][,j] = 
        data[[paste("group_1_",i,sep="")]][,j] + 0.3*rnorm(horizon)
    }
  }
}

# define GIMME model
factor_strings = c()
for(j in 1:m){
  if(j %% (m/RANK)){
    factor_strings = c(factor_strings, paste('X',j, ' ~ X', j+1, 'lag ', sep=''))
  }
}
s = paste(factor_strings, collapse="\n")
s = 'X1~X2
    X2~X4lag'

# fit GIMME model
fit <- gimmeSEM(data = data,
                out = "./results/synthetic",
                paths = s)