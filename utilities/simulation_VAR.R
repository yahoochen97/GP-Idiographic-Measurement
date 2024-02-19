# install.packages("dynr")
args = commandArgs(trailingOnly=TRUE)
options(show.error.locations = TRUE)

if (length(args)==0) {
  SEED = 20
  n = 10
  m = 20
  horizon = 30
  RANK = 5
  TYPE = "TVAR"
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
library(mgm)
set.seed(SEED)

# load data
HYP = paste("n", n, '_m', m, '_t', horizon, '_rank', RANK, '_SEED', SEED, sep="")
data = read.csv(paste("data/synthetic/data_", HYP,'.csv', sep=""))

# build n*horizon*m 3d matrix
DSEM_data = array(array(0, n*horizon*m), c(n,horizon,m))
train_mask = array(array(0, n*horizon*m), c(n,horizon,m))
for(i in 1:n){
  for(j in 1:m){
    for(h in 1:horizon){
      tmp = data[data$unit==(i-1) & data$item==(j-1) & data$time==(h-1),]
      DSEM_data[i,h,j] = tmp$y
      if (tmp$train){ train_mask[i,h,j]=1 }
    }
  }
}

# reshape to 2d matrix
C = length(unique(array(DSEM_data, c(n*horizon*m))))
dim(DSEM_data) = c(n*horizon,m)
dim(train_mask) = c(n*horizon,m)
train_data = DSEM_data

tvvar_obj <- tvmvar(data = DSEM_data,
                    type = rep("g", m),
                    level = rep(1, m), 
                    lambdaSel = "CV",
                    estpoints =  seq(0, 1, length = horizon),
                    timepoints = rep(1:horizon, each=n)/horizon,
                    bandwidth = 0.25,
                    lags = 1,
                    scale = TRUE,
                    pbar = TRUE)

correlation_matrix = tvvar_obj$wadj[,,1,1]*tvvar_obj$signs[,,1,1]
correlation_matrix[is.na(correlation_matrix)] = 0
correlation_matrix = (correlation_matrix + t(correlation_matrix)) / 2
if (correlation_matrix[1,1]<0){
  correlation_matrix = -correlation_matrix
}
loadings = NULL

pred_obj <- predict(object = tvvar_obj, 
                    data =DSEM_data, 
                    errorCon = c("R2", "RMSE"),
                    tvMethod = "weighted")

pred_y = pred_obj$predicted
pred_y = (pred_y-min(pred_y)+1)/(max(pred_y)-min(pred_y))*(C-1)
pred_y = round(pred_y, digits = 0)

train_acc = mean(pred_y==DSEM_data[2:300,])
train_ll = mean(log(dnorm(pred_y-DSEM_data[2:300,])))
log_lik = NULL
thetas = NULL
BIC = NULL

test_acc = NULL
test_ll = NULL

save(loadings, correlation_matrix, log_lik, BIC, thetas,train_acc, train_ll,test_acc, test_ll,
     file=paste("./results/synthetic/", TYPE,"_", HYP, ".RData" , sep=""))

