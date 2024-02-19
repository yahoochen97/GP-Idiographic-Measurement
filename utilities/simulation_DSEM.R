# install.packages("dynr")
args = commandArgs(trailingOnly=TRUE)
options(show.error.locations = TRUE)

if (length(args)==0) {
  SEED = 20
  n = 10
  m = 20
  horizon = 30
  RANK = 5
  TYPE = "DSEM"
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
library("lavaan")
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
test_data = DSEM_data
train_data[train_mask==0] = NA
test_data[train_mask==1] = NA

train_data = as.data.frame(DSEM_data)
colnames(train_data) = unlist(lapply(1:m,function(i) paste("y",as.character(i), sep="")))

myModel <- ' 
 # latent variables 
   l1 =~ y1 + y2 + y3 + y4 
   l2 =~ y5 + y6 + y7 + y8  
   l3 =~ y9 + y10 + y11 + y12  
   l4 =~ y13 + y14 + y15 + y16  
   l5 =~ y17 + y18 + y19 + y20 
'
fit <- sem(model = myModel, 
           data = train_data) 
correlation_matrix = matrix(0, nrow = m, ncol=m)
for (i in 1:m){
  for (j in 1:m){
    correlation_matrix[i,j] = fitted(fit)$cov[i,j]
    correlation_matrix[j,i] = fitted(fit)$cov[j,i]
  }
}

loadings = parameterEstimates(fit)[1:20,"est"]

log_lik = fitMeasures(fit)[["logl"]]
BIC = BIC(fit)
thetas = predict(fit, newdata = train_data)

pred_y = matrix(0, nrow = n*horizon, ncol=m)
for (i in 1:(n*horizon)){
  pred_y[i,] = rep( thetas[i,], each = m/5) * loadings
}

pred_y = (pred_y-min(pred_y)+1)/(max(pred_y)-min(pred_y))*(C-1)
pred_y = round(pred_y, digits = 0)

train_acc = mean(pred_y[train_mask==1]==train_data[train_mask==1])
train_ll = log_lik / n / m / horizon
test_ll = train_ll 
test_acc = NULL
test_ll = NULL

save(loadings, correlation_matrix, log_lik, BIC, thetas,train_acc, train_ll,test_acc, test_ll,
     file=paste("./results/synthetic/", TYPE,"_", HYP, ".RData" , sep=""))
