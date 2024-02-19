# install.packages("dynr")
args = commandArgs(trailingOnly=TRUE)
options(show.error.locations = TRUE)

if (length(args)==0) {
  RANK = 1
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
