args = commandArgs(trailingOnly=TRUE)
options(show.error.locations = TRUE)

TYPEs = c("graded_multi",  "gpcm_multi", "sequential_multi")

if (length(args)==0) {
  RANK = 1
  TYPE = "gpcm_multi"
}
if (length(args)==2){
  RANK = as.integer(args[1])
  TYPE = args[2]
}

R_path="~/R/x86_64-redhat-linux-gnu-library/4.0"
.libPaths(R_path)
library(mirt)
library(dplyr)

# load data
data = read.csv("./data/loopr_data.csv")
data = subset(data, select = -c(1) )
n = nrow(data)
m = ncol(data)
C = 5

# define mirt model
factor_strings = c()
for (r in 1:RANK){
  factor_strings = c(factor_strings, paste('F',r, ' = ', m/RANK*(r-1)+1, '-', m/RANK*r, sep=''))
}

s = paste(factor_strings, collapse="\n")
factor_model <- mirt.model(s)

MODEL_NAME = unlist(strsplit(TYPE, "_"))[1]
UNI = unlist(strsplit(TYPE, "_"))[2]
EM_method = "QMCEM"
if(UNI=="uni"){
  factor_model = 1
  EM_method="EM"
}

# fit mirt model
mirt_fit <- mirt(data = data.frame(data), 
                   model = factor_model,
                   itemtype = MODEL_NAME,
                   method = EM_method,
                   optimizer = "nlminb",
                   verbose = FALSE)

if(MODEL_NAME=="sequential"){
  coefs = coef(mirt_fit, simplify = TRUE)$items
} else{
  coefs = coef(mirt_fit, IRTpars = TRUE, simplify = TRUE)$items
}

if(UNI=="uni"){
  loadings = matrix(as.vector(coefs[,1]))
} else{
  loadings = matrix(as.vector(coefs[,1:RANK]), nrow=m)
}

correlation_matrix = loadings %*% t(loadings)
write.csv(loadings, file=paste("./results/loopr/", TYPE,"_", RANK, ".csv" , sep=""))
# quit()