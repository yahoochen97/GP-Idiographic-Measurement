args = commandArgs(trailingOnly=TRUE)
options(show.error.locations = TRUE)

TYPEs = c("graded_uni", "graded_multi", "gpcm_uni", "gpcm_multi",
          "sequential_uni", "sequential_multi")

if (length(args)==0) {
  SEED = 20
  n = 10
  m = 20
  horizon = 30
  RANK = 5
  TYPE = "graded_uni"
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

# reshape to 2d matrix
C = length(unique(array(mirt_data, c(n*horizon*m))))
dim(mirt_data) = c(n*horizon,m)
dim(train_mask) = c(n*horizon,m)
train_data = mirt_data
test_data = mirt_data
train_data[train_mask==0] = NA
test_data[train_mask==1] = NA

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
mirt_fit <- mirt(data = data.frame(train_data), 
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
log_lik = mirt_fit@Fit$logLik
BIC = mirt_fit@Fit$BIC
if(UNI=="uni"){
  thetas = array(fscores(mirt_fit), c(n,horizon))
} else{
  thetas = array(as.vector(fscores(mirt_fit)), c(n,horizon, RANK))
}

get_latent_f = function(as, theta, bs){
  # set na as very extreme number
  if(MODEL_NAME=="sequential"){
    bs[is.na(bs)] = -1000
  } else{
    if(as>0){
      bs[is.na(bs)] = 1000
    } else{
      bs[is.na(bs)] = -1000
    }
  }
  
  # compute latent f
  if(MODEL_NAME=="graded"){
    f = as*(theta-bs)
  } else if(MODEL_NAME=="gpcm"){
    f = as*(theta-bs)
    f = cumsum(f)
  } else if(MODEL_NAME=="sequential"){
    f = as*theta-bs
  }
  return(f)
}

# predict test observations and likelihood
train_acc = c()
train_ll = c()
test_acc = c()
test_ll = c()
dim(test_data) = c(n, horizon, m)
dim(train_data) = c(n, horizon, m)
for(i in 1:n){
  for(j in 1:m){
    for(h in 1:horizon){
      if(UNI=="uni"){
        # tmp = coefs[j]*(thetas[i,j]-coefs[j,(2):C])
        tmp = get_latent_f(coefs[j],thetas[i,h],coefs[j,2:C])
      } else{
        tmp = 0
        for(r in 1:RANK){
          # tmp = tmp + coefs[j,r]*(thetas[i,j,r]-coefs[j,(RANK+1):(RANK+C-1)])
          tmp = tmp + get_latent_f(coefs[j,r],thetas[i,h,r],coefs[j,(RANK+1):(RANK+C-1)])
        }
      }
      if( MODEL_NAME=="graded"){
        tmp = exp(tmp)/sum(exp(tmp))
        ps = c(1-tmp[1])
        for(c in 1:(C-2)){
          ps = c(ps, tmp[c]-tmp[c+1])
        }
        ps = as.vector(c(ps, tmp[C-1]))
      } else if (MODEL_NAME=="gpcm"){
        tmp = c(0, tmp)
        ps = as.vector(exp(tmp)/sum(exp(tmp)))
      } else if (MODEL_NAME=="sequential"){
        tmp = plogis(tmp)
        ps = c(tmp,1)*c(1,cumprod(1-tmp))
      }
      pred_y = which.max(ps)
      if(!is.na(test_data[i,h,j])){
        test_acc = c(test_acc, pred_y==test_data[i,h,j])
        if(ps[test_data[i,h,j]]<=-1e-6){
          print(paste(i,"_",h,"_",j,sep=""))
        }
        test_ll = c(test_ll, log(1e-6+ps[test_data[i,h,j]]))
      }else{
        train_acc = c(train_acc, pred_y==train_data[i,h,j])
        if(ps[train_data[i,h,j]]<=-1e-6){
          print(paste(i,"_",h,"_",j,sep=""))
        }
        train_ll = c(train_ll, log(1e-6+ps[train_data[i,h,j]]))
      }
    }
  }
}

train_acc = mean(train_acc)
train_ll = mean(train_ll[!is.na(train_ll)])
test_acc = mean(test_acc)
test_ll = mean(test_ll[!is.na(test_ll)])

save(loadings, correlation_matrix, log_lik, BIC, thetas,train_acc, train_ll,test_acc, test_ll,
     file=paste("./results/synthetic/", TYPE,"_", HYP, ".RData" , sep=""))

# quit()