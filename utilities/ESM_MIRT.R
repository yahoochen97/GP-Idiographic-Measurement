args = commandArgs(trailingOnly=TRUE)
options(show.error.locations = TRUE)
# ESM MIRT AND SEM for in-sample comparison

TYPEs = c("graded_mult", "gpcm_multi","sequential_multi", "sem")

if (length(args)==0) {
  RANK = 5
  TYPE = "sem"
}
if (length(args)==2){
  RANK = as.integer(args[1])
  TYPE = args[2]
}

R_path="~/R/x86_64-redhat-linux-gnu-library/4.0"
.libPaths(R_path)
library(mirt)
library(dplyr)
library(lavaan)

# load data
data = read.csv("./data/GP_ESM.csv")
m = 45


if(TYPE=="sem"){
  C = 5
  train_data = data[,1:m]
  colnames(train_data) = unlist(lapply(1:m,function(i) paste("y",as.character(i), sep="")))
  
  if (RANK==5){
    myModel <- '
    q1 =~ y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9
    q2 =~ y10 + y11 + y12 + y13 + y14 + y15 + y16 + y17 + y18
    q3 =~ y19 + y20 + y21 + y22 + y23 + y24 + y25 + y26 + y27
    q4 =~ y28 + y29 + y30 + y31 + y32 + y33 + y34 + y35 + y36
    q5 =~ y37 + y38 + y39 + y40 + y41 + y42 + y43 + y44 + y45
    '
  }
  fit <- sem(model = myModel, data = train_data, missing = "ML")
  log_lik = fitMeasures(fit)[["logl"]]
  BIC = BIC(fit)
  
  loadings = parameterEstimates(fit)[1:(m*RANK),"est"]
  loadings = matrix(loadings, nrow = m)
  correlation_matrix = loadings %*% t(loadings)
  
  thetas = predict(fit, newdata = data)
  pred_y = matrix(0, nrow = nrow(data), ncol=m)
  for (i in 1:nrow(data)){
    pred_y[i,] = rep( thetas[i,], each = m/5) * loadings
  }
  
  pred_y = (pred_y-min(pred_y)+1)/(max(pred_y)-min(pred_y))*(C-1)
  pred_y = round(pred_y, digits = 0)
  
  train_acc = mean(pred_y[train_mask==1]==train_data[train_mask==1])
  train_ll = log_lik / n / m / horizon
  
  
  write.csv(loadings, file=paste("./results/loopr/", TYPE,"_", RANK, ".csv" , sep=""))
}else{
  train_data = data
  test_data = data
  test_data[!is.na(test_data)] = NA
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
  mirt_fit <- mirt(data = data.frame(train_data[,1:m]), 
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
    thetas = array(fscores(mirt_fit), c(nrow(data),1))
  } else{
    thetas = array(as.vector(fscores(mirt_fit)), c(nrow(data), RANK))
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
  for(i in 1:nrow(data)){
    for(j in 1:m){
        if(UNI=="uni"){
          tmp = get_latent_f(coefs[j],thetas[i],coefs[j,2:C])
        } else{
          tmp = 0
          for(r in 1:RANK){
            tmp = tmp + get_latent_f(coefs[j,r],thetas[i,r],coefs[j,(RANK+1):(RANK+C-1)])
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
        if(!is.na(test_data[i,j])){
          test_acc = c(test_acc, pred_y==test_data[i,j])
          test_ll = c(test_ll, log(1e-6+ps[test_data[i,j]]))
        }
        if(!is.na(train_data[i,j])){
          train_acc = c(train_acc, pred_y==train_data[i,j])
          train_ll = c(train_ll, log(1e-6+ps[train_data[i,j]]))
        }
      }
    }
  
  train_acc = mean(train_acc)
  train_ll = mean(train_ll[!is.na(train_ll)])
  # test_acc = mean(test_acc)
  # test_ll = mean(test_ll[!is.na(test_ll)])
}

print(RANK)
print("train acc")
print(train_acc)
print("train ll")
print(train_ll)

save(train_acc, train_ll,
     file=paste("./results/GP_ESM/baselines/", TYPE,"_", RANK, ".RData" , sep=""))
