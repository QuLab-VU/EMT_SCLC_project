library(nsprcomp)
nnpca_r <- function(expr, ncomp){
    d = nsprcomp(expr,nneg=TRUE,ncomp=ncomp)
    return(d)
}
