xwx <- function(xtx_in, x, w) {
  
  # computes inverse of Z'Z, 
  # where Z = rows of X with non-zero weight
  
  # xtx_in = inverse of X'X
  # x = n x p covariate matrix X
  # w = length n vector of zero-one weight
  # w cannot be all-zero
  if (sum(w) == 0) stop('All rows have zero weight.')
  
  
  
  # case 1: no zero-weights; same as inverse of X'X
  if (prod(w) == 1) return(xtx_in)
  
  # case 2: number of zero-weight rows >= p
  # invert directly
  else if (sum(w) < ncol(x)) {
    z1 <- x[w == 1, , drop = F]
    return(ginv(t(z1) %*% z1))
  }
  
  # case 3: number of zero-weight rows < p
  # use woodbury identity to invert a smaller matrix
  else {
    z0 <- x[w == 0, , drop = F]
    A <- z0 %*% xtx_in
    B <- -A %*% t(z0)
    diag(B) <- diag(B) + 1
    return(xtx_in + t(A) %*% ginv(B) %*% A)
  }
}

xwy <- function(x, y, w) {
  
  # computes Z'y,
  # where Z are the rows of x with non-zero weight
  
  # x = n x p covariate matrix X
  # y = length n vector
  # w = length n vector of zero-one weight
  # w cannot be all-zero
  if (sum(w) == 0) stop('All rows have zero weight.')
  
  # set zero-weight entries of y to be zero
  y[w == 0] <- 0
  return(t(x) %*% y)
}


#library(MASS)
#x <- matrix(runif(5 * 1), 5, 1)
#xtx_in <- solve(t(x) %*% x)
#w <- c(1,0,0,0,0)
#y <- runif(5)

#xwx(xtx_in, x, w)
#xwy(x, y, w)
#xwx(xtx_in, x, w) %*% xwy(x, y, w)
