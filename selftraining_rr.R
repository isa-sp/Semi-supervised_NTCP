# Self-training function with ridge regression as classifier

library(docstring)
library(glmnet)

selftraining.rr <- function(x, y, thr.conf, max.iter) {
  
  #' @title Self-training function with ridge regression
  #' @description This function performs the semi-supervised method of self-training in which a ridge regression
  #' model is used as classifier and is iteratively retrained on (pseudo-)labeled data.
  #' The function includes adjustable parameters for the confidence threshold and maximum number of iterations. The stopping 
  #' criterion of the function is defined by the indicated number of maximum iterations or when all outcomes have
  #' been (pseudo-)labeled.
  #' @param x matrix of the predictors 
  #' @param y vector of the outcome
  #' @param thr.conf numeric value (between 0-1) indicating the confidence threshold to add newly labelled data
  #' @param max.iter numeric value for the maximum number of iterations
  #' @return Returns a list consisting of: ridge regression model results of last iteration, binary outcome 
  #' results of last iteration, number of iterations performed, and numbers of added pseudolabels per iteration
  
  # start with iteration at zero
  iteration <- 0 
  # to store the number of added pseudolabels per iteration
  additions <- c() 
  
  # repeat the model development using ridge regression and each time add only the outcomes 
  # that are more certain than the specified confidence threshold to the dataset
  for (i in 1:max.iter) { 
    adds <- 0
    y.vector <- y[complete.cases(y)] 
    x.matrix <- x[-which(is.na(y)),] 
    
    model <- glmnet(x = x.matrix, 
                    y = y.vector,
                    alpha = 0,
                    family = "binomial")
    set.seed(16)
    ridge_cv <- cv.glmnet(x = x.matrix, 
                          y = y.vector, 
                          alpha = 0,
                          family = "binomial")
    best_model <- glmnet(x = x.matrix, 
                         y = y.vector, 
                         alpha = 0, 
                         lambda = ridge_cv$lambda.min,
                         family = "binomial")
    
    for (j in 1:length(y)) { 
      if (is.na(y[j])) { 
        probability <- as.numeric(predict(best_model, 
                                          x[j,], 
                                          s = ridge_cv$lambda.min,
                                          type = "response"))
        if (probability > thr.conf) { 
          # if the probability is larger than the threshold, replace the NA
          y[j] <- 1 
          adds <- adds + 1
        } else if (probability < 1-thr.conf) {
          # and if the probability is smaller than 1-threshold, replace the NA
          y[j] <- 0 
          adds <- adds + 1
        }
      }
      additions[i] <- adds
    }
    
    iteration <- iteration + 1 
    # stop the repeated process if reaching max.iter
    if (iteration == max.iter) { 
      model <- glmnet(x = x.matrix, 
                      y = y.vector,
                      alpha = 0,
                      family = "binomial")
      set.seed(16)
      ridge_cv <- cv.glmnet(x = x.matrix,
                            y = y.vector, 
                            alpha = 0,
                            family = "binomial")
      best_model <- glmnet(x = x.matrix,
                           y = y.vector, 
                           alpha = 0, 
                           lambda = ridge_cv$lambda.min,
                           family = "binomial")
      break
      # or stop the repeated process if all data have been labelled
    } else if (sum(is.na(y)) == 0) { 
      model <- glmnet(x = x.matrix, 
                      y = y.vector,
                      alpha = 0,
                      family = "binomial")
      set.seed(16)
      ridge_cv <- cv.glmnet(x = x.matrix,
                            y = y.vector, 
                            alpha = 0,
                            family = "binomial")
      best_model <- glmnet(x = x.matrix,
                           y = y.vector, 
                           alpha = 0, 
                           lambda = ridge_cv$lambda.min,
                           family = "binomial")
      break
    }
  }
  
  results <- list(best.model = best_model, 
                  new.outcome = y, 
                  iteration = iteration,
                  additions = additions)
  return(results)
}


# Test example with sample data:
set.seed(1)
selftraining.rr(x = data.matrix(data.frame(predictor1 = sample(x = 0:50, size = 100, replace = TRUE),
                                           predictor2 = sample(x = 0:30, size = 100, replace = TRUE))),
                y = sample(c(c(rep(NA,10)), c(rep(1,20)), c(rep(0,70)))),
                thr.conf = 0.8, 
                max.iter = 5)

# Output of test example:
#
# $best.model
#
# Call:  glmnet(x = x.matrix, y = y.vector, family = "binomial", alpha = 0,      lambda = ridge_cv$lambda.min) 
#
# Df %Dev Lambda
# 1  2 1.66 0.4726
#
# $new.outcome
# [1]  0  0 NA  0  0  0  1  0  0  0  0  1  0  0 NA  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  0  0  0  0  1  0  0  0 NA  1
# [41]  0  0 NA  0  1  1  0  0  0  0  0  0  1  0  0  0  0  1  1  0  1  0  0  0  0  1  0  0  0  0  1  1  0  0  0  0  0 NA  0  1
# [81]  0  0  0 NA  1  1  0  1 NA  0  0  0  1  0  0  0  0 NA  0  0
#
# $iteration
# [1] 5
#
# $additions
# [1] 2 0 0 0 0
