# Self-training function with logistic regression as classifier

library(docstring)

selftraining.lr <- function(formula, outcome, data, thr.conf, max.iter) {
  
  #' @title Self-training function with logistic regression
  #' @description This function performs the semi-supervised method of self-training in which a logistic    
  #' regression model is used as classifier and is iteratively retrained on (pseudo-)labeled data.
  #' The function includes adjustable parameters for the confidence threshold and maximum number of iterations.
  #' The stopping criterion of the function is defined by the indicated number of maximum iterations or when all 
  #' outcomes have been (pseudo-)labeled.
  #' @param formula formula of predictors and outcome used in logistic regression 
  #' @param outcome character string of the name of the outcome in the data
  #' @param data dataframe with column names containing the outcome and predictors
  #' @param thr.conf numeric value (between 0-1) indicating the confidence threshold to add newly labelled data
  #' @param max.iter numeric value for the maximum number of iterations
  #' @return Returns a list consisting of: logistic regression model results of last iteration, binary outcome 
  #' results of last iteration, number of iterations performed, and numbers of added pseudolabels per iteration
  
  # start with iteration at zero
  iteration <- 0 
  # to store the number of added pseudolabels per iteration
  additions <- c() 
  
  # repeat the model development using logistic regression and each time add only the outcomes 
  # that are more certain than the specified confidence threshold to the dataset
  repeat { 
    adds <- 0
    model <- glm(formula = formula, family = "binomial", data = data) 
    
    for (j in 1:nrow(data)) { 
      if (is.na(data[outcome][j,])) {
        probability <- predict(model, data[j,], type = "response") 
        if (probability > thr.conf) { 
          # if the probability is larger than the threshold, replace the NA with 1
          data[outcome][j,] <- 1 
          adds <- adds + 1
        } else if (probability < 1-thr.conf) {
          # and if the probability is smaller than 1-threshold, replace the NA with 0
          data[outcome][j,] <- 0 
          adds <- adds + 1
        }
      }
    }
    iteration <- iteration + 1 
    additions[iteration] <- adds
    # stop the repeated process if reaching max.iter
    if (iteration == max.iter) { 
      model <- glm(formula = formula, family = "binomial", data = data) 
      break
    # or stop the repeated process if all data have been (pseudo-)labeled
    } else if (sum(is.na(data[,outcome])) == 0) { 
      model <- glm(formula = formula, family = "binomial", data = data)
      break
    }
  }
  results <- list(model = model, 
                  new.outcome = data[, outcome], 
                  iteration = iteration,
                  additions = additions)
  return(results)
}


# Test example with sample data:
set.seed(1)
selftraining.lr(formula = "outcome ~ predictor1 + predictor2",
                outcome = "outcome",
                data = data.frame(outcome = sample(c(c(rep(NA,10)), c(rep(1,20)), c(rep(0,70)))),
                                  predictor1 = sample(x = 0:50, size = 100, replace = TRUE),
                                  predictor2 = sample(x = 0:30, size = 100, replace = TRUE)), 
                thr.conf = 0.8, 
                max.iter = 5)

# Output of test example:
#
# $model
# 
# Call:  glm(formula = formula, family = "binomial", data = data)
# 
# Coefficients:
#   (Intercept)   predictor1   predictor2  
# -0.37201     -0.03028     -0.00872  
#
# Degrees of Freedom: 91 Total (i.e. Null);  89 Residual
# (8 observations deleted due to missingness)
# Null Deviance:	    96.34 
# Residual Deviance: 92.91 	AIC: 98.91
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
