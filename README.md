# Binary classifications models.
# Customers fraud predictions: Machine Learning Classes.
<p align="center"><img src = "https://upload.wikimedia.org/wikipedia/commons/f/f3/Apache_Spark_logo.svg">

Contains the codes for Spark's Scala API classifications methods based on the MLLib Library Dataframe API.

This repository was built to respond the requierement of predicting the fraud class probability regarding a group of significant variables for a particular set of customers. The result, the confussion matrix of each method, can be written into HDFS by implementing a few lines of extra code. 
The data loaded into the Spark session arrives from HDFS and it's design came from a Cloudera Impala ETL. The Impala SQL is not currently available in this repository. The user can define the elements inside this repository as a functional API that allows the estimation of 5 different types of statistical methods: decision tres, adaptative boosting based on trees, random forest, logistic regression and naive-Bayes for binary classification (it's possible to consider the multi-level classification case). The performance of calculations depends on the Spark's pipeline stages model.

The correct order for understanding this **_"development"_** goes as the following:

### 1. Modules (modules_fraud_model.scala): :books:
  Import the Scala classes needed to transform data and estimate the different methods. The latter is defined through the Cross-validation model using the AUPR metric (area under precision-recall curve). It's .
  
### 2. Data set (data_set_fraud_model.scala): :floppy_disk:
  Import the data set, stored in HDFS, into Spark session. The data structure is org.apache.spark.sql.DataFrame. After user calling the the data set is parallelized with the **cache()** instruction. This one is extremely important since calculations are running in parallel using the resources of the cluster.
  
### 3. Fraud model (fraud_model.scala): :space_invader:
  The final implementation where the API's **_kernel_** dwells. The user can call all the 5 methods in the usual _object-oriented_ form previous **Modules** asnd **Data set** call. Inside every method there's a routine performed by the pipeline model; the stages are: _vector assembler_, _minmax scaler_ (it's possible to invoke others), _the machine learning method_ and the _pipeline_ itself.
  
The last file **_Execution.scala_** contains just the code lines to execute on the **_Spark shell_**.

Finally I must say that the Spark's version used was the **_2.2.0_**. :white_check_mark: :white_check_mark: :white_check_mark:

######  **_Considerations_**:
###### 1. The current way to access Spark is through **_SFTP_** connection. **MobaXterm** is an alternative to doing so, but it has no support, indeed, is has IT restrictions, however, it's our only tool.
###### 2. The source codes in this repository can not be executed inside the GitHub platform.
###### 3. The updates published here are for the good of the version control. The new versions themselves don't migrate directly to the Landing Zone. The user has to copy these new versions into the node using e.g. WinSPC or FileZilla. 
