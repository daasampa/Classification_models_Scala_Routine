object models {
def lambda(dataframe : org.apache.spark.sql.DataFrame, feature : String) : Double = {
return (dataframe.filter(dataframe(feature) === 0).count().toDouble / dataframe.count().toDouble) * 100
}
def view_features(dataframe : org.apache.spark.sql.DataFrame) : org.apache.spark.sql.DataFrame = {
val features = dataframe.drop("documento", "segmento", "fraude").columns
val percentage_zeros = features.map(x => models.lambda(dataframe, x))
val results = sc.parallelize(features zip percentage_zeros).toDF("features", "percentage_zeros")
return results.orderBy(desc("percentage_zeros"))
}
def decision_tree(features:Array[String], percentage:Double, depth:Array[Int], folds:Int) : Unit = {
val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
val dt = new DecisionTreeClassifier().setLabelCol("fraude").setFeaturesCol("features")
val Array(trainingData,testData) = base_modelo.randomSplit(Array(percentage, 1 - percentage))
val pipeline = new Pipeline().setStages(Array(assembler, dt))
val paramGrid = new ParamGridBuilder().addGrid(dt.maxDepth, depth).addGrid(dt.impurity, Array("gini", "entropy")).build()
val evaluator = new BinaryClassificationEvaluator
evaluator.setMetricName("areaUnderPR").setLabelCol("fraude")
val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(folds)
trainingData.cache()
testData.cache()
val cvModel = cv.fit(trainingData)
val predictions = cvModel.transform(testData)
predictions.cache()
val results = predictions.withColumnRenamed("label", "fraude").withColumnRenamed("predicted_label", "prediction").groupBy("fraude", "prediction").agg(count("documento").alias("clientes")).orderBy("fraude", "prediction")
val table_name = ("results_decision_tree").mkString
results.createOrReplaceTempView(table_name)
spark.sql("drop table if exists proceso_seguridad_externa." ++ table_name)
spark.sql("create table if not exists proceso_seguridad_externa." ++ table_name ++ " stored as parquet as select * from " ++ table_name)
}
def ada_boost(features:Array[String], percentage:Double, depth:Array[Int], n_trees:Array[Int], folds:Int) : Unit = {
val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
val gbt = new GBTClassifier().setLabelCol("fraude").setFeaturesCol("features")
val Array(trainingData, testData) = base_modelo.randomSplit(Array(percentage, 1 - percentage))
val pipeline = new Pipeline().setStages(Array(assembler, gbt))
val paramGrid = new ParamGridBuilder().addGrid(gbt.maxDepth, depth).addGrid(gbt.maxIter, n_trees).addGrid(gbt.impurity, Array("gini", "entropy")).build()
val evaluator = new BinaryClassificationEvaluator
evaluator.setMetricName("areaUnderPR").setLabelCol("fraude")
val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(folds)
trainingData.cache()
testData.cache()
val cvModel = cv.fit(trainingData)
val predictions = cvModel.transform(testData)
predictions.cache()
val results = predictions.withColumnRenamed("label", "fraude").withColumnRenamed("predicted_label", "prediction").groupBy("fraude", "prediction").agg(count("documento").alias("clientes")).orderBy("fraude", "prediction")
val table_name = ("results_ada_boost").mkString
results.createOrReplaceTempView(table_name)
spark.sql("drop table if exists proceso_seguridad_externa." ++ table_name)
spark.sql("create table if not exists proceso_seguridad_externa." ++ table_name ++ " stored as parquet as select * from " ++ table_name)
}
def random_forest(features:Array[String], percentage:Double, depth:Array[Int], n_trees:Array[Int], folds:Int) : Unit = {
val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
val rf = new RandomForestClassifier().setLabelCol("fraude").setFeaturesCol("features")
val Array(trainingData, testData) = base_modelo.randomSplit(Array(percentage, 1 - percentage))
val pipeline = new Pipeline().setStages(Array(assembler, rf))
val paramGrid = new ParamGridBuilder().addGrid(rf.maxDepth, depth).addGrid(rf.numTrees, n_trees).addGrid(rf.impurity, Array("gini", "entropy")).build()
val evaluator = new BinaryClassificationEvaluator
evaluator.setMetricName("areaUnderPR").setLabelCol("fraude")
val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(folds)
trainingData.cache()
testData.cache()
val cvModel = cv.fit(trainingData)
val predictions = cvModel.transform(testData)
predictions.cache()
val results = predictions.withColumnRenamed("label", "fraude").withColumnRenamed("predicted_label", "prediction").groupBy("fraude", "prediction").agg(count("documento").alias("clientes")).orderBy("fraude", "prediction")
val table_name = ("results_random_forest").mkString
results.createOrReplaceTempView(table_name)
spark.sql("drop table if exists proceso_seguridad_externa." ++ table_name)
spark.sql("create table if not exists proceso_seguridad_externa." ++ table_name ++ " stored as parquet as select * from " ++ table_name)
}
def logistic_regression(features:Array[String], percentage:Double) : Unit = {
val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
val lr = new LogisticRegression().setFamily("binomial").setLabelCol("fraude").setFeaturesCol("features")
val Array(trainingData, testData) = base_modelo.randomSplit(Array(percentage, 1 - percentage))
val pipeline = new Pipeline().setStages(Array(assembler, lr))
trainingData.cache()
testData.cache()
val model = pipeline.fit(trainingData)
val predictions = model.transform(testData)
predictions.cache()
val results = predictions.withColumnRenamed("label", "fraude").withColumnRenamed("predicted_label", "prediction").groupBy("fraude", "prediction").agg(count("documento").alias("clientes")).orderBy("fraude", "prediction")
val table_name = ("results_logistic_regresion").mkString
results.createOrReplaceTempView(table_name)
spark.sql("drop table if exists proceso_seguridad_externa." ++ table_name)
spark.sql("create table if not exists proceso_seguridad_externa." ++ table_name ++ " stored as parquet as select * from " ++ table_name)
}
def naive_Bayes(features:Array[String], percentage:Double, apriori:String) : Unit = {
val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
val nb = new NaiveBayes().setLabelCol("fraude").setFeaturesCol("features").setModelType(apriori)
val Array(trainingData, testData) = base_modelo.randomSplit(Array(percentage, 1 - percentage))
val pipeline = new Pipeline().setStages(Array(assembler, nb))
trainingData.cache()
testData.cache()
val model = pipeline.fit(trainingData)
val predictions = model.transform(testData)
predictions.cache()
val results = predictions.withColumnRenamed("label", "fraude").withColumnRenamed("predicted_label", "prediction").groupBy("fraude", "prediction").agg(count("documento").alias("clientes")).orderBy("fraude", "prediction")
val table_name = ("results_naive_Bayes").mkString
results.createOrReplaceTempView(table_name)
spark.sql("drop table if exists proceso_seguridad_externa." ++ table_name)
spark.sql("create table if not exists proceso_seguridad_externa." ++ table_name ++ " stored as parquet as select * from " ++ table_name)
}
}
