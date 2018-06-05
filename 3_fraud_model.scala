object models{
def decision_tree(features:Array[String], percentage:Double, depth:Array[Int], folds:Int) : org.apache.spark.sql.DataFrame = {
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
cvModel.save("dt_fraud_model")
val predictions = cvModel.transform(testData)
predictions.cache()
val results = predictions.withColumn("label", when($"fraude"===1.0, "FRAUDE").otherwise("NO_FRAUDE")).withColumn("predicted_label", when($"prediction" === 1.0,"FRAUDE").otherwise("NO_FRAUDE")).groupBy("label", "predicted_label").agg(count("documento").alias("clientes")).orderBy("label", "predicted_label")
return results
}
def ada_boost(features:Array[String], percentage:Double, depth:Array[Int], n_trees:Array[Int], folds:Int):org.apache.spark.sql.DataFrame={
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
cvModel.save("gbt_fraud_model")
val predictions = cvModel.transform(testData)
predictions.cache()
val results = predictions.withColumn("label", when($"fraude"===1.0,"FRAUDE").otherwise("NO_FRAUDE")).withColumn("predicted_label", when($"prediction" === 1.0, "FRAUDE").otherwise("NO_FRAUDE")).groupBy("label", "predicted_label").agg(count("documento").alias("clientes")).orderBy("label", "predicted_label")
return results
}
def random_forest(features:Array[String], percentage:Double, depth:Array[Int], n_trees:Array[Int], folds:Int) : org.apache.spark.sql.DataFrame = {
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
cvModel.save("rf_fraud_model")
val predictions = cvModel.transform(testData)
predictions.cache()
val results = predictions.withColumn("label", when($"fraude" === 1.0, "FRAUDE").otherwise("NO_FRAUDE")).withColumn("predicted_label",when($"prediction"===1.0,"FRAUDE").otherwise("NO_FRAUDE")).groupBy("label", "predicted_label").agg(count("documento").alias("clientes")).orderBy("label", "predicted_label")
return results
}
def logistic_regression(features:Array[String], percentage:Double) : org.apache.spark.sql.DataFrame = {
val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
val lr = new LogisticRegression().setFamily("binomial").setLabelCol("fraude").setFeaturesCol("features")
val Array(trainingData, testData) = base_modelo.randomSplit(Array(percentage, 1 - percentage))
val pipeline = new Pipeline().setStages(Array(assembler, lr))
trainingData.cache()
testData.cache()
val model = pipeline.fit(trainingData)
model.save("lr_fraud_model")
val predictions = model.transform(testData)
predictions.cache()
val results = predictions.withColumn("label", when($"fraude" === 1.0, "FRAUDE").otherwise("NO_FRAUDE")).withColumn("predicted_label",when($"prediction"===1.0,"FRAUDE").otherwise("NO_FRAUDE")).groupBy("label", "predicted_label").agg(count("documento").alias("clientes")).orderBy("label", "predicted_label")
return results
}
def naive_Bayes(features:Array[String], percentage:Double, apriori:String) : org.apache.spark.sql.DataFrame = {
val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
val nb = new NaiveBayes().setLabelCol("fraude").setFeaturesCol("features").setModelType(apriori)
val Array(trainingData, testData) = base_modelo.randomSplit(Array(percentage, 1 - percentage))
val pipeline = new Pipeline().setStages(Array(assembler, nb))
trainingData.cache()
testData.cache()
val model = pipeline.fit(trainingData)
model.save("nb_fraud_model")
val predictions = model.transform(testData)
predictions.cache()
val results = predictions.withColumn("label", when($"fraude" === 1.0, "FRAUDE").otherwise("NO_FRAUDE")).withColumn("predicted_label",when($"prediction"===1.0,"FRAUDE").otherwise("NO_FRAUDE")).groupBy("label", "predicted_label").agg(count("documento").alias("clientes")).orderBy("label", "predicted_label")
return results
}
}

val data_gmm = spark.sql("""select documento,
							ifnull(svp_cnt_trx_no_monetarias_mes, 0) as c_1,
							ifnull(svp_cnt_trx_monetarias_mes, 0) as c_2,
							ifnull(svp_mnt_trx_monetarias_mes, 0) as c_3,
							ifnull(svp_max_cnt_trx_monetarias_mes, 0) as c_4,
							ifnull(svp_max_cnt_trx_no_monetarias_mes, 0) as c_5,
							ifnull(svp_mnt_trx_transfer_mes, 0) as c_6,
							ifnull(svp_cnt_trx_transfer_mes, 0) as c_7,
							ifnull(svp_max_cnt_trx_mes, 0) as c_8,
							ifnull(svp_mnt_trx_transfer_lineneg_mes, 0) as c_9,
							ifnull(svp_cnt_trx_transfer_lineneg_mes, 0) as c_10,
							ifnull(svp_max_cnt_trx_transfer_lineneg_mes, 0) as c_11,
							ifnull(svp_mnt_trx_pag_factu_mes, 0) as c_12,
							ifnull(svp_cnt_trx_pag_factu_mes, 0) as c_13,
							ifnull(svp_max_cnt_trx_pag_factu_mes, 0) as c_14,
							ifnull(svp_mnt_trx_audcred_mes, 0) as c_15,
							ifnull(svp_cnt_trx_audcred_mes, 0) as c_16,
							ifnull(svp_max_cnt_trx_audcred_mes, 0) as c_17,
							ifnull(svp_mnt_trx_e_prepag_mes, 0) as c_18,
							ifnull(svp_cnt_trx_e_prepag_mes, 0) as c_19,
							ifnull(svp_max_cnt_trx_e_prepag_mes, 0) as c_20,
							ifnull(svp_mnt_trx_pagtc_mes, 0) as c_21,
							ifnull(svp_cnt_trx_pagtc_mes, 0) as c_22,
							ifnull(svp_max_cnt_trx_pagtc_mes, 0) as c_23,
							ifnull(svp_cnt_trx_consulta_saldo_mes, 0) as c_24,
							ifnull(app_cnt_trx_no_monetarias_mes, 0) as c_25,
							ifnull(app_cnt_trx_monetarias_mes, 0) as c_26,
							ifnull(app_mnt_trx_monetarias_mes, 0) as c_27,
							ifnull(app_max_cnt_trx_monetarias_mes, 0) as c_28,
							ifnull(app_max_cnt_trx_no_monetarias_mes, 0) as c_29,
							ifnull(app_mnt_trx_transfer_mes, 0) as c_30,
							ifnull(app_cnt_trx_transfer_mes, 0) as c_31,
							ifnull(app_max_cnt_trx_mes, 0) as c_32,
							ifnull(app_mnt_trx_transfer_lineneg_mes, 0) as c_33,
							ifnull(app_cnt_trx_transfer_lineneg_mes, 0) as c_34,
							ifnull(app_max_cnt_trx_transfer_lineneg_mes, 0) as c_35,
							ifnull(app_mnt_trx_pag_factu_mes, 0) as c_36,
							ifnull(app_cnt_trx_pag_factu_mes, 0) as c_37,
							ifnull(app_max_cnt_trx_pag_factu_mes, 0) as c_38,
							ifnull(app_mnt_trx_pagtc_mes, 0) as c_39,
							ifnull(app_cnt_trx_pagtc_mes, 0) as c_40,
							ifnull(app_max_cnt_trx_pagtc_mes, 0) as c_41,
							ifnull(app_cnt_trx_consulta_saldo_mes, 0) as c_42
							from resultados_seguridad_externa.dsc_mdd_transaccional
							where f_act = 20180515 and documento is not null
							order by documento asc""")

val data_gmm = spark.sql("select * from proceso_seguridad_externa.scoring_variables_preferenciales order by documento")
data_gmm.cache()

val complete_gmm = data_gmm.drop("documento")

def select_vars(variable:String) : Double = {
return (scaled_data.filter(scaled_data(variable) === 0).count().toDouble / complete_gmm.count().toDouble)*100
}

val names = complete_gmm.drop("documento_virtuales","indicador").columns
val percentage_zeros = complete_gmm.drop("documento_virtuales","indicador").columns.map(x => select_vars(x))

val results = sc.parallelize(names zip percentage_zeros).toDF("names", "percentage_zeros")

results.orderBy(desc("percentage_zeros")).show()

results.filter(results("percentage_zeros") >= 50.0).select("names").orderBy(asc("percentage_zeros")).show()

val features = complete_gmm.drop("documento_virtuales","indicador","t1_visa","t5_visa","t4_visa","t2_visa","t1_master","t5_master","t4_master","t3_visa","t2_master","t3_master","c_12","c_14","c_13","t6_visa","t6_master","t1_amex","c_9","c_11","c_10","t2_amex").columns

val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")


val assembled_data = assembler.transform(complete_gmm.drop("t1_visa","t5_visa","t4_visa","t2_visa","t1_master","t5_master","t4_master","t3_visa","t2_master","t3_master","c_12","c_14","c_13","t6_visa","t6_master","t1_amex","c_9","c_11","c_10","t2_amex"))
val scaler_object = scaler.fit(assembled_data)

val scaled_data = scaler_object.transform(assembled_data)


val gmm = new GaussianMixture().setK(10).setFeaturesCol("scaledFeatures")

val trainingData = scaled_data.filter(scaled_data("indicador") === "estimación")
trainingData.cache()
val testData = scaled_data.filter(scaled_data("indicador") === "pronóstico")
testData.cache()

val model = gmm.fit(trainingData)
val trainingData_model = model.transform(trainingData)
trainingData_model.cache()
trainingData_model.groupBy("prediction").agg(count("documento_virtuales").alias("clientes")).orderBy(desc("clientes")).show()
val testData_model = model.transform(testData)
testData_model.cache()
testData_model.groupBy("prediction").agg(count("documento_virtuales").alias("clientes")).orderBy(desc("clientes")).show()