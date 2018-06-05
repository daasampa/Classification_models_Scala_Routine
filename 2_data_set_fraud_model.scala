/*Through the spark.sql method it's possible to call data stored in HDFS and give it the structure of Spark dataframe (org.apache.spark.sql.DataFrame).
The base_model, default name for this object, needs a sql sentece to retrieve the data.*/
val base_modelo = spark.sql("""select * from proceso_seguridad_externa.arl_base_fraude_mdp where documento is not null""")
/*The following command distribute the data available in base_modelo through all the slaves set.*/
base_modelo.cache()
