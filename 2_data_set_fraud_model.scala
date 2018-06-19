/*Through the spark.sql method it's possible to call data stored in HDFS and give it the structure of Spark dataframe (org.apache.spark.sql.DataFrame).
The base_model, default name for this object, needs a sql sentece to retrieve the data.*/
val base_modelo = spark.sql("""select documento, segmento,
	                                    cambio_otp as c1, cambio_topes as c2, ip_riesgosa as c3, enumeracion as c4,
	                                    actualizacion_datos as c5, edad_cliente as c6, evidente_riesgoso as c7,
	                                    regeneracion_clave as c8, fraude
	                                    from proceso_seguridad_externa.arl_matriz_variables
	                                    where segmento = 'plus'""")
/*The following command distribute the data available in base_modelo through all the slaves set.*/
base_modelo.cache()
