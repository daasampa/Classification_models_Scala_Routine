val base_modelo = spark.sql("""select * from proceso_seguridad_externa.arl_base_fraude_mdp where documento is not null""")
base_modelo.cache()