import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
sesh=pyspark.sql.SparkSession.builder.appName('v1').getOrCreate()
sesh
df=sesh.read.option('header','true').csv('Crop_recommendation.csv',inferSchema=True)
df.printSchema()
df.describe().show()
df_cleaned=df.na.drop()
df_cleaned.describe().show()
columns = df.columns
feature_columns = columns[:-1]

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

df_assembled = assembler.transform(df)
df_assembled.show()
# StringIndexer
indexer = StringIndexer(inputCol="label", outputCol="label_indexed")
df_indexed = indexer.fit(df_assembled).transform(df_assembled)

# OneHotEncoder
# encoder = OneHotEncoder(inputCols=["label_indexed"], outputCols=["label_encoded"])
# df_encoded = encoder.fit(df_indexed).transform(df_indexed)

df_indexed.show()
df_final = df_indexed.select("features", "label_indexed")
df_final.show()
train_ratio = 0.7
test_ratio = 0.3

train_df, test_df = df_final.randomSplit([train_ratio, test_ratio], seed=42)
from pyspark.ml.classification import RandomForestClassifier

# Create a RandomForestClassifier object
rf = RandomForestClassifier(featuresCol="features", labelCol="label_indexed")

# Train the model
rf_model = rf.fit(train_df)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

predictions = rf_model.transform(test_df)


evaluator = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedPrecision")
precision = evaluator.evaluate(predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="weightedRecall")
recall = evaluator.evaluate(predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

