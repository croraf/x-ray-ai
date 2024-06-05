import tf from "@tensorflow/tfjs";

export const resultAnalysis = async (
  model: tf.LayersModel,
  data: tf.Tensor4D,
  labelsPlain: number[],
) => {
  const predictions = model.predict(data) as tf.Tensor<tf.Rank.R2>;
  console.log("\nPredictions tensor:");
  predictions.print();

  const classProbabilities = predictions.dataSync();

  for (let i = 0; i < classProbabilities.length; i++) {
    console.log(`Probability of class ${i}: ${classProbabilities[i]}`);
  }

  // summary
  /* const predictionsInt: number[] = [];
  for (let i = 0; i < predictions.shape[0]; i++) {
    for (let j = 0; j < predictions.shape[1]; j++) {
      if (classProbabilities[i * predictions.shape[1] + j] === 1) {
        predictionsInt.push(j);
      }
    }
  }
  console.log(
    "\nPredictions:\n",
    predictionsInt.map((item) => item.toString()),
  );

  const expectedDiff = predictionsInt.map((prediction, index) =>
    prediction - labelsPlain[index] === 0 ? " " : labelsPlain[index].toString(),
  );
  console.log("Actual value:\n", expectedDiff); */
};
