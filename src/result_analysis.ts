import tf from "@tensorflow/tfjs";

export const resultAnalysis = async (
  model: tf.LayersModel,
  data: tf.Tensor4D,
  labelsPlain: number[],
) => {
  const predictions = model.predict(data) as tf.Tensor<tf.Rank.R2>;
  predictions.print();

  const predictionsData = predictions.dataSync();

  // summary
  const predictionsInt: number[] = [];
  for (let i = 0; i < predictions.shape[0]; i++) {
    for (let j = 0; j < predictions.shape[1]; j++) {
      if (predictionsData[i * predictions.shape[1] + j] === 1) {
        predictionsInt.push(j);
      }
    }
    console.log();
  }
  console.log(predictionsInt.map((item) => item.toString()));

  const wrongEvaluations = predictionsInt.map((prediction, index) =>
    prediction - labelsPlain[index] === 0 ? " " : labelsPlain[index].toString(),
  );
  console.log(wrongEvaluations);
};
