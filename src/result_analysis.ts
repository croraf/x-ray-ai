import tf from "@tensorflow/tfjs";

export const resultAnalysis = async (
  model: tf.LayersModel,
  data: tf.Tensor4D,
  labelsPlain: number[],
) => {
  const predictions = model.predict(data) as tf.Tensor<tf.Rank>;

  const predictionsInt: number[] = [];
  for (let i = 0; i < predictions.shape[0]; i++) {
    for (let j = 0; j < predictions.shape[1]; j++) {
      if (predictions.dataSync()[i * predictions.shape[1] + j] === 1) {
        predictionsInt.push(j);
      }
    }
  }
  console.log(predictionsInt.map((item) => item.toString()));

  const wrongEvaluations = predictionsInt.map((prediction, index) =>
    prediction - labelsPlain[index] === 0 ? " " : labelsPlain[index].toString(),
  );
  console.log(wrongEvaluations);

  /* for (let i = 0; i < labels.length; i++) {
    const predictedLabel = Math.round(predictions[i][0]); // Assuming single-class prediction
    const trueLabel = labels[i];

    if (predictedLabel !== trueLabel) {
      wrongEvaluations.push({
        index: i,
        prediction: predictedLabel,
        truth: trueLabel,
      });
    }
  } */

  // Analyze the wrongEvaluations array to identify patterns in errors
  //console.log(wrongEvaluations);
};
