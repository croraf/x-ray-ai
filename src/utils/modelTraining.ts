import tf from "@tensorflow/tfjs";

export const modelTraining = async (
  model: tf.LayersModel,
  trainImages: tf.Tensor4D,
  trainLabels: tf.Tensor<tf.Rank>,
  batchSize: number,
  epochs: number,
  modelSavePath: String | null | undefined,
) => {
  let epochBeginTime;
  let millisPerStep;
  const validationSplit = 0.15;
  const numTrainExamplesPerEpoch = trainImages.shape[0] * (1 - validationSplit);
  const numTrainBatchesPerEpoch = Math.ceil(
    numTrainExamplesPerEpoch / batchSize,
  );

  const dateTrainingStart = Date.now();
  await model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit,
  });

  if (modelSavePath != null) {
    await model.save(`file://${modelSavePath}`);
    console.log(`Saved model to path: ${modelSavePath}`);
  }

  console.log(
    "Total training duration [min]: ",
    (Date.now() - dateTrainingStart) / 1000 / 60,
  );
};
