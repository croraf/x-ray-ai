import "@tensorflow/tfjs-node-gpu";
import tf from "@tensorflow/tfjs";
import argparse from "argparse";

import data from "./data";
import { getModel } from "./model";
import { readCustomTestData } from "./image_manipulation";
import { resultAnalysis } from "./result_analysis";
import { modelTraining } from "./modelTraining";

async function run(
  epochs: number,
  batchSize: number,
  modelSavePath: String | null | undefined,
) {
  await data.loadData();

  const { images: trainImages, labels: trainLabels } = data.getTrainData();

  const model = getModel();
  /* const model = await tf.loadLayersModel(
    "file:///home/croraf/Desktop/Programiranje/open-source/tensorflow_AI/model/model.json",
  );
  console.log("Model loaded successfully!"); */

  const optimizer = "rmsprop";
  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  model.summary();

  await modelTraining(
    model,
    trainImages,
    trainLabels,
    batchSize,
    epochs,
    modelSavePath,
  );

  //const { images: testImages, labels: testLabels } = data.getTestData();
  const {
    images: testImages,
    labels: testLabels,
    labelsPlain,
  } = readCustomTestData();

  const evalOutput = model.evaluate(testImages, testLabels);
  console.log(
    `\nEvaluation result:\n` +
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; ` +
      `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`,
  );

  await resultAnalysis(model, testImages, labelsPlain);
}

const parser = new argparse.ArgumentParser({
  description: "TensorFlow.js-Node MNIST Example.",
  add_help: true,
});
parser.add_argument("--epochs", {
  type: "int",
  default: 4,
  help: "Number of epochs to train the model for.",
});
parser.add_argument("--batch_size", {
  type: "int",
  default: 128,
  help: "Batch size to be used during model training.",
});
parser.add_argument("--model_save_path", {
  type: "string",
  default: "model",
  help: "Path to which the model will be saved after training.",
});
const args = parser.parse_args();

run(args.epochs, args.batch_size, args.model_save_path);
