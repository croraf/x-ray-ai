import "@tensorflow/tfjs-node-gpu";
import argparse from "argparse";
import data from "./utils/data";
import { getModel } from "./utils/model";
import { modelTraining } from "./utils/modelTraining";

async function run(
  epochs: number,
  batchSize: number,
  modelSavePath: String | null | undefined,
) {
  await data.loadData();

  ////////////////////////////////////////////
  const { images: trainImages, labels: trainLabels } = data.getTrainData();
  ////////////////////////////////////////////

  const model = getModel();

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
  ////////////////////////////////////////////

  const { images: testImages, labels: testLabels } = data.getTestData();

  const evalOutput = model.evaluate(testImages, testLabels);
  console.log(
    `\nEvaluation result:\n` +
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; ` +
      `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`,
  );
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
