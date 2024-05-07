import "@tensorflow/tfjs-node-gpu";
import tf from "@tensorflow/tfjs";
import { readCustomTestData } from "./utils/image_manipulation";
import { resultAnalysis } from "./utils/result_analysis";

const predict = async () => {
  ////////////////////////////////////////////
  const model = await tf.loadLayersModel(
    "file://model/model.json",
  );
  
  console.log("\x1b[36m" + "\nModel loaded successfully!\n" + "\x1b[39m");

  const optimizer = "rmsprop";
  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  ////////////////////////////////////////////

  const {
    images: testImages,
    labels: testLabels,
    labelsPlain,
  } = readCustomTestData();

  await resultAnalysis(model, testImages, labelsPlain);
};

predict();
