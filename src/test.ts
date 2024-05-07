import "@tensorflow/tfjs-node-gpu";
import tf from "@tensorflow/tfjs";
import data from "./utils/data";

async function testAccuracy() {
  await data.loadData();

  const model = await tf.loadLayersModel("file://model/model.json");
  console.log("\x1b[36m" + "\nModel loaded successfully!\n" + "\x1b[39m");

  const optimizer = "rmsprop";
  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  model.summary();
  ////////////////////////////////////////////

  const { images: testImages, labels: testLabels } = data.getTestData();

  const evalOutput = model.evaluate(testImages, testLabels);
  console.log(
    `\nEvaluation result:\n` +
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; ` +
      `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`,
  );
}

testAccuracy();
