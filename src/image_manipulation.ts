import fs from "fs";
import tf from "@tensorflow/tfjs";
import { PNG } from "pngjs";

const imageResolution = 28 * 28;

const normalizeData = (png, images, index) => {
  const offset = index * imageResolution;

  for (let i = 0; i < png.height; i++) {
    for (let j = 0; j < png.width * 4; j = j + 4) {
      const idx = Math.round(
        (png.data[png.width * 4 * i + j] +
          png.data[png.width * 4 * i + j + 1] +
          png.data[png.width * 4 * i + j + 2]) /
          3,
      );
      images.set([idx], j / 4 + png.width * i + offset);
      process.stdout.write(idx > 150 ? "█" : " ");
    }
    console.log();
  }

  for (let i = 0; i < png.height; i++) {
    for (let j = 0; j < png.width; j++) {
      const idx = images[png.width * i + j + offset];
      process.stdout.write(idx > 150 ? "█" : " ");
    }
    console.log();
  }
};

export const readCustomTestData = () => {
  const labelsData = [3, 8, 8];
  const fileNames = ["3.png", "8.png", "8_2.png"];
  const pngs = fileNames.map((fileName) =>
    PNG.sync.read(fs.readFileSync(fileName)),
  );

  const imagesShape: [number, number, number, number] = [
    fileNames.length,
    28,
    28,
    1,
  ];
  const images = new Float32Array(tf.util.sizeFromShape(imagesShape));
  const labels = new Int32Array(tf.util.sizeFromShape([fileNames.length, 1]));

  pngs.forEach((png, index) => normalizeData(png, images, index));

  labels.set(labelsData);

  return {
    images: tf.tensor4d(images, imagesShape),
    labels: tf.oneHot(tf.tensor1d(labels, "int32"), 10).toFloat(),
  };
};
