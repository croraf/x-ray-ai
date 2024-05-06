import fs from "fs";
import tf from "@tensorflow/tfjs";
import { PNG } from "pngjs";

export const readCustomTestData = () => {
  const data = fs.readFileSync("test3.png");
  const png = PNG.sync.read(data);
  console.log(png.height, png.width, png.data.length);

  const imagesShape: [number, number, number, number] = [
    1, // number of images
    28,
    28,
    1,
  ];
  const images = new Float32Array(tf.util.sizeFromShape(imagesShape));
  const labels = new Int32Array(tf.util.sizeFromShape([1 /* size */, 1]));

  for (let i = 0; i < png.height; i++) {
    for (let j = 0; j < png.width * 4; j = j + 4) {
      const idx = Math.round(
        (png.data[png.width * 4 * i + j] +
          png.data[png.width * 4 * i + j + 1] +
          png.data[png.width * 4 * i + j + 2]) /
          3,
      );
      images.set([idx], j / 4 + png.width * i);
      process.stdout.write(idx > 150 ? "█" : " ");
    }
    console.log();
  }

  console.log("rafa", images.length);

  for (let i = 0; i < png.height; i++) {
    for (let j = 0; j < png.width; j++) {
      const idx = images[png.width * i + j];
      process.stdout.write(idx > 150 ? "█" : " ");
    }
    console.log();
  }

  labels.set([3]);

  return {
    images: tf.tensor4d(images, imagesShape),
    labels: tf.oneHot(tf.tensor1d(labels, "int32"), 10).toFloat(),
  };
};
