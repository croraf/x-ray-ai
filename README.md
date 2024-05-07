# About

Works with PNG 28x28 images

## GPU computations

If your system supports GPU-CUDA the GPU computations setup depending on your system is required (which might be challanging).
Check https://www.tensorflow.org/install/pip#linux for instructions

# Usage

## Requirements:

1. NodeJS with NPM
2. `npm install` to install dependencies

## Train

1. `npm run train`

Will save the model in `/model` folder

## Predict

1. `npm run predict`

This script loads the stored model from the `/model` folder.
It loads PNG images defined in `src/predict.ts` script and compares it with labels defined there.
Actual images must be present in `test_data` folder.
