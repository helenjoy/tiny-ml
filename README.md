# TinyML Motion Detection

A step-by-step TinyML (Edge AI) example that uses smartphone accelerometer data to classify **idle** versus **motion**. The project demonstrates the complete workflow from collecting sensor data to preparing a quantized model for deployment on resource-constrained embedded devices.

## What this project demonstrates

The notebook [`motion_detection_tiny_ml_.ipynb`](motion_detection_tiny_ml_.ipynb) covers:

1. Collecting real-world accelerometer data with a smartphone.
2. Uploading and cleaning a CSV sensor recording in Google Colab.
3. Standardizing the three acceleration channels:
   - `accel_x`
   - `accel_y`
   - `accel_z`
4. Assigning `Idle` and `Motion` labels to the recording.
5. Visualizing the acceleration signals with Matplotlib.
6. Extracting statistical features from overlapping time windows.
7. Training a small TensorFlow/Keras neural network.
8. Converting the model to fully quantized INT8 TensorFlow Lite format.
9. Running local INT8 inference with a TensorFlow Lite interpreter.
10. Exporting the model as a C header for use with TensorFlow Lite for Microcontrollers.

## Model and data pipeline

Each 50-sample window advances by 25 samples. Six features are extracted from every window:

- Mean and standard deviation of `accel_x`
- Mean and standard deviation of `accel_y`
- Mean and standard deviation of `accel_z`

The classifier is a compact multilayer perceptron with:

- 6 input features
- Dense layer with 8 ReLU neurons
- Dense layer with 4 ReLU neurons
- 2-class softmax output (`Idle` or `Motion`)

The example notebook trains for 35 epochs with a batch size of 8. Its recorded run achieved approximately **97.04% training accuracy** and produced an INT8 TensorFlow Lite model of approximately **2.8 KB**.

> The accuracy and model size depend on the sensor recording and environment. The notebook does not include a separate validation or test split, so the displayed accuracy should not be treated as a generalization benchmark.

## Getting started

### Option 1: Run in Google Colab

Open the notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helenjoy/tiny-ml/blob/main/motion_detection_tiny_ml_.ipynb)

Then:

1. Record accelerometer data with a smartphone sensor app such as [Phyphox](https://phyphox.org/).
2. Record two phases: an idle/resting phase followed by a motion phase.
3. Export the recording as a CSV file.
4. Run the notebook and upload the CSV when prompted.
5. If necessary, update the column names in the notebook to match your sensor app's export format.
6. Execute the cells from top to bottom.

### Option 2: Run locally

The notebook is written for Python 3 and uses the following main packages:

```bash
pip install numpy pandas matplotlib tensorflow jupyter
```

Start Jupyter and open the notebook:

```bash
jupyter notebook motion_detection_tiny_ml_.ipynb
```

The file-upload cell uses `google.colab.files`, so local execution may require replacing that cell with a normal local file path, for example:

```python
import pandas as pd

df_phone = pd.read_csv("Raw Data.csv")
```

## Expected CSV columns

The notebook currently expects these source columns:

```text
Linear Acceleration x (m/s^2)
Linear Acceleration y (m/s^2)
Linear Acceleration z (m/s^2)
```

They are renamed to `accel_x`, `accel_y`, and `accel_z`. If your application exports different headers, update the `column_mapping` dictionary in the notebook.

## Generated artifacts

Running the notebook generates files in the execution environment, including:

- `phone_motion_model.tflite` — fully INT8-quantized TensorFlow Lite model
- `phone_motion_model.h` — C byte-array representation intended for embedded firmware

The C header is generated with `xxd` and can be included in a TensorFlow Lite for Microcontrollers project after adding the appropriate runtime, model arena, and inference code.

## Repository contents

```text
.
├── README.md
└── motion_detection_tiny_ml_.ipynb
```

## Limitations and next steps

- Labels are created by splitting the recording at its midpoint; for a real application, label recordings explicitly.
- The example trains on the same data used to report accuracy; add train/validation/test splits for meaningful evaluation.
- Sensor orientation, sampling rate, device differences, and noise can affect predictions.
- The generated model still needs to be integrated with a target microcontroller and tested on hardware.
- The notebook currently verifies inference locally rather than continuously classifying live sensor data.

## License

No license is currently specified for this repository. Add a license file before distributing or reusing the project.
