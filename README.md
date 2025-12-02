# DeepFlood: Water Depth Estimation with PINNs & GANs

This repository implements a pipeline for simulating flood events using the **ANUGA** Shallow Water Equation solver and training a **Generative Adversarial Network (GAN)** to predict future water depth and momentum based on current states. The project combines physics-based data generation with deep learning for fast flood forecasting.

## 📂 Project Structure

```text
.
├── configs/
│   └── demo_austin.yaml       # Configuration for ANUGA simulations
├── data_gen/
│   ├── anuga_simulator.py     # Wrapper for ANUGA physics engine
│   ├── rainfall_scenarios.py  # Rainfall hyetograph generation
│   └── sww_to_grid.py         # Interpolation logic (SWW mesh -> Regular Grid)
├── dataset/
│   ├── gan_dataset.py         # PyTorch Dataset for loading NPZ data
│   └── deepflood_anuga_dataset.npz # (Generated) Compiled dataset
├── models/
│   └── gan_models.py          # UNet Generator and PatchGAN Discriminator
├── scripts/
│   ├── run_generate_sims.py   # Batch runner for simulations
│   ├── run_build_dataset.py   # Compiles simulation outputs into a dataset
│   ├── gan_train.py           # Main training script
│   ├── preprocess_dem.py      # Utilities to fill/clean DEM files
│   └── visualize_*.py         # Various visualization tools
└── README.md
````

## 🚀 Environment & Installation

This project relies on `anuga` (for physics simulation), `rasterio`, `netCDF4`, and `pytorch`.


*Note: Ensure your GPU devices are correctly mapped in the script.*

## 🛠️ Pipeline Workflow

### 1\. Data Generation (Physics Simulation)

Generate raw flood data using the ANUGA solver. This script reads a DEM file and runs multiple simulations with varying rainfall parameters defined in `configs/demo_austin.yaml`.

```bash
python -m scripts.run_generate_sims configs/demo_austin.yaml
```

  * **Input:** DEM file (e.g., `DEM/austin_filled.asc`) and config parameters.
  * **Output:** `.sww` (NetCDF) simulation files in `sims_austin/`.

### 2\. Dataset Construction

Convert the unstructured mesh data (`.sww` files) into regular grid pairs for the neural network.

```bash
python -m scripts.run_build_dataset
```

  * **Process:** Interpolates depth and momentum (Qx, Qy) from the mesh to a 100x100 grid.
  * **Pairs:** Creates pairs of $(X_t, Y_{t+\Delta})$.
      * **Input ($X_t$):** 5 channels `[Depth, Qx, Qy, Inflow, Rain]` at time $t$.
      * **Target ($Y_{t+\Delta}$):** 3 channels `[Depth, Qx, Qy]` at time $t + 30\text{min}$.
  * **Output:** `dataset/deepflood_anuga_dataset.npz`.

### 3\. Training the GAN

Train the Pix2Pix-style model (UNet Generator + PatchGAN Discriminator).

```bash
python -m scripts.gan_train
```

  * **Configuration:** Adjust batch size, learning rate, and epochs inside `gan_train.py`.
  * **Output:** Checkpoints saved to `gan_checkpoints/`.

## 📊 Visualization & Analysis

### Visualize GAN Results

Compare Ground Truth vs. GAN Predictions on the validation set.

```bash
python -m scripts.visualize_gan_results
```

  * **Output:** Generates side-by-side comparison images in `gan_vis/`.

### Visualize Raw SWW Data

Inspect specific time steps from the raw ANUGA output.

```bash
python -m scripts.visualize_sww_depth \
    --sww sims_austin/sim_0000.sww \
    --nx 256 --ny 256 \
    --time-sec 3600 \
    --out depth_images_sww/sim_0000_t3600.png
```

## 🧠 Model Architecture

The core model is defined in `models/gan_models.py`:

  * **Generator (UNet):**
      * **Encoder:** 5 layers of Convolution -\> BatchNorm -\> LeakyReLU. Compresses 100x100 input to 3x3 bottleneck.
      * **Decoder:** 5 layers of Transposed Convolution with skip connections from the encoder.
      * **Input:** 5 Channels (Depth, Qx, Qy, Inflow, Rain).
      * **Output:** 3 Channels (Predicted Depth, Qx, Qy).
  * **Discriminator (PatchGAN):**
      * Classifies $N \times N$ patches of the image as real or fake.
      * Takes 8 channels (Input + Target/Prediction) concatenated.

## ⚙️ Configuration

The simulation is controlled via `configs/demo_austin.yaml`:

  * `dem_path`: Path to the elevation raster.
  * `yieldstep`: Time interval (seconds) between data snapshots.
  * `max_triangle_area`: Controls mesh resolution (lower = finer).
  * `rainfall_peak_min/max`: Range for random rainfall intensity generation.

<!-- end list -->

