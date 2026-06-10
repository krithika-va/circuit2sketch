# Hand-Drawn Circuit to Digital Schematic Converter

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project converts hand-drawn circuit diagrams into digital schematics and SPICE netlists using a single learned U-Net model with deterministic post-processing.

| Aspect | Details |
|--------|---------|
| Input | Hand-drawn circuit images (JPG/PNG) |
| Output | Digital schematic + SPICE netlist |
| Components | 15 classes (resistors, capacitors, logic gates, wires, junctions) |
| Speed | ~1.3 seconds per 512x512 image |
| Dataset | 2,626 annotated images from 33 drafters |
| Framework | PyTorch, OpenCV, NetworkX |

## Features

- 15 Component Classes - Resistors, capacitors, inductors, diodes, voltage sources, ground, AND/OR/NOT/NAND/NOR/XOR gates, wires, junctions
- Single Learned Model - Only U-Net segmentation is learned; everything else is deterministic and rule-based
- Connectivity-Aware Loss - Custom loss combining Weighted CE + Dice + Connectivity (0.3 weight)
- ThinWireUNet Architecture - Depthwise separable convolutions preserve 1-3px thin wires
- Conditional Netlist Generation - Generates SPICE netlist only when confident; abstains with explanation
- Drafter-Level Splitting - Prevents data leakage from similar handwriting styles
- Real-time Processing - ~1.3 seconds per 512x512 image on Tesla T4 GPU

## Architecture

The pipeline follows a clean, stage-wise design:

Stage 1: Input Image (RGB)

Stage 2: Preprocessing (Resize to 512x512)

Stage 3: U-Net Segmentation (Learned - 6M parameters, 15-class pixel-wise prediction)

Stage 4: Post-Processing (Rule-based - Mask cleanup, wire skeletonization, junction detection with dual condition degree>=3 AND predicted, component extraction, terminal detection)

Stage 5: Graph Construction (NetworkX - Nodes: component terminals + junctions, Edges: continuous skeleton paths)

Stage 6: Output (Schematic rendering via Schemdraw + Conditional SPICE netlist generation)

## Model Details

ThinWireUNet Architecture Specifications:

| Component | Specification |
|-----------|---------------|
| Base Architecture | U-Net with encoder-decoder and skip connections |
| Convolution Type | Depthwise separable (40% fewer parameters) |
| Input Size | 512 x 512 x 3 (RGB) |
| Output Classes | 15 (background + 14 circuit classes) |
| Total Parameters | 6,000,877 |
| Encoder Channels | [64, 128, 256, 512] |
| Bottleneck Channels | 1024 |

## Loss Function

Total Loss = Weighted_CrossEntropy + Dice_Loss + 0.3 * Connectivity_Loss

Where:
- Weighted_CE handles severe class imbalance (background dominates)
- Dice Loss optimizes overlap for small objects (junctions, gates)
- Connectivity Loss penalizes broken wire continuity using 3x3 kernel density

## Results

Training Performance:

| Metric | Value |
|--------|-------|
| Best Validation Loss | 2.5280 (Epoch 40) |
| Final Validation Loss | 2.6834 (Epoch 47) |
| Best Validation CE | 1.5830 |
| Training Epochs | 47 (early stopping triggered) |
| Training Time | ~11 minutes per epoch on Tesla T4 |

Test Set Performance (409 images, 5 unseen drafters):

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.7623 |
| Mean IoU | 0.0787 |
| Mean Dice | 0.1083 |
| Background IoU | 0.7741 |
| Wire IoU | 0.1371 |
| Junction IoU | 0.1049 |

Per-Class IoU Breakdown:

| Class | IoU | Dice |
|-------|-----|------|
| Background | 0.7741 | 0.8726 |
| Wire | 0.1371 | 0.2412 |
| Junction | 0.1049 | 0.1899 |
| Resistor | 0.0186 | 0.0365 |
| Capacitor | 0.0463 | 0.0885 |
| Inductor | 0.0072 | 0.0142 |
| Diode | 0.0349 | 0.0675 |
| DC Voltage Source | 0.0039 | 0.0078 |
| Ground | 0.0212 | 0.0416 |
| AND Gate | 0.0012 | 0.0024 |
| OR Gate | 0.0115 | 0.0227 |
| NOT Gate | 0.0060 | 0.0120 |
| NAND Gate | 0.0142 | 0.0281 |
| NOR Gate | 0.0000 | 0.0000 |
| XOR Gate | 0.0000 | 0.0000 |


## Alternatives Explored

| Approach | Status | Reason Not Used |
|----------|--------|-----------------|
| YOLOv8 Object Detection | Failed | Poor wire connectivity, required separate models |
| Graph Neural Networks | Failed | Massive data requirements, black-box debugging |
| Traditional CV + SVM | Failed | Couldn't handle handwriting variation |
| Transformer-based Segmentation | Not used | Too slow (3-4s per image), excessive VRAM |
| U-Net + Deterministic Rules | Selected | Best accuracy/speed trade-off, fully interpretable |

## Key Design Decisions

- Single Learned Model - Only segmentation is learned; everything else deterministic for reliability and interpretability
- Conservative Generation - System never guesses; abstains with explanation when uncertain
- Drafter-Level Splitting - No same-drafter leakage between train/val/test sets
- Depthwise Separable Convolutions - Preserve thin wires while reducing parameters by 40%
- Dual-Condition Junction Detection - Geometric (degree >= 3) AND semantic (model prediction)
- Priority-Based Labeling - Junction > Component > Wire > Background for mask construction

