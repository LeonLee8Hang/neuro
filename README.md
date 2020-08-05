# neuro
# Neuromorphic Cognitive Computing

My name is Li Hang.

## Table of Contents
 - [Conference Papers](#conference-papers)
   - 2017: [ISSCC](#2017-isscc), [ISCA](#2017-isca), [MICRO](#2017-micro), [HPCA](#2017-hpca), [ASPLOS](#2017-asplos), [DAC](#2017-dac), [FPGA](#2017-fpga), [ICCAD](#2017-iccad), [DATE](#2017-date), [VLSI](#2017-vlsi), [FCCM](#2017-fccm), [HotChips](#2017-hotchips)

 - [Important Topics](#important-topics)
   - [Tutorial and Survey](#tutorial-and-survey)
   - [Benchmarks](#benchmarks)
   - [Network Compression](#network-compression)
   - [Other Topics](#other-topics)

## Conference Papers
This is a collection of conference papers that interest me. The emphasis is focused on, but not limited to neural networks on silicon. Papers of significance are marked in **bold**. My comments are marked in *italic*.

### 2017 FPGA
- **An OpenCL Deep Learning Accelerator on Arria 10.** (Intel)
  - *Minimum bandwidth requirement: All the intermediate data in AlexNet's CONV layers are cached in the on-chip buffer, so their architecture is compute-bound.*
  - *Reduced operations: Winograd transformation.*
  - *High usage of the available DSPs+Reduced computation -> Higher performance on FPGA -> Competitive efficiency vs. TitanX.*
- **ESE: Efficient Speech Recognition Engine for Compressed LSTM on FPGA.** (Stanford University, DeepPhi, Tsinghua University, NVIDIA)
- **FINN: A Framework for Fast, Scalable Binarized Neural Network Inference.** (Xilinx, Norwegian University of Science and Technology, University of Sydney)
- **Can FPGA Beat GPUs in Accelerating Next-Generation Deep Neural Networks?** (Intel)
- **Accelerating Binarized Convolutional Neural Networks with Software-Programmable FPGAs.** (Cornell University, UCLA, UCSD)
- **Improving the Performance of OpenCL-based FPGA Accelerator for Convolutional Neural Network.** (UW-Madison)
- **Frequency Domain Acceleration of Convolutional Neural Networks on CPU-FPGA Shared Memory System.** (USC)
- **Optimizing Loop Operation and Dataflow in FPGA Acceleration of Deep Convolutional Neural Networks.** (Arizona State University)

### 2017 ISSCC
- **A 2.9TOPS/W Deep Convolutional Neural Network SoC in FD-SOI 28nm for Intelligent Embedded Systems.** (ST)
- **DNPU: An 8.1TOPS/W Reconfigurable CNN-RNN Processor for General Purpose Deep Neural Networks.** (KAIST)
- **ENVISION: A 0.26-to-10TOPS/W Subword-Parallel Computational Accuracy-Voltage-Frequency-Scalable Convolutional Neural Network Processor in 28nm FDSOI.** (KU Leuven)
- **A 288µW Programmable Deep-Learning Processor with 270KB On-Chip Weight Storage Using Non-Uniform Memory Hierarchy for Mobile Intelligence.** (University of Michigan, CubeWorks)
- A 28nm SoC with a 1.2GHz 568nJ/Prediction Sparse Deep-NeuralNetwork Engine with >0.1 Timing Error Rate Tolerance for IoT Applications. (Harvard)
- A Scalable Speech Recognizer with Deep-Neural-Network Acoustic Models and Voice-Activated Power Gating (MIT)
- A 0.62mW Ultra-Low-Power Convolutional-Neural-Network Face Recognition Processor and a CIS Integrated with Always-On Haar-Like Face Detector. (KAIST)

### 2017 HPCA
- **FlexFlow: A Flexible Dataflow Accelerator Architecture for Convolutional Neural Networks.** (Chinese Academy of Sciences)
- **PipeLayer: A Pipelined ReRAM-Based Accelerator for Deep Learning.** (University of Pittsburgh, University of Southern California)
- Towards Pervasive and User Satisfactory CNN across GPU Microarchitectures. (University of Florida)
  - *Satisfaction of CNN (SoC) is the combination of SoCtime, SoCaccuracy and energy consumption.*
  - *The P-CNN framework is composed of offline compilation and run-time management.*
    - *Offline compilation: Generally optimizes runtime, and generates scheduling configurations for the run-time stage.*
    - *Run-time management: Generates tuning tables through accuracy tuning, and calibrate accuracy+runtime (select the best tuning table) during the long-term execution.*
- Supporting Address Translation for Accelerator-Centric Architectures. (UCLA)

### 2017 ASPLOS
- **Tetris: Scalable and Efficient Neural Network Acceleration with 3D Memory.** (Stanford University)
  - *Move accumulation operations close to the DRAM banks.*
  - *Develop a hybrid partitioning scheme that parallelizes the NN computations over multiple accelerators.*
- SC-DCNN: Highly-Scalable Deep Convolutional Neural Network using Stochastic Computing. (Syracuse University, USC, The City College of New York)

### 2017 ISCA
- **Maximizing CNN Accelerator Efficiency Through Resource Partitioning.** (Stony Brook University)
  - *An Extension of their FPL'16 paper.*
- **In-Datacenter Performance Analysis of a Tensor Processing Unit.** (Google)
- **SCALEDEEP: A Scalable Compute Architecture for Learning and Evaluating Deep Networks.** (Purdue University, Intel)
  - *Propose a full-system (server node) architecture, focusing on the challenge of DNN training (intra and inter-layer heterogeneity).*
- **SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks.** (NVIDIA, MIT, UC Berkeley, Stanford University)
- **Scalpel: Customizing DNN Pruning to the Underlying Hardware Parallelism.** (University of Michigan, ARM)
- Understanding and Optimizing Asynchronous Low-Precision Stochastic Gradient Descent. (Stanford)
- LogCA: A High-Level Performance Model for Hardware Accelerators. (AMD, University of Wisconsin-Madison)
- APPROX-NoC: A Data Approximation Framework for Network-On-Chip Architectures. (TAMU)

### 2017 FCCM
- **Escher: A CNN Accelerator with Flexible Buffering to Minimize Off-Chip Transfer.** (Stony Brook University)
- **Customizing Neural Networks for Efficient FPGA Implementation.**
- **Evaluating Fast Algorithms for Convolutional Neural Networks on FPGAs.**
- **FP-DNN: An Automated Framework for Mapping Deep Neural Networks onto FPGAs with RTL-HLS Hybrid Templates.** (Peking University, HKUST, MSRA, UCLA)
  - *Compute-instensive part: RTL-based generalized matrix multiplication kernel.*
  - *Layer-specific part: HLS-based control logic.*
  - *Memory-instensive part: Several techniques for lower DRAM bandwidth requirements.*
- FPGA accelerated Dense Linear Machine Learning: A Precision-Convergence Trade-off.
- A Configurable FPGA Implementation of the Tanh Function using DCT Interpolation.

### 2017 DAC
- **Deep^3: Leveraging Three Levels of Parallelism for Efficient Deep Learning.** (UCSD, Rice)
- **Real-Time meets Approximate Computing: An Elastic Deep Learning Accelerator Design with Adaptive Trade-off between QoS and QoR.** (CAS)
  - *I'm not sure whether the proposed tuning scenario and direction are reasonable enough to find out feasible solutions.*
- **Exploring Heterogeneous Algorithms for Accelerating Deep Convolutional Neural Networks on FPGAs.** (PKU, CUHK, SenseTime)
- **Hardware-Software Codesign of Highly Accurate, Multiplier-free Deep Neural Networks.** (Brown University)
- **A Kernel Decomposition Architecture for Binary-weight Convolutional Neural Networks.** (KAIST)
- **Design of An Energy-Efficient Accelerator for Training of Convolutional Neural Networks using Frequency-Domain Computation.** (Georgia Tech)
- **New Stochastic Computing Multiplier and Its Application to Deep Neural Networks.** (UNIST)
- **TIME: A Training-in-memory Architecture for Memristor-based Deep Neural Networks.** (THU, UCSB)
- **Fault-Tolerant Training with On-Line Fault Detection for RRAM-Based Neural Computing Systems.** (THU, Duke)
- **Automating the systolic array generation and optimizations for high throughput convolution neural network.** (PKU, UCLA, Falcon)
- **Towards Full-System Energy-Accuracy Tradeoffs: A Case Study of An Approximate Smart Camera System.** (Purdue)
  - *Synergistically tunes componet-level approximation knobs to achieve system-level energy-accuracy tradeoffs.*
- **Error Propagation Aware Timing Relaxation For Approximate Near Threshold Computing.** (KIT)
- RESPARC: A Reconfigurable and Energy-Efficient Architecture with Memristive Crossbars for Deep Spiking Neural Networks. (Purdue)
- Rescuing Memristor-based Neuromorphic Design with High Defects. (University of Pittsburgh, HP Lab, Duke)
- Group Scissor: Scaling Neuromorphic Computing Design to Big Neural Networks. (University of Pittsburgh, Duke)
- Towards Aging-induced Approximations. (KIT, UT Austin)
- SABER: Selection of Approximate Bits for the Design of Error Tolerant Circuits. (University of Minnesota, TAMU)
- On Quality Trade-off Control for Approximate Computing using Iterative Training. (SJTU, CUHK)

### 2017 DATE
- **DVAFS: Trading Computational Accuracy for Energy Through Dynamic-Voltage-Accuracy-Frequency-Scaling.** (KU Leuven)
- **Accelerator-friendly Neural-network Training: Learning Variations and Defects in RRAM Crossbar.** (Shanghai Jiao Tong University, University of Pittsburgh, Lynmax Research)
- **A Novel Zero Weight/Activation-Aware Hardware Architecture of Convolutional Neural Network.** (Seoul National University)
  - *Solve the zero-induced load imbalance problem.*
- **Understanding the Impact of Precision Quantization on the Accuracy and Energy of Neural Networks.** (Brown University)
- **Design Space Exploration of FPGA Accelerators for Convolutional Neural Networks.** (Samsung, UNIST, Seoul National University)
- **MoDNN: Local Distributed Mobile Computing System for Deep Neural Network.** (University of Pittsburgh, George Mason University, University of Maryland)
- **Chain-NN: An Energy-Efficient 1D Chain Architecture for Accelerating Deep Convolutional Neural Networks.** (Waseda University)
- **LookNN: Neural Network with No Multiplication.** (UCSD)
  - *Cluster weights and use LUT to avoid multiplication.*
- Energy-Efficient Approximate Multiplier Design using Bit Significance-Driven Logic Compression. (Newcastle University)
- Revamping Timing Error Resilience to Tackle Choke Points at NTC Systems. (Utah State University)

### 2017 VLSI
- **A 3.43TOPS/W 48.9pJ/Pixel 50.1nJ/Classification 512 Analog Neuron Sparse Coding Neural Network with On-Chip Learning and Classification in 40nm CMOS.** (University of Michigan, Intel)
- **BRein Memory: A 13-Layer 4.2 K Neuron/0.8 M Synapse Binary/Ternary Reconfigurable In-Memory Deep Neural Network Accelerator in 65 nm CMOS.** (Hokkaido University, Tokyo Institute of Technology, Keio University)
- **A 1.06-To-5.09 TOPS/W Reconfigurable Hybrid-Neural-Network Processor for Deep Learning Applications.** (Tsinghua University)
- **A 127mW 1.63TOPS sparse spatio-temporal cognitive SoC for action classification and motion tracking in videos.** (University of Michigan)

### 2017 ICCAD
- **AEP: An Error-bearing Neural Network Accelerator for Energy Efficiency and Model Protection.** (University of Pittsburgh)
- VoCaM: Visualization oriented convolutional neural network acceleration on mobile system. (George Mason University, Duke)
- AdaLearner: An Adaptive Distributed Mobile Learning System for Neural Networks. (Duke)
- MeDNN: A Distributed Mobile System with Enhanced Partition and Deployment for Large-Scale DNNs. (Duke)
- TraNNsformer: Neural Network Transformation for Memristive Crossbar based Neuromorphic System Design. (Purdue).
- A Closed-loop Design to Enhance Weight Stability of Memristor Based Neural Network Chips. (Duke)
- Fault injection attack on deep neural network. (CUHK)
- ORCHARD: Visual Object Recognition Accelerator Based on Approximate In-Memory Processing. (UCSD)

### 2017 HotChips
- **A Dataflow Processing Chip for Training Deep Neural Networks.** (Wave Computing)
- **Brainwave: Accelerating Persistent Neural Networks at Datacenter Scale.** (Microsoft)
- **DNN ENGINE: A 16nm Sub-uJ Deep Neural Network Inference Accelerator for the Embedded Masses.** (Harvard, ARM)
- **DNPU: An Energy-Efficient Deep Neural Network Processor with On-Chip Stereo Matching.** (KAIST)
- **Evaluation of the Tensor Processing Unit (TPU): A Deep Neural Network Accelerator for the Datacenter.** (Google)
- NVIDIA’s Volta GPU: Programmability and Performance for GPU Computing. (NVIDIA)
- Knights Mill: Intel Xeon Phi Processor for Machine Learning. (Intel)
- XPU: A programmable FPGA Accelerator for diverse workloads. (Baidu)

### 2017 MICRO
- **Bit-Pragmatic Deep Neural Network Computing.** (NVIDIA, University of Toronto)
- **CirCNN: Accelerating and Compressing Deep Neural Networks Using Block-Circulant Weight Matrices.** (Syracuse University, City University of New York, USC, California State University, Northeastern University)
- **DRISA: A DRAM-based Reconfigurable In-Situ Accelerator.** (UCSB, Samsung)
- **Scale-Out Acceleration for Machine Learning.** (Georgia Tech, UCSD)
  - Propose CoSMIC, a full computing stack constituting language, compiler, system software, template architecture, and circuit generators, that enable programmable acceleration of learning at scale.
- DeftNN: Addressing Bottlenecks for DNN Execution on GPUs via Synapse Vector Elimination and Near-compute Data Fission. (Univ. of Michigan, Univ. of Nevada)
- Data Movement Aware Computation Partitioning. (PSU, TOBB University of Economics and Technology)
  - *Partition computation on a manycore system for near data processing.*

## Important Topics
This is a collection of papers on other important topics related to neural networks. Papers of significance are marked in **bold**. My comments are in marked in *italic*.

### Tutorial and Survey
- [Tutorial on Hardware Architectures for Deep Neural Networks.](http://eyeriss.mit.edu/tutorial.html) (MIT)
- [A Survey of Neuromorphic Computing and Neural Networks in Hardware.](https://arxiv.org/abs/1705.06963) (Oak Ridge National Lab)
- [A Survey of FPGA Based Neural Network Accelerator.](https://arxiv.org/abs/1712.08934) (Tsinghua)
- [Toolflows for Mapping Convolutional Neural Networks on FPGAs: A Survey and Future Directions.](https://arxiv.org/abs/1803.05900) (Imperial College London)

