# SCAMP-5 Hardware Facts, Abstractions, and Proxy Assumptions

Date: 2026-09-30  
Scope: cost modelling for the MNIST software project. No value in this document is a chip measurement from this project.

## Measurement boundary

The checked-in models run on CPU/GPU. Accuracy, F1, tensor values, and software checkpoint statistics are directly measured in software. Operation counts are analytical counts. Latency and energy values produced by `hardware_model/` are normalized proxies and must never be reported as SCAMP-5 cycles, seconds, joules, watts, or FPS.

## A. Known hardware facts

The following facts are stated by the SCAMP5d programming guide and University of Manchester material:

- SCAMP-5 uses a 256 by 256 pixel-processor array with SIMD-style array operations.
- A processing element has local analogue and digital state and communicates with its four NEWS neighbours.
- The programming guide describes seven writable analogue registers (A-F and NEWS), thirteen digital registers (R0-R12), and FLAG control state per processing element.
- Analogue values are approximate, noisy, and subject to decay; the guide describes an approximately 8-bit-like useful dynamic range rather than exact digital arithmetic.
- Many analogue instructions take approximately two clock cycles, while macro instructions can expand to multiple lower-level instruction-control words. This is not sufficient to infer end-to-end latency without the final instruction schedule.
- Instruction and analogue-instruction-control storage is finite.

Primary source: SCAMP5d Programming Guide, https://scamp.gitlab.io/scamp5d_doc/Scamp5d_Programming_Guide.pdf

Supporting source: FAST-based Feature Extraction in a Binocular SCAMP5d System, https://research.manchester.ac.uk/files/133783395/ICIP19_FAST_extraction_SCAMP5d.pdf

## B. Literature-supported mappings

- Published SCAMP-5 CNN work maps convolution-like processing to in-pixel weight/mask storage, local shifts and additions, pooling, and on-chip classification operations.
- The original firmware included in this repository maps sixteen 64 by 64 feature regions onto the 256 by 256 PPA, performs the sixteen positions of a 4 by 4 stencil as sequential tap stages, uses NEWS/DNEWS-style local routing, and represents positive and negative classifier contributions with masks and sparse global sums.
- Consequently, the number of retained stencil offsets and the routing between them are more relevant to a sequential-depth proxy than the global percentage of individual zero weights.

Primary paper: Convolutional Neural Networks for Image Interpretation Directly on the Sensor, https://arxiv.org/abs/2007.08236

Repository evidence: `original_repo/MAIN_MNIST_SINGLE_LAYER_16.cpp` and its associated weight/register headers.

## C. Reasonable software abstractions

These abstractions are defensible for comparison but are not literal hardware simulation:

1. A ternary +1 term is counted as an addition, a -1 term as a subtraction, and a zero term as a skipped scalar candidate term.
2. Convolution tap depth is the count of 4 by 4 spatial offsets that contain at least one non-zero weight across all simultaneously mapped filters. An offset can be removed from the schedule only when the whole offset group is zero.
3. Offset-routing depth is approximated by Manhattan distance between successively scheduled offsets. The default order is row-major; alternate schedules may be compared explicitly.
4. Feature traffic is expressed as logical element- or bit-equivalent movement. Local neighbour movement, feature ingress/egress, mask loading, and layer-boundary materialization are reported separately.
5. Pool reduction depth is approximated by a configurable staged local reduction, not by CPU max-pool runtime.
6. Classifier depth counts non-empty positive/negative class-mask passes plus readout stages.
7. Register pressure is a normalized live-plane proxy relative to the documented analogue/digital register counts. It is not a register allocator or proof that a schedule fits.

## D. Explicit proxy assumptions

All values below are configuration parameters in `hardware_model/config.py` and are included in benchmark output:

- Logical tensor widths: input/activation 8 bits, ternary signs 2 bits, scale/bias metadata 16 bits. These are accounting widths, not claims about analogue precision.
- Baseline live-plane estimate: three analogue planes and four digital planes. Extra planes can be supplied by an architecture or schedule analyser.
- Pool schedule depth defaults to four stages for a 4 by 4 window.
- Arithmetic, movement, register-access, and control coefficients used by the energy proxy are dimensionless. Several sensitivity profiles are provided; no single profile is presented as calibrated hardware energy.
- Composite hardware cost is a weighted sum of metrics normalized to a named reference model. Default weights are equal and sensitivity profiles reweight compute, movement, storage, and schedule costs.
- A spill penalty is applied only when the assumed live-plane count exceeds the documented register count; it is a warning proxy, not a predicted spill instruction count.

## Interpretation rules

- Report `active_terms`, `skipped_terms`, and `tap_depth` separately.
- A lower active scalar count without a lower tap depth is an activity/energy opportunity, not automatically a latency improvement.
- A lower parameter count is not automatically a lower data-movement cost.
- CPU/GPU wall-clock runtime is never labelled SCAMP latency.
- Only normalized ratios against a fixed reference may be compared across methods.
- Conclusions must remain stable across the predefined cost-weight sensitivity profiles before being described as robust.

## Secondary overview

The following review is useful context but is not used to override primary documentation: https://pmc.ncbi.nlm.nih.gov/articles/PMC8321119/
