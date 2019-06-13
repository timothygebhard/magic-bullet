# CNNs: A Magic Bullet for GW Detection?

[![Python](https://img.shields.io/badge/Python-3.6-yellow.svg)]()
[![CodeFactor](https://www.codefactor.io/repository/github/timothygebhard/magic-bullet/badge)](https://www.codefactor.io/repository/github/timothygebhard/magic-bullet)
[![GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/timothygebhard/magic-bullet/blob/master/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-1904.08693-red.svg)](https://arxiv.org/abs/1904.08693)
[![DOI](https://zenodo.org/badge/180809281.svg)](https://zenodo.org/badge/latestdoi/180809281)

This repository contains all scripts that were used to produce the results presented in [*Convolutional neural networks: a magic bullet for gravitational-wave detection?*](https://arxiv.org/abs/1904.08693) by Gebhard et al. (2019) [arXiv:1904.08693].



## Reproducing the analysis

In case you want to reproduce our analysis, you will need to run the following steps:

1. Create a `training`, `validation` and `testing` dataset. We suggest using the [ggwd repository](<https://github.com/timothygebhard/ggwd>), which contains a collection of useful scripts to generate synthetic gravitational-wave data. If you also want to test the performance of the trained network on real, confirmed GW events, you also want to generate a `real_events` dataset. See the repository documentation for more information.

2. Adjust the values in the `CONFIG.json` file to match your setup (e.g., set up the paths for the datasets).

3. Ensure your Python environment has all the necessary dependencies specified in `requirements.txt`, for example, by running the following:

   ```
   pip install -r requirements.txt
   ```

   We would recommend to use a fresh virtual environment for this. Also, we require at least Python 3.6.

4. Train the fully convolutional neural network model with PyTorch by running:

   ```
   python train_model.py
   ```

   During training, the current best set of weights for the network will be saved in a file `best.pth` in the `./checkpoints` directory. This checkpoint will also later be used when applying the model to make predictions (see below).

   ---

   **Warning:** Depending on your setup, training the model as described in the paper can take quite a while (and use a lot of memory)! For reference: Even when using 5 Tesla V-100 GPUs (with 32 GB of memory each), it took us over 30 hours to train the network for 64 epochs. 
   To reduce the training time, you can decrease the number of channels by modifying the definition of the `FCNN` class in `./utils/models.py`. This may then of course also decrease the network's performance.

   ---

   **Note:** You can monitor the training progress using [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard). The necessary log files are stored in the `./tensorboard` directory. To start TensorBoard, you will need to run the following line:

   ```
   tensorboard --logdir=tensorboard --port=<port>
   ```

   ---

5. Apply the network to the testing dataset to make predictions:

   ```
   python apply_model.py --apply-to=testing
   ```

   This will create a file `predictions_testing.hdf` in the `./results` directory. If you have also prepared a dataset containing real events, you can analogously apply the trained network to these by running:

   ```
   python apply_model.py --apply-to=real_events
   ```

   This will then also output another file, `predictions_real_events.hdf`, to the `./results` directory.

6. Next, you need to post-process the predictions on the testing dataset to count the number of detections (i.e., successfully recovered injections) and the number of false positives. To this end, run the following script:

   ```
   python find_triggers.py
   ```

   This script will count the detections and false positives for different values of ∆t, that is, the width of the interval around the ground truth injection time within which a prediction is still counted as a detection. The values for this can be defined in the `CONFIG.json` file, where this parameter is called `slack_width` (because ∆t seemed a little generic).

   The `find_triggers.py` script creates a file `found_triggers.hdf` in the `./results` directory, containing a group of results for each value of ∆t. Each such group again contains three groups: 

   * `figures_of_merit`: The attributes of this group hold the global values (i.e., averaged over all injection SNRs) for the detection ratio, the false alarm ratio, the (inverse) false positive rate, as well as the mean and standard deviation for the deviation between the predicted event time and the ground truth injection time. These values are also written to the command line by `find_triggers.py`.
   * `injection`: Results for examples that _do_ contain an injection, consisting of three data sets, namely `detected`, `false_positives` and `injection_snr`.
   * `noise`: Results for examples that _do not_ contain an injection, consisting of only one data set, namely `false_positives`.

7. Now you can compute the detection ratio as a function of the injection SNR, and check how the post-processing parameters (smoothing window size and thresholding value) affect the detection ratio and the (inverse) false positive rate. To this end, run the following two scripts:

   ```
   python compute_dr_over_snr.py
   python compute_dr_over_ifpr.py
   ```

   This will create two files in the `./results` directory, named `dr_over_snr.json` and `dr_over_ifpr.json`.

8. Using these JSON files (as well as the `found_triggers.hdf` file from step 6), you can then plot the results from the last step by running the following:

   ```
   python plot_dr_over_snr.py
   python plot_dr_over_ifpr.py
   python plot_ifpr_over_delta_t.py
   ```

   The resulting plots will be saved as PDF files in the `./plots` directory.

9. Independently of the last three steps, you can also simply plot the predictions of your trained model on the dataset containing real events by running:

   ```
   python plot_real_events.py
   ```

   The results are again stored in the `./plots` directory.

10. Finally, you can try to check "what you network has really learned", by looking for find preimages for a given target output through optimization. This is related to the concept of *adversarial examples*, and serves to illustrate that not every input that causes the network to predict a signal necessarily looks like a signal. To find a preimage for a particular target output, you can use:

   ```
   python find_preimage.py --constraint=<contraint> --index=<N>
   ```

   The `--constraint` parameter can be used to guide the optimization procedure, or impose additional (e.g., unphysical) constraints on the input. For our paper, we used the following `constraint` options: `gw_like`, `minimal_perturbation`, ` positive_strain`, ` zero_coalescence` and `minimal_amplitude`. The `--index` parameter specifies the index of the noise-only example from the testing dataset, which is used as a starting point for the optimization.

   The results of the preimage search are stored in the `./results/preimages` directory. They can be plotted by running:

   ```
   python plot_preimages.py
   ```

   This will create a PDF for every preimage in the `./plots/preimages` directory. Note that you may have to generate multiple preimages to find one that looks as "clear" as the examples presented in our paper.

   

## Using this code

We'd be happy to see you using our code here as a starting point for you own studies on gravitational waves and convolutional neural networks! This is why we are releasing it under a permissive license. We only ask you that if you make use of our code and publish your work, please be sure to cite our paper, and consider letting us know about it! :)
