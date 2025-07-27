Overview

Get ready to rev your engines and race to the subsurface! This challenge puts you in the driver's seat to develop cutting-edge techniques for seismic velocity inversion. Your mission? To construct high-resolution models of subsurface velocities from seismic data, unlocking valuable insights into the Earth's hidden structures. Just like tuning a high-performance machine, you'll need to expertly handle complex data and innovative algorithms to achieve top speeds and accuracy in revealing the subterranean velocity landscape. This challenge will test your ability to transform raw seismic signals into detailed velocity models, a crucial step in various applications, from resource exploration to hazard assessment.
Background

Imagine trying to understand the intricate workings of an engine just by listening to the sounds it makes. That's a bit like trying to understand the Earth's subsurface by analyzing seismic waves. Seismic velocity inversion is the process of taking seismic data, which records the echoes of sound waves traveling through the Earth, and using it to estimate the velocity at which these waves propagate through different underground materials (Figure 1). This velocity information is fundamental for characterizing subsurface properties, identifying geological formations, and even locating valuable resources.

However, this task isn't a straightforward quarter mile sprint. Traditional methods like Full Waveform Inversion (FWI), while powerful for high-resolution reconstruction, often face hurdles such as the cycle-skipping issue, where simulations get out of sync with observations, leading to inaccurate results. Furthermore, FWI can be computationally intensive and relies heavily on a good initial velocity model. The band-limited nature of seismic data and limitations in how the data is collected also make velocity inversion an ill-posed inverse problem, meaning there can be multiple velocity models that fit the observed data. To tackle this, geophysicists often employ regularization techniques to guide the inversion towards more plausible solutions by incorporating constraints like model smoothness or prior knowledge. Integrating various types of data, such as well logging information, can also provide complementary insights and improve the accuracy of the inversion.
speednstructureoverview.png

Figure 1. An example of five seismic shot records, and corresponding subsurface velocity structure model. These examples serve to illustrate the type of data provided for the challenge and the relationship between seismic signals and the underlying subsurface velocities.

In recent years, a new class of techniques has emerged, leveraging the incredible learning capabilities of deep learning. Deep learning methods can learn complex relationships directly from data and have shown promise in efficiently estimating subsurface velocities.

This challenge is your opportunity to innovate! Here are some potential avenues you might explore, drawing inspiration from the latest research:

    Data-Driven Velocity Estimation with Deep Learning: Consider developing deep learning models that can directly predict velocity models from seismic data. Architectures like encoder-decoder networks or even more advanced generative models could be effective.

    Generative Diffusion Models: Explore the use of Generative Diffusion Models (GDMs), a state-of-the-art generative technique, for velocity inversion. As highlighted in recent work, GDMs can be conditioned on various inputs to generate velocity models that adhere to different constraints.

        You could develop conditional GDMs that take seismic data as input to generate velocity models consistent with the observed wave propagation.

        Another exciting direction is to integrate multiple sources of information into the GDM framework. This could involve:

            Incorporating background velocity models to guide the generation process, potentially by using them as a starting point or by combining their low-frequency components with the GDM output.

            Leveraging geological knowledge by training an unconditional GDM on a diverse set of velocity models representing different geological scenarios. This geology-oriented GDM could then be combined with the seismic-data GDM to improve the generalization and geological plausibility of the inverted models.

            Integrating high-resolution but spatially sparse well log data by training a conditional GDM on well log information. This well-log GDM could then be combined with the seismic-data GDM to ensure the generated velocity models honor the available well data. The combination of these GDMs could be achieved through weighted summation during the sampling process, allowing flexible control over the influence of each information source. You would have to make your own synthetic well log data from the training data if you choose to follow this path. 

    Hybrid Approaches: Consider combining traditional FWI with deep learning techniques. For example, a DL model could be used to generate a better starting model for FWI, or to regularize the FWI process.

    Low-Frequency Enhancement: Given the importance of low frequencies for mitigating cycle-skipping, you might explore DL techniques for extrapolating low frequencies from band-limited seismic data before performing velocity inversion.

The key to success will be developing robust and efficient algorithms that can accurately and rapidly estimate subsurface velocities, potentially by effectively integrating diverse data sources.
Data

For this challenge, we are providing you with 2,000 training samples of seismic shot records and paired ground truth velocity data. Use these sample to train your model. To evaluate your model's performance you will make predictions for the 150 test samples and upload the results to see if you are in the pole position on the predictive leaderboard. The synthetic data is delivered as Numpy arrays with a shape of (300,1259). You are free to use any subset of the data that you choose for training.
Evaluation

To evaluate the performance of your solution, submit an .npz file containing sample IDs and their corresponding predictions. Each prediction should be a 2D NumPy array representing the velocity model for that sample. Detailed instructions and submission file generation code are provided in the starter notebook.

For each sample in the test dataset, the mean absolute percentage error (MAPE) will be calculated between your prediction and the ground-truth velocity model. These MAPEs will then be averaged across all samples to determine the total MAPE for the test dataset. This total MAPE will be used to rank your solution on the predictive leaderboard for this challenge.
Final Evaluation

For the Final Evaluation, the top submissions on the Predictive Leaderboard will be invited to send ThinkOnward Challenges their fully reproducible Python code to be reviewed by a panel of judges. The judges will run a submitted algorithm on up to an AWS SageMaker g5.12xlarge instance, and inference must run within 1 hour on a hold out dataset with a similar distribution as the test dataset. The evaluation metric used for the Predictive Leaderboard will be used to score final submissions on an unseen hold out dataset. The score on this hold out dataset will determine 90% of your final score. The remaining 10% of the final score will assess submissions on the interpretability of their submitted Jupyter Notebook. The interpretability criterion focuses on the degree of documentation (i.e., docstrings and markdown), clearly stating variables, and reasonably following standard Python style guidelines. For our recommendations on what we are looking for on interpretability see our example GitHub repository (link). Additionally, the Starter Notebook contains important points, guidelines, and requirements to help you understand what we are looking for in terms of interpretability.

A successful final submission must contain the following:

    Jupyter Notebook: Your notebook will be written in Python and clearly outline the steps in your pipeline

    Requirements.txt file: This file will provide all required packages we need to run the training and inference for your submission

    Supplemental data or code: Include any additional data, scripts, or code that are necessary to accurately reproduce your results

    Model Checkpoints (If applicable): Include model checkpoints created during training so we can replicate your results

    Software License: An open-source license to accompany your code

Your submission must contain the libraries and their versions and the Python version (>=3.12).  See the Starter Notebook on the Data Tab for an example.
Timelines and Prizes

Challenge will open on 14 May 2025 and close at 22:00 UTC 7 August 2025. Winners will be announced on 4 September 2025. 

The main prize pool will have prizes awarded for the first ($14,000), second ($8,000), and third ($6,000) in the final evaluation. There will be two $1,000 honorable mentions for valid submissions that take novel approaches to solving the problem.