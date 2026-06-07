"""Source code for the dynamic Deep-Halo SSM project.

Sub-packages:
  - ``datasets``      : synthetic DGPs and ChoiceDataset adapters.
  - ``choice_models`` : trainable choice-model entities --- MacroDeepHalo,
                        DeepHalo, SimpleMLP, SimpleMNL. These are the
                        objects that ``model.fit(...)`` trains.
  - ``inference``     : inference algorithms that consume a fitted choice
                        model --- bootstrap PF, differentiable PF + Adam
                        joint MLE, both TFP-backed and custom Q2-style.
  - ``validation``    : scripts that validate the pipeline on synthetic
                        data (oracle PF, robustness sweep, cross-impl).
  - ``experiments``   : scripts that run the two main experiments
                        (Stage 2 conditional MLE; end-to-end joint MLE + PF).
"""
