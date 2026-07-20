"""
src package
-----------
Speech Emotion Recognition pipeline for the RAVDESS dataset.

Modules:
    config              - paths, label mappings, shared settings
    data_parser         - parses RAVDESS filenames into metadata.csv
    feature_extraction  - extracts MFCC/mel-spectrogram features (+ augmentation)
    augmentation        - audio augmentation functions (pitch/time/noise)
    dataset             - PyTorch Dataset/DataLoader construction
    model               - EmotionCNN architecture
    train               - training loop
    evaluate            - test-set evaluation, confusion matrix, reports
    predict             - single-file inference
"""