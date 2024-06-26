class ProtocolConstants:
    # Feedback types
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    CHARACTERIZATION = "characterization"
    TRANSCRIPTION = "transcription"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    PSUEDO_CLASSIFICATION = "psuedo_labels_classification"
    SCORE = "score"

    # Detection requirement
    REQUIRED = "REQUIRED"
    NOTIFY_AND_CONTINUE = "NOTIFY_AND_CONTINUE"
    IGNORE = "IGNORE"
    SKIP = "SKIP"

    # Hint types
    TYPE_A = "typeA"
    TYPE_B = "typeB"