import numpy as np

def add_features(envelope, wanted_features):
    features = []

    if('duration' in wanted_features):
        durs = len(envelope) / 100
        features.append(durs)

    if('sonority_total_energy' in wanted_features):
        en_sonor_total = np.sum(envelope)
        features.append(en_sonor_total)
    
    if('sonority_mean_energy' in wanted_features):
        en_sonor_mean = np.mean(envelope)
        features.append(en_sonor_mean)
    
    if('sonority_SD_energy' in wanted_features):
        en_sonor_sd = np.std(envelope)
        features.append(en_sonor_sd)
    
    return features