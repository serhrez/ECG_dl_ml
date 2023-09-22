import json 

with open('utils/snomed_ct_codes.json', 'r') as f:
    snomed_ct_codes = json.load(f)

snomed_ct_codes_reversed = {v: k for k, v in snomed_ct_codes.items()}

def get_snomed(arrhythmia):
    if isinstance(arrhythmia, list):
        return [snomed_ct_codes[a] for a in arrhythmia]
    return snomed_ct_codes[arrhythmia]

def get_arrhythmia(snomed_ct):
    return snomed_ct_codes_reversed[snomed_ct]