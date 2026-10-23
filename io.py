import numpy as np
import csv


def load_csv(filepath, delimiter=',', skip_header=1, missing_val=np.nan, dtype=float):
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        if skip_header > 0:
            for _ in range(skip_header):
                next(reader, None)
                
        data = []
        for row in reader:
            parsed_row = []
            for val in row:
                val = val.strip()
                if val == '' or val.lower() == 'nan' or val.lower() == 'null':
                    parsed_row.append(missing_val)
                else:
                    try:
                        parsed_row.append(float(val))
                    except ValueError:
                        parsed_row.append(val)
            data.append(parsed_row)
            
    try:
        return np.array(data, dtype=dtype)
    except ValueError:
        return np.array(data, dtype=object)