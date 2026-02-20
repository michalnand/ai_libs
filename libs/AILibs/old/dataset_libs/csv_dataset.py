import  numpy 
import csv

class CSVDataset:

    def __init__(self, file_name):
        rows = []

        with open(file_name, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row

            for row in reader:
                rows.append(row)

        # Transpose to check column types
        columns = list(zip(*rows))
        numeric_columns = []

        for col in columns:
            try:
                # Try converting entire column to float
                float_col = [float(cell) for cell in col]
                numeric_columns.append(float_col)
            except ValueError:
                # If any value fails, skip this column
                continue

        # Transpose back to rows
        self.x = numpy.array(list(zip(*numeric_columns)), dtype=numpy.float32)
        self.length = len(self.x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x[idx]