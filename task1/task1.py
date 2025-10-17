import pandas as pd
import numpy as np

class FuturesRollAdjuster:
    def __init__(self, data_file='ES.h5'):
        self.data_file = data_file
        self.tick_data = None
        self.continuous_data = None
        
    def explore_data(self):
        """Explore file structure and get basic info without loading all data"""
        print(f"Exploring data from {self.data_file}")
        
        # Read just a small sample to understand structure
        sample = pd.read_hdf(self.data_file, 'tick/trades', start=0, stop=1000)
        print(f"Columns: {sample.columns.tolist()}")
        print(f"Sample data:\n{sample.head()}")
        print(f"Data types:\n{sample.dtypes}")
        
        # Get total number of rows
        with pd.HDFStore(self.data_file, 'r') as store:
            total_rows = store.get_storer('tick/trades').nrows
            print(f"Total rows in file: {total_rows:,}")
        
        return sample
        
    def load_data_chunked(self, chunk_size=100000):
        """Generator to load data in chunks"""
        print(f"Loading data in chunks of {chunk_size:,}")
        
        for chunk in pd.read_hdf(self.data_file, 'tick/trades', chunksize=chunk_size):
            yield chunk
            
    def load_sample_data(self, n_rows=50000):
        """Load a sample of data for quick analysis"""
        print(f"Loading sample of {n_rows:,} rows")
        self.tick_data = pd.read_hdf(self.data_file, 'tick/trades', start=0, stop=n_rows)
        print(f"Loaded {len(self.tick_data):,} tick records")
        return self.tick_data
        
    def identify_roll_dates(self):
        """
        Identify contract roll dates by detecting contract symbol changes
        and volume shifts between contracts
        """
        if self.tick_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Look for changes in Instrument column
        unique_contracts = self.tick_data['Instrument'].unique()
        print(f"Unique contracts: {unique_contracts}")

        # Find where contract changes occur
        contract_changes = self.tick_data['Instrument'] != self.tick_data['Instrument'].shift(1)
        roll_indices = self.tick_data[contract_changes].index.tolist()
        
        # Get the actual roll dates (times)
        roll_dates = self.tick_data.loc[roll_indices, 'Time'].tolist()
        
        print(f"Found {len(roll_dates)} roll dates at indices: {roll_indices[:10]}...")
        print(f"Roll dates: {roll_dates[:5]}...")
        
        return roll_indices, roll_dates
        
    def calculate_roll_adjustments(self, roll_indices):
        """
        Calculate price differences at each roll point
        """
        adjustments = []
        
        for i, roll_idx in enumerate(roll_indices):
            if roll_idx == 0:  # Skip first index (no previous contract)
                continue
                
            # Get last price of expiring contract (price just before roll)
            old_price = self.tick_data.loc[roll_idx - 1, 'Price']
            
            # Get first price of new contract (price at roll point)
            new_price = self.tick_data.loc[roll_idx, 'Price']
            
            # Calculate adjustment = new_price - old_price
            adjustment = new_price - old_price
            adjustments.append((roll_idx, adjustment))
            
            print(f"Roll {i}: {self.tick_data.loc[roll_idx-1, 'Instrument']} -> {self.tick_data.loc[roll_idx, 'Instrument']}, "
                  f"Price: {old_price:.2f} -> {new_price:.2f}, Adjustment: {adjustment:.2f}")
        
        return adjustments
        
    def apply_backward_adjustment(self, adjustments):
        """
        Apply cumulative backward adjustments to create continuous series
        """
        if self.tick_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        self.continuous_data = self.tick_data.copy()
        cumulative_adjustment = 0
        
        # Sort adjustments by index (newest first for backward adjustment)
        sorted_adjustments = sorted(adjustments, key=lambda x: x[0], reverse=True)
        
        # Apply backward adjustments
        for roll_idx, adjustment in sorted_adjustments:
            # Add cumulative adjustment to all prices before this roll point
            self.continuous_data.loc[:roll_idx-1, 'Price'] += cumulative_adjustment
            cumulative_adjustment += adjustment
            
        print(f"Applied {len(adjustments)} roll adjustments")
        return self.continuous_data
        
    def create_continuous_series(self, use_sample=True, sample_size=50000):
        """Main method to create roll-adjusted continuous price series"""
        if use_sample:
            self.load_sample_data(sample_size)
        # For full dataset processing, implement chunked approach
        
        roll_indices, roll_dates = self.identify_roll_dates()
        adjustments = self.calculate_roll_adjustments(roll_indices)
        return self.apply_backward_adjustment(adjustments)

def main():
    roll_adjuster = FuturesRollAdjuster()
    
    # First explore the data structure
    roll_adjuster.explore_data()
    
    # For processing large dataset, use chunked approach:
    # for chunk in roll_adjuster.load_data_chunked(100000):
    #     # Process each chunk
    #     pass

if __name__ == "__main__":
    main()