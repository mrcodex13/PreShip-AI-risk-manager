"""Relational features module: graph-based risk signals (extensible framework)."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging


logger = logging.getLogger(__name__)


class RelationalFeatureExtractor:
    """
    Extract relational/graph-based features from order data.
    
    IMPORTANT: This module is designed as an extension point.
    The current CSV lacks customer identifiers (address, phone, device, IP),
    so all functions are no-ops. When these identifiers are added to the dataset,
    uncomment the real implementations below.
    
    See FINAL_REPORT.md Section 10 for the priority to add these identifiers.
    """
    
    REQUIRED_COLUMNS = {
        'velocity': ['Customer_ID', 'Order_Date'],
        'duplicate_addresses': ['Customer_ID', 'Address'],
        'duplicate_phones': ['Customer_ID', 'Phone'],
        'first_time_buyer': ['Customer_ID', 'Order_Date'],
        'high_value_cod': ['Order_Value', 'Payment_Method']
    }
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize extractor and check for required columns.
        
        Args:
            data: full dataset
        """
        self.data = data
        self.missing_columns = self._check_required_columns()
        
        if self.missing_columns:
            logger.warning(
                f"Relational features unavailable — dataset lacks identifiers: {self.missing_columns}. "
                "To unlock these high-value signals, the dataset must include: "
                "Customer_ID, Order_Date, Address, Phone, Device_ID, and IP_Address."
            )
    
    def _check_required_columns(self) -> List[str]:
        """Check which identifier columns are missing."""
        all_required = set()
        for feature, cols in self.REQUIRED_COLUMNS.items():
            all_required.update(cols)
        
        missing = [col for col in all_required if col not in self.data.columns]
        return missing
    
    def extract_all(self) -> pd.DataFrame:
        """
        Extract all available relational features.
        
        Returns:
            DataFrame with columns: order_velocity, duplicate_address_count, 
                                   duplicate_phone_count, is_first_time_buyer, 
                                   is_high_value_cod
        """
        features = {
            'order_velocity': self.order_velocity(),
            'duplicate_address_count': self.duplicate_address_count(),
            'duplicate_phone_count': self.duplicate_phone_count(),
            'is_first_time_buyer': self.is_first_time_buyer(),
            'is_high_value_cod': self.is_high_value_cod()
        }
        
        # Filter to available features
        available = {k: v for k, v in features.items() if v is not None}
        
        if not available:
            # Return a no-op dataframe with all NaN
            return pd.DataFrame({
                'order_velocity': np.nan,
                'duplicate_address_count': np.nan,
                'duplicate_phone_count': np.nan,
                'is_first_time_buyer': np.nan,
                'is_high_value_cod': np.nan
            }, index=self.data.index)
        
        return pd.DataFrame(available, index=self.data.index)
    
    def order_velocity(self) -> Optional[pd.Series]:
        """
        EXTENSION POINT: Orders per customer in past 7 days.
        
        Returns None if Customer_ID or Order_Date missing.
        """
        required = ['Customer_ID', 'Order_Date']
        if not all(col in self.data.columns for col in required):
            return None
        
        # TODO: Uncomment when data is available
        # self.data['Order_Date'] = pd.to_datetime(self.data['Order_Date'])
        # velocity = self.data.groupby('Customer_ID').size()
        # return self.data['Customer_ID'].map(velocity)
        
        return None
    
    def duplicate_address_count(self) -> Optional[pd.Series]:
        """
        EXTENSION POINT: Count of customers at same address.
        
        High count = risky indicator (fraud ring, bot, etc).
        Returns None if Address missing.
        """
        if 'Address' not in self.data.columns:
            return None
        
        # TODO: Uncomment when data is available
        # return self.data['Address'].map(self.data['Address'].value_counts())
        
        return None
    
    def duplicate_phone_count(self) -> Optional[pd.Series]:
        """
        EXTENSION POINT: Count of customers with same phone.
        
        Returns None if Phone missing.
        """
        if 'Phone' not in self.data.columns:
            return None
        
        # TODO: Uncomment when data is available
        # return self.data['Phone'].map(self.data['Phone'].value_counts())
        
        return None
    
    def is_first_time_buyer(self) -> Optional[pd.Series]:
        """
        EXTENSION POINT: First order from this customer.
        
        Returns None if Customer_ID or Order_Date missing.
        """
        required = ['Customer_ID', 'Order_Date']
        if not all(col in self.data.columns for col in required):
            return None
        
        # TODO: Uncomment when data is available
        # self.data['Order_Date'] = pd.to_datetime(self.data['Order_Date'])
        # first_order = self.data.groupby('Customer_ID')['Order_Date'].transform('min')
        # return self.data['Order_Date'] == first_order
        
        return None
    
    def is_high_value_cod(self) -> Optional[pd.Series]:
        """
        EXTENSION POINT: High-value order + Cash on Delivery.
        
        Returns None if Order_Value or Payment_Method missing.
        """
        required = ['Order_Value', 'Payment_Method']
        if not all(col in self.data.columns for col in required):
            return None
        
        # TODO: Uncomment when data is available
        # high_value_threshold = self.data['Order_Value'].quantile(0.75)
        # is_cod = self.data['Payment_Method'].str.lower() == 'cod'
        # return (self.data['Order_Value'] > high_value_threshold) & is_cod
        
        return None


def get_relational_features_documentation() -> str:
    """Return a detailed explanation of relational features for the UI."""
    return (
        "**Relational Features Status: UNAVAILABLE**\n\n"
        "The current dataset lacks customer identifiers (address, phone, device IP) needed to compute:\n"
        "- Order velocity (orders from same customer in past 7 days)\n"
        "- Duplicate addresses (accounts using same address)\n"
        "- Duplicate phones (accounts using same phone)\n"
        "- First-time buyer flag\n"
        "- High-value + COD combination flag\n\n"
        "**This is the single highest-value addition for a production version.** "
        "These signals are among the strongest fraud indicators. "
        "See FINAL_REPORT.md Section 10 for implementation priority."
    )
